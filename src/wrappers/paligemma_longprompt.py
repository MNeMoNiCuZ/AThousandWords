"""
Paligemma LongPrompt Wrapper

Based on: docs/source_captioners/paligemma_longprompt/batch.py
Model: mnemic/paligemma-longprompt-v1-safetensors

This wrapper implements Paligemma fine-tuned for detailed long-form captions.
Uses fixed "caption en" prompt. Includes text cleanup for better output quality.
"""

from .base import BaseCaptionModel
from typing import List, Dict, Any
from PIL import Image
import torch
import re


from src.features import (
    StripContentsInsideFeature,
    MaximumWordLengthFeature
)

class PaligemmaLongPromptWrapper(BaseCaptionModel):
    """
    Wrapper for Paligemma LongPrompt model.
    """
    
    MODEL_ID = "mnemic/paligemma-longprompt-v1-safetensors"
    
    # Text cleanup settings from source
    MIN_TOKENS = 20
    MAX_RETRIES = 10
    REMOVE_WORDS = ["#", "/", "、", "@", "__", "|", "  ", ";", "~", '"', "*", "^", ",,", "ON DISPLAY:"]
    PRUNE_END = True
    RETRY_WORDS = ["no_parallel"]
    
    def __init__(self, config):
        super().__init__(config)
        self.processor = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    def _load_model(self):
        """Load Paligemma model and processor from HuggingFace Hub."""
        if self.model is not None:
            return
        
        from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
        
        # Use config model_id if available (Group B fix)
        model_id = self.config.get('model_id', self.MODEL_ID)
        print(f"Loading Paligemma LongPrompt model: {model_id}...")
        
        # Load from HuggingFace Hub (force GPU)
        self.model = PaliGemmaForConditionalGeneration.from_pretrained(
            model_id,
            dtype=torch.bfloat16,
        ).to(self.device).eval()
        
        self.processor = AutoProcessor.from_pretrained(model_id)
        
        print(f"Paligemma LongPrompt loaded on {self.device}")
    
    def _prune_text(self, text: str) -> str:
        """Prune text at last period or comma."""
        if not self.PRUNE_END:
            return text
        last_period = text.rfind('.')
        last_comma = text.rfind(',')
        prune_index = max(last_period, last_comma)
        if prune_index != -1:
            return text[:prune_index].strip()
        return text
    
    def _contains_retry_word(self, text: str) -> bool:
        """Check if text contains retry words."""
        return any(word in text for word in self.RETRY_WORDS)
    
    def _remove_unwanted_words(self, text: str) -> str:
        """Remove unwanted characters/words."""
        for word in self.REMOVE_WORDS:
            text = text.replace(word, ' ')
        return text
    
    def _clean_text(self, text: str, args: Dict[str, Any]) -> str:
        """Full text cleanup pipeline."""
        text = self._remove_unwanted_words(text)
        
        # Use Features
        if args.get('strip_contents_inside', True):
            text = StripContentsInsideFeature.apply(text)
            
        text = MaximumWordLengthFeature.apply(text, **args)
        
        text = re.sub(r'[^\x00-\x7F]+', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        if args.get('remove_underscore_tags', True):
            text = self._remove_underscore_tags(text)
            
        return text
    
    def _remove_underscore_tags(self, text: str) -> str:
        """
        Remove words containing underscores (e.g. 'blue_sky', 'looking_at_viewer').
        Specific to preventing booru-style tags from polluting natural language captions.
        """
        words = text.split()
        cleaned_words = [word for word in words if '_' not in word]
        return ' '.join(cleaned_words)
    
    def _run_inference(self, images: List[Image.Image], prompt: List[str], args: Dict[str, Any]) -> List[str]:
        """
        Run Paligemma inference on images.
        """
        max_tokens = args.get('max_tokens', 256)
        repetition_penalty = args.get('repetition_penalty', 1.15)
        temperature = args.get('temperature', 0.7)
        top_k = args.get('top_k', 50)
        
        # Prepare batch prompts
        # Use prompt from args (task_prompt) or default fallback per image
        prompts = []
        for p in prompt:
             prompts.append(p if p else "<image>caption en")
        
        results = []
        
        try:
            # Batch encoding
            inputs = self.processor(
                text=prompts,
                images=images,
                return_tensors="pt",
                padding=True
            ).to(self.model.device)
            
            input_len = inputs["input_ids"].shape[-1]
            
            # Batch generation
            with torch.inference_mode():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    do_sample=True,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=0.9,
                    no_repeat_ngram_size=2,
                    repetition_penalty=repetition_penalty
                )
            
            # Decode all outputs
            generated_ids = outputs[:, input_len:]
            decoded_list = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
            
            # Post-process results
            results = []
            for decoded in decoded_list:
                pruned = self._prune_text(decoded)
                cleaned = self._clean_text(pruned, args)
                results.append(cleaned)
                
        except Exception as e:
            print(f"Error during PaliGemma batch inference: {e}")
            results = [""] * len(images)
        
        return results
    
    def unload(self):
        """Free model resources using shared utility."""
        from src.core.model_utils import unload_model, UnloadMode
        unload_model(self.model, self.processor, UnloadMode.STANDARD)
        self.model = None
        self.processor = None

