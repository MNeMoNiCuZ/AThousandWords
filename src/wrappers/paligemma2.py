"""
PaliGemma2 Wrapper

Based on: docs/source_captioners/paligemma2/batch.py
Model: google/paligemma2-10b-ft-docci-448

This wrapper implements PaliGemma2 with extensive text cleanup and quality control.
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

class Paligemma2Wrapper(BaseCaptionModel):
    """
    Wrapper for google/paligemma2-10b-ft-docci-448 model.
    """
    
    MODEL_ID = "google/paligemma2-10b-ft-docci-448"

    
    def __init__(self, config):
        """Initialize the wrapper."""
        super().__init__(config)
        self.processor = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    def _load_model(self):
        """Load PaliGemma2 model and processor."""
        if self.model is not None:
            return
        
        from transformers import PaliGemmaProcessor, PaliGemmaForConditionalGeneration
        
        # Use config model_id if available or fallback to class constant
        model_id = self.config.get('model_id', self.MODEL_ID)
        print(f"Loading PaliGemma2 model: {model_id}...")
        
        self.model = PaliGemmaForConditionalGeneration.from_pretrained(
            model_id,
            dtype=torch.bfloat16,
            device_map="auto"
        ).eval()
        
        self.processor = PaliGemmaProcessor.from_pretrained(model_id, use_fast=True)
        
        print(f"PaliGemma2 loaded on {self.device}")
    
    def _prune_text(self, text: str, prune_end: bool) -> str:
        """Prune text to last period or comma."""
        if not prune_end:
            return text
        last_period_index = text.rfind('.')
        last_comma_index = text.rfind(',')
        prune_index = max(last_period_index, last_comma_index)
        if prune_index != -1:
            return text[:prune_index].strip()
        return text
    
    def _remove_unwanted_words(self, text: str, remove_words: List[str]) -> str:
        """Remove unwanted words and characters."""
        for word in remove_words:
            text = text.replace(word, ' ')
        return text
    
    
    def _remove_underscore_tags(self, text: str) -> str:
        """Remove words containing underscores (booru tags)."""
        if not text:
            return text
        return ' '.join([word for word in text.split() if '_' not in word])

    def _clean_text(self, text: str, args: Dict[str, Any]) -> str:
        """Apply all text cleanup functions."""
        # 1. Remove specific unwanted words (internal list)
        remove_words = args.get('remove_words', [])
        text = self._remove_unwanted_words(text, remove_words)
        
        # 2. Strip contents inside brackets (Feature)
        if args.get('strip_contents_inside', True):
            text = StripContentsInsideFeature.apply(text, strip_bracket_types=args.get('strip_bracket_types'))
            
        # 3. Remove long words (Feature)
        text = MaximumWordLengthFeature.apply(text, **args)
        
        # 4. Basic ASCII and whitespace cleanup
        text = re.sub(r'[^\x00-\x7F]+', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        # 5. Remove underscore tags (Internal Logic)
        if args.get('remove_underscore_tags', True):
            text = self._remove_underscore_tags(text)
        
        return text
    
    def _run_inference(self, images: List[Image.Image], prompt: List[str], args: Dict[str, Any]) -> List[str]:
        """
        Run PaliGemma2 inference on a batch of images.
        
        Args:
            images: List of PIL Images
            prompt: List of text prompts (one per image)
            args: Dictionary of generation parameters
            
        Returns:
            List of generated and cleaned captions
        """
        max_tokens = args.get('max_tokens', 512)
        temperature = args.get('temperature', 0.7)
        top_k = args.get('top_k', 50)
        repetition_penalty = args.get('repetition_penalty', 1.15)
        
        # Prepare batch inputs
        batch_inputs = []
        for image, p in zip(images, prompt):
            # Ensure prompt starts with <image>
            if not p.strip().startswith("<image>"):
                p = f"<image>{p}"
                
            inputs = self.processor(
                text=p,
                images=image,
                return_tensors="pt"
            ).to(torch.bfloat16).to(self.device)
            batch_inputs.append(inputs)
        
        # Concatenate for batch processing
        input_lens = [inputs["input_ids"].shape[-1] for inputs in batch_inputs]
        concatenated_inputs = {
            k: torch.cat([inputs[k] for inputs in batch_inputs], dim=0)
            for k in batch_inputs[0].keys()
        }
        
        # Generate
        with torch.inference_mode():
            generations = self.model.generate(
                **concatenated_inputs,
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=temperature,
                top_k=top_k,
                top_p=0.9,
                no_repeat_ngram_size=2,
                repetition_penalty=repetition_penalty
            )
        
        # Decode and clean
        captions = []
        for generation, input_len in zip(generations, input_lens):
            generation = generation[input_len:]
            decoded = self.processor.decode(generation, skip_special_tokens=True)
            
            pruned_text = self._prune_text(decoded, args.get('prune_end', True))
            cleaned_text = self._clean_text(pruned_text, args)
            
            captions.append(cleaned_text)
        
        return captions
    
    def unload(self):
        """Free model resources using shared utility."""
        from src.core.model_utils import unload_model, UnloadMode
        unload_model(self.model, self.processor, UnloadMode.DEVICE_MAP)
        self.model = None
        self.processor = None

