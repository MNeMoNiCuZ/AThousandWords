"""
Qwen2-VL-7B-Captioner-Relaxed Wrapper

Based on: docs/source_captioners/qwen2-vl-7b-captioner-relaxed/batch.py
Model: Ertugrul/Qwen2-VL-7B-Captioner-Relaxed

This wrapper implements the relaxed version of Qwen2-VL-7B using Qwen2VLForConditionalGeneration.
"""

from .base import BaseCaptionModel
from typing import List, Dict, Any
from PIL import Image
import torch


class Qwen2VLRelaxedWrapper(BaseCaptionModel):
    """
    Wrapper for Ertugrul/Qwen2-VL-7B-Captioner-Relaxed model.
    
    Model-specific behavior:
    - Uses Qwen2VLForConditionalGeneration architecture
    - Fine-tuned version with relaxed prompt constraints
    - Supports temperature, top_k, repetition_penalty
    - Uses chat template formatting
    - Uses bfloat16 precision with autocast
    """
    
    MODEL_ID = "Ertugrul/Qwen2-VL-7B-Captioner-Relaxed"
    
    def __init__(self, config):
        super().__init__(config)
        self.processor = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    def _load_model(self):
        """Load Qwen2-VL-Relaxed model and processor."""
        if self.model is not None:
            return
        
        try:
            from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
        except ImportError as e:
            raise ImportError(
                "Failed to import Qwen2VL components. "
                "Please ensure transformers is up to date: pip install -U transformers"
            ) from e
        
        print(f"Loading Qwen2-VL-Relaxed model: {self.MODEL_ID}...")
        
        self.processor = AutoProcessor.from_pretrained(self.MODEL_ID)
        
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            self.MODEL_ID,
            dtype=torch.bfloat16,
            device_map="auto"
        )
        
        print(f"Qwen2-VL-Relaxed loaded on {self.device}")
    
    def _run_inference(self, images: List[Image.Image], prompt: str, args: Dict[str, Any]) -> List[str]:
        """
        Run Qwen2-VL-Relaxed inference on a batch of images.
        
        Args:
            images: List of PIL Images
            prompt: Text prompt for captioning
            args: Dictionary of generation parameters
            
        Returns:
            List of generated captions
        """
        # Get model-specific parameters
        max_tokens = args.get('max_tokens', 384)
        temperature = args.get('temperature', 0.7)
        top_k = args.get('top_k', 50)
        repetition_penalty = args.get('repetition_penalty', 1.3)
        
        # Build messages for each image using Qwen chat format
        texts = []
        for _ in images:
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
            
            # Add system prompt
            system_prompt = args.get('system_prompt')
            if system_prompt:
                conversation.insert(0, {
                    "role": "system",
                    "content": [{"type": "text", "text": system_prompt}]
                })
            
            # Apply the chat template to format the message for processing
            text_prompt = self.processor.apply_chat_template(
                conversation, add_generation_prompt=True
            )
            texts.append(text_prompt)
            
        # Prepare the inputs for the model in a batch
        inputs = self.processor(
            text=texts,
            images=images,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.device)
        
        # Generate with autocast as per source script
        with torch.no_grad():
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    do_sample=True,
                    temperature=temperature,
                    use_cache=True,
                    top_k=top_k,
                    repetition_penalty=repetition_penalty,
                )
        
        # Trim the generated IDs to remove the input part from the output
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, output_ids)
        ]
        
        # Decode the trimmed output into text, skipping special tokens
        output_texts = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        
        return output_texts
    
    def unload(self):
        """Free model resources using shared utility."""
        from src.core.model_utils import unload_model, UnloadMode
        unload_model(self.model, self.processor, UnloadMode.DEVICE_MAP)
        self.model = None
        self.processor = None

