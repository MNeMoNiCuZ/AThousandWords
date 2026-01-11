"""
QwenVL 2.7B Wrapper

Based on: docs/source_captioners/qwenvl2.7b/batch.py
Model: Qwen/Qwen2-VL-7B-Instruct

This wrapper implements the Qwen2-VL-7B captioner using Qwen2VLForConditionalGeneration.
"""

from .base import BaseCaptionModel
from typing import List, Dict, Any, Union
from PIL import Image
from pathlib import Path
import torch


class QwenVL27BWrapper(BaseCaptionModel):
    """
    Wrapper for Qwen2-VL-7B-Instruct model.
    
    Model-specific behavior:
    - Uses Qwen2VLForConditionalGeneration architecture
    - Requires qwen_vl_utils for vision processing
    - Supports temperature, top_k, repetition_penalty
    - Uses chat template formatting
    """
    
    MODEL_ID = "Qwen/Qwen2-VL-7B-Instruct"
    
    def __init__(self, config):
        super().__init__(config)
        self.processor = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    def _load_model(self):
        """Load Qwen2-VL model and processor."""
        if self.model is not None:
            return
        
        try:
            from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
        except ImportError as e:
            raise ImportError(
                "Failed to import Qwen2VL components. "
                "Please ensure transformers is up to date: pip install -U transformers"
            ) from e
        
        try:
            from qwen_vl_utils import process_vision_info
            self.process_vision_info = process_vision_info
        except ImportError as e:
            raise ImportError(
                "qwen_vl_utils is required for Qwen2-VL models. "
                "Please install it: pip install qwen-vl-utils"
            ) from e
        
        print(f"Loading QwenVL model: {self.MODEL_ID}...")
        
        self.processor = AutoProcessor.from_pretrained(
            self.MODEL_ID,
            trust_remote_code=True
        )
        
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            self.MODEL_ID,
            trust_remote_code=True,
            dtype="auto"
        ).to(self.device).eval()
        
        print(f"QwenVL loaded on {self.device}")
    
    def _run_inference(self, images: List[Union[Image.Image, Path]], prompt: List[str], args: Dict[str, Any]) -> List[str]:
        """
        Run Qwen2-VL inference on a batch of images.
        
        Args:
            images: List of PIL Images or Path objects (for videos)
            prompt: List of text prompts (one per image)
            args: Dictionary of generation parameters
            
        Returns:
            List of generated captions
        """
        # Get model-specific parameters
        max_tokens = args.get('max_tokens', 256)
        temperature = args.get('temperature', 0.7)
        top_k = args.get('top_k', 50)
        repetition_penalty = args.get('repetition_penalty', 1.3)
        
        # Resolution control - Hard coded intentionally
        min_pixels = 256 * 28 * 28
        
        # Calculate max_pixels based on image_size or max_width/max_height
        # Qwen2-VL uses native resolution up to max_pixels token budget
        
        # 1. Try image_size (Standard GUI slider)
        image_size = args.get('image_size')
        
        # 2. Try max_width/max_height (Advanced/Core overrides)
        max_width = args.get('max_width')
        max_height = args.get('max_height')
        
        # Default Qwen2-VL max pixels (from docs/examples)
        default_max_pixels = 1280 * 28 * 28
        
        if image_size:
            # Treat image_size as the long edge or square target
            # For Qwen max_pixels, we square it to get total pixel budget
            # e.g. 1024 -> 1M pixels
            try:
                size_val = int(image_size)
                max_pixels = size_val * size_val
            except (ValueError, TypeError):
                max_pixels = default_max_pixels
                
        elif max_width or max_height:
            # If user specified resizing constraints, use them to guide max_pixels
            w = int(max_width) if max_width else 4096 
            h = int(max_height) if max_height else 4096
            max_pixels = w * h
        else:
            max_pixels = default_max_pixels

        messages_batch = []
        for image, p in zip(images, prompt):
            # Prepare content item based on type
            content_item = {}
            
            # Handle Path objects (Videos or Image paths)
            if isinstance(image, (str, Path)):
                # CRITICAL Fix for Windows: qwen_vl_utils requires string paths, not Path objects
                path_str = str(image.absolute()) if isinstance(image, Path) else str(image)
                
                # Detect video by extension
                lower_path = path_str.lower()
                is_video = lower_path.endswith(('.mp4', '.avi', '.mov', '.mkv', '.webm'))
                
                if is_video:
                    content_item = {
                        "type": "video",
                        "video": path_str,
                        "max_pixels": max_pixels,
                        # "fps": 1.0, # Optional: could expose this
                    }
                else:
                    content_item = {
                        "type": "image",
                        "image": path_str,
                        "min_pixels": min_pixels,
                        "max_pixels": max_pixels,
                    }
            
            # Handle PIL Images (In-memory)
            else:
                 content_item = {
                    "type": "image",
                    "image": image,
                    "min_pixels": min_pixels,
                    "max_pixels": max_pixels,
                }
            
            messages = [
                {
                    "role": "user",
                    "content": [
                        content_item,
                        {"type": "text", "text": p},
                    ],
                }
            ]
            
            # Add system prompt if configured
            system_prompt = args.get('system_prompt')
            if system_prompt:
                messages.insert(0, {
                    "role": "system", 
                    "content": [{"type": "text", "text": system_prompt}]
                })
                
            messages_batch.append(messages)
        
        # Apply chat template to each message
        texts = [
            self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            for messages in messages_batch
        ]
        
        # Process vision info for each message
        all_image_inputs = []
        all_video_inputs = []
        for messages in messages_batch:
            image_inputs, video_inputs = self.process_vision_info(messages)
            all_image_inputs.append(image_inputs)
            all_video_inputs.append(video_inputs)
        
        # Flatten for batch processing - handle None/empty cases
        flat_image_inputs = None
        if any(all_image_inputs):
            flat_image_inputs = [img for sublist in all_image_inputs if sublist for img in sublist]
            if not flat_image_inputs:
                flat_image_inputs = None
        
        flat_video_inputs = None
        if any(all_video_inputs):
            flat_video_inputs = [vid for sublist in all_video_inputs if sublist for vid in sublist]
            if not flat_video_inputs:
                flat_video_inputs = None
        
        # Prepare inputs for the model
        inputs = self.processor(
            text=texts,
            images=flat_image_inputs,
            videos=flat_video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.device)
        
        # Generate captions
        # Enable sampling when temperature > 0
        do_sample = temperature > 0
        
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                repetition_penalty=repetition_penalty,
                temperature=temperature if do_sample else None,
                top_k=top_k if do_sample else None,
                do_sample=do_sample
            )
        
        # Trim the input part from generated output
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        
        # Decode the output into text
        output_texts = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )
        
        return output_texts
    
    def unload(self):
        """Free model resources using shared utility."""
        from src.core.model_utils import unload_model, UnloadMode
        unload_model(self.model, self.processor, UnloadMode.STANDARD)
        self.model = None
        self.processor = None

