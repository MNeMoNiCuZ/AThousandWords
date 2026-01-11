"""
Qwen3-VL Wrapper

Model: Qwen/Qwen3-VL Series
Supports: Images, Video, Flash Attention 2, Pixel/Token Budgeting
"""

from .base import BaseCaptionModel
from typing import List, Dict, Any, Union
from PIL import Image
from pathlib import Path
import torch

class Qwen3VLWrapper(BaseCaptionModel):
    """
    Wrapper for Qwen3-VL models.
    """
    
    MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"  # Fallback
    
    def __init__(self, config):
        super().__init__(config)
        self.processor = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    def _load_model(self):
        """Load Qwen3-VL model and processor."""
        if self.model is not None:
            return
            
        try:
            from transformers import AutoProcessor, AutoModelForImageTextToText
        except ImportError as e:
            raise ImportError(
                "Please update transformers: pip install -U transformers"
            ) from e
            
        # Determine model path from variant or default
        # args might not be set if _load_model is called outside run (shouldn't happen in normal flow but for safety)
        args = getattr(self, 'current_args', {})
        
        version = args.get('model_version')
        # If no version selected (None or empty), check config defaults
        if not version:
            version = self.config.get('defaults', {}).get('model_version', "8B")
            
        model_path = self.config.get('model_versions', {}).get(version, self.MODEL_ID)
        
        # Check Flash Attention setting
        # Check Flash Attention setting
        use_flash_attn = args.get('flash_attention', False)
        
        # Default to None (auto/SDPA) instead of eager for better performance
        attn_implementation = None 
        dtype_value = "auto"
        
        if use_flash_attn:
            attn_implementation = "flash_attention_2"
            dtype_value = torch.bfloat16
            print("Enabling Flash Attention 2")
        
        self.processor = AutoProcessor.from_pretrained(model_path)
        
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_path,
            device_map=self.device,
            attn_implementation=attn_implementation,
            dtype=dtype_value
        ).eval()
        
        print(f"Qwen3-VL loaded on {self.device}")
        
    def _run_inference(self, media_items: List[Union[Image.Image, str, Path]], prompt: List[str], args: Dict[str, Any]) -> List[str]:
        """
        Run inference on images or videos.
        """
        # Extract params
        max_tokens = args.get('max_tokens', 1024)
        temperature = args.get('temperature', 0.7)
        top_k = args.get('top_k', 50)
        repetition_penalty = args.get('repetition_penalty', 1.1)
        fps = args.get('fps', 4)
        
        # Configure visual token limits
        min_pixels = args.get('min_visual_tokens', 256)
        max_pixels = args.get('max_visual_tokens', 1280)
        min_video_pixels = args.get('min_video_tokens', 256)
        max_video_pixels = args.get('max_video_tokens', 16384)
        
        # Set processor limits using 32x compression factor logic derived from user request.
        # Formula: tokens * 32 * 32 (pixels)
        
        if hasattr(self.processor, 'image_processor'):
            self.processor.image_processor.size = {
                "longest_edge": max_pixels * 32 * 32,
                "shortest_edge": min_pixels * 32 * 32
            }
            
        if hasattr(self.processor, 'video_processor'):
            self.processor.video_processor.size = {
                 "longest_edge": max_video_pixels * 32 * 32 * 2, # User included *2 for temporal?
                 "shortest_edge": min_video_pixels * 32 * 32 * 2
            }

        # Build messages
        messages = []
        for item, p in zip(media_items, prompt):
            content = []
            
            # Check if video (Path or string ending in video ext)
            is_video = False
            if isinstance(item, (str, Path)):
                path_str = str(item).lower()
                if path_str.endswith(('.mp4', '.avi', '.mov', '.mkv')):
                    is_video = True
                    content.append({"type": "video", "video": str(item)})
            
            if not is_video:
                # Assume Image object or image path
                content.append({"type": "image", "image": item})
                
            content.append({"type": "text", "text": p})
            messages.append([{"role": "user", "content": content}])

        # Prepare inputs
        # apply_chat_template processes a list of conversations (list of lists of dicts)
        # But we prepare one batch at a time usually? 
        # Qwen's apply_chat_template handles a list of messages (one conversation) or list of lists (batch).
        # Wrapper base sends a list of images (batch).
        # We constructed `messages` as a list of conversations (one per image).
        
        # Inject system prompt if provided
        system_prompt = args.get('system_prompt')
        if system_prompt:
             for msg in messages:
                # Insert system message at the beginning
                msg.insert(0, {"role": "system", "content": [{"type": "text", "text": system_prompt}]})

        texts = [
            self.processor.apply_chat_template(
                msg, tokenize=False, add_generation_prompt=True
            ) for msg in messages
        ]
        
        # Processing inputs
        # Robustly extract images and videos from ALL messages in the conversation
        image_inputs = []
        video_inputs = []
        
        for conversation in messages:
            for message in conversation:
                for content_part in message.get("content", []):
                    if content_part.get("type") == "image":
                        image_inputs.append(content_part["image"])
                    elif content_part.get("type") == "video":
                        video_inputs.append(content_part["video"])
        
        # If empty lists, pass None
        if not image_inputs: image_inputs = None
        if not video_inputs: video_inputs = None
        
        inputs = self.processor(
            text=texts,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
            fps=fps if video_inputs else None 
        )
        
        inputs = inputs.to(self.device)
        
        # Generate
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_k=top_k,
                repetition_penalty=repetition_penalty
            )
            
        # Decode
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        
        return output_text

    def unload(self):
        """Free model resources using shared utility."""
        from src.core.model_utils import unload_model, UnloadMode
        unload_model(self.model, self.processor, UnloadMode.DEVICE_MAP)
        self.model = None
        self.processor = None

