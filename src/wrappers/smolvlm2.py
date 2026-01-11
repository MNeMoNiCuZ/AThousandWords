"""
SmolVLM2 Wrapper

Based on: docs/source_captioners/smolVLM2/batch.py
Supports: 256M, 500M, and 2.2B versions

This wrapper implements video captioning support using AutoModelForImageTextToText.
The model processes videos via the processor's chat template with video paths.
"""

from .base import BaseCaptionModel
from typing import List, Dict, Any
from PIL import Image
from pathlib import Path
import torch


class SmolVLM2Wrapper(BaseCaptionModel):
    """
    Wrapper for SmolVLM2 model - supports video captioning with multiple model versions.
    
    Model-specific behavior:
    - Uses AutoModelForImageTextToText
    - Processes videos by passing video paths to processor
    - Uses flash_attention_2 if available for memory efficiency
    - Fixed generation with do_sample=False as per source
    - Supports 256M, 500M, and 2.2B model versions
    """
    
    def __init__(self, config):
        super().__init__(config)
        self.processor = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.current_version = None
    
    def _load_model(self, version: str = None):
        """Load SmolVLM2 model and processor."""
        # Resolve version
        if not version:
            version = self.config.get('defaults', {}).get('model_version', "2.2B")
            
        # Check if we need to reload for different version
        if self.model is not None and self.current_version == version:
            return
        
        # Unload existing model if switching versions
        if self.model is not None and self.current_version != version:
            self._print_item("Switching Version", f"{self.current_version} -> {version}")
            self.unload()
        
        from transformers import AutoProcessor, AutoModelForImageTextToText
        
        # Get ID from config
        versions_map = self.config.get('model_versions', {})
        model_id = versions_map.get(version)
        
        if not model_id:
            # Fallback if config is missing entries
            model_id = "HuggingFaceTB/SmolVLM2-2.2B-Instruct"
            self._print_item("Warning", f"Version '{version}' not found in config. Using fallback: {model_id}")
            
        self.current_version = version
        
        self._print_item("Loading Model", f"{model_id} ({version})")
        
        # Try flash_attention_2 first, fall back to default if not available
        attn_impl = None
        try:
            import flash_attn
            attn_impl = "flash_attention_2"
            self._print_item("Attention", "Using flash_attention_2")
        except ImportError:
            self._print_item("Attention", "flash_attn not available, using default (SDPA)")
        
        # Load model with flash attention if available
        model_kwargs = {
            "dtype": torch.bfloat16 if self.device == "cuda" else torch.float32,
        }
        if attn_impl:
            model_kwargs["_attn_implementation"] = attn_impl
            
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            **model_kwargs
        ).to(self.device)
        
        self.processor = AutoProcessor.from_pretrained(model_id)
        # Set pad_token to eos_token for batch processing
        if self.processor.tokenizer.pad_token is None:
            self.processor.tokenizer.pad_token = self.processor.tokenizer.eos_token
        
        # Verify model size
        param_count = sum(p.numel() for p in self.model.parameters())
        size_mb = param_count * 2 / (1024 * 1024)  # Approx for bfloat16 (2 bytes)
        self._print_item("Model Stats", f"Params: {param_count:,} (~{size_mb:.0f} MB)")
        
        self._print_item("Status", f"Model loaded on {self.device}")
    
    def run(self, dataset, args: Dict[str, Any]) -> str:
        """Override run to pass model_version to _load_model."""
        version = args.get('model_version', '2.2B')
        self._load_model(version)
        return super().run(dataset, args)
    
    def _run_inference(self, media: List[Any], prompt: List[str], args: Dict[str, Any]) -> List[str]:
        """
        Run SmolVLM2 inference on videos or images in batch.
        
        Note: SmolVLM2's processor requires each message to be processed individually
        through apply_chat_template, then manually batched for generation.
        """
        max_tokens = args.get('max_tokens', 512)
        
        system_prompt = args.get('system_prompt')
        
        
        # Process each media item individually through apply_chat_template
        batch_inputs = []
        for item, p in zip(media, prompt):
            if isinstance(item, (Path, str)):
                # Video
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "video", "path": str(item)},
                            {"type": "text", "text": p}
                        ]
                    }
                ]
                
                if system_prompt:
                    messages.insert(0, {
                        "role": "system",
                        "content": [{"type": "text", "text": system_prompt}]
                    })
            else:
                # Image - need to pass the actual image to apply_chat_template
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image"},
                            {"type": "text", "text": p}
                        ]
                    }
                ]
                
                if system_prompt:
                    messages.insert(0, {
                        "role": "system",
                        "content": [{"type": "text", "text": system_prompt}]
                    })
            
            # Process this single message with tokenize=True
            # For images, we need to pass the image separately
            if isinstance(item, (Path, str)):
                # Video path goes in messages
                inputs = self.processor.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=True,
                    return_dict=True,
                    return_tensors="pt",
                )
            else:
                # Image needs to be passed to processor along with text
                # Use processor to handle both text and image together
                text_prompt = self.processor.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=False
                )
                inputs = self.processor(
                    text=[text_prompt],
                    images=[item],
                    return_tensors="pt",
                )
            
            batch_inputs.append(inputs)
        
        # Manually batch the inputs by padding to max length
        if len(batch_inputs) > 1:
            # Find max sequence length
            max_len = max(inp['input_ids'].shape[1] for inp in batch_inputs)
            
            # Pad all inputs to max length
            batched = {
                'input_ids': [],
                'attention_mask': [],
            }
            
            # Check if pixel_values exist (for images)
            has_pixel_values = 'pixel_values' in batch_inputs[0]
            if has_pixel_values:
                batched['pixel_values'] = []
            
            for inp in batch_inputs:
                # Pad input_ids and attention_mask
                seq_len = inp['input_ids'].shape[1]
                pad_len = max_len - seq_len
                
                if pad_len > 0:
                    # Pad on the left (for decoder-only models)
                    input_ids = torch.nn.functional.pad(
                        inp['input_ids'], 
                        (pad_len, 0), 
                        value=self.processor.tokenizer.pad_token_id or 0
                    )
                    attention_mask = torch.nn.functional.pad(
                        inp['attention_mask'], 
                        (pad_len, 0), 
                        value=0
                    )
                else:
                    input_ids = inp['input_ids']
                    attention_mask = inp['attention_mask']
                
                batched['input_ids'].append(input_ids)
                batched['attention_mask'].append(attention_mask)
                
                if has_pixel_values:
                    batched['pixel_values'].append(inp['pixel_values'])
            
            # Stack tensors
            inputs = {
                'input_ids': torch.cat(batched['input_ids'], dim=0),
                'attention_mask': torch.cat(batched['attention_mask'], dim=0),
            }
            
            if has_pixel_values:
                inputs['pixel_values'] = torch.cat(batched['pixel_values'], dim=0)
        else:
            inputs = batch_inputs[0]
        
        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Generate in batch
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=max_tokens
            )
        
        # Decode
        generated_texts = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
        )
        
        # Post-process strings
        captions = []
        for caption in generated_texts:
            # Strip chat template artifacts from output
            caption = caption.strip()
            if "User:" in caption:
                caption = caption.split("User:", 1)[-1].strip()
            if "Assistant:" in caption:
                caption = caption.split("Assistant:", 1)[-1].strip()
            captions.append(caption)
        
        return captions
    
    def unload(self):
        """Free model resources using shared utility."""
        from src.core.model_utils import unload_model, UnloadMode
        unload_model(self.model, self.processor, UnloadMode.STANDARD)
        self.model = None
        self.processor = None
        self.current_version = None

