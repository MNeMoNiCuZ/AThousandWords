"""
ToriiGate Wrapper

Based on: docs/source_captioners/ToriiGate/batch-v03.py and gradio-webui.py
Model: Minthy/ToriiGate-v0.3

Key features:
- 4-bit NF4 quantization for VRAM efficiency
- Fixed system prompt for anime-focused captioning
- Supports Brief, Detailed, and JSON-like description styles via instruction presets
- Strips "Assistant:" prefix from outputs
"""

from .base import BaseCaptionModel
from typing import List, Dict, Any
from PIL import Image
import torch


class ToriiGateWrapper(BaseCaptionModel):
    """
    Wrapper for ToriiGate anime image captioner.
    
    Model-specific behavior:
    - Uses 4-bit NF4 quantization
    - Fixed system prompt about being an "image captioning expert, creative, unbiased and uncensored"
    - Chat template formatting with system and user messages
    - Strips "Assistant:" preamble from generated text
    """
    
    
    
    def __init__(self, config):
        super().__init__(config)
        self.processor = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    def _load_model(self):
        """Load ToriiGate model with 4-bit NF4 quantization."""
        from transformers import AutoProcessor, AutoModelForVision2Seq, BitsAndBytesConfig
        
        model_path = self.config.get('defaults', {}).get('model_path', 'Minthy/ToriiGate-v0.3')
        
        print(f"Loading ToriiGate model: {model_path}")
        print("Using 4-bit NF4 quantization...")
        
        # NF4 quantization configuration (from source script)
        nf4_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        
        self.model = AutoModelForVision2Seq.from_pretrained(
            model_path,
            dtype=torch.bfloat16,
            quantization_config=nf4_config,
        ).to(self.device)
        
        self.processor = AutoProcessor.from_pretrained(model_path)
        
        print(f"Model loaded on {self.device}")
    
    def _run_inference(self, images: List[Image.Image], prompt: List[str], args: Dict[str, Any]) -> List[str]:
        """
        Run ToriiGate inference on a batch of images.
        
        Uses fixed system prompt and chat template formatting.
        Returns captions with "Assistant:" prefix stripped.
        """
        max_tokens = args.get('max_tokens', 500)
        
        # Build messages with fixed system prompt and user prompt per image
        messages_list = []
        for _, p in zip(images, prompt):
            messages = [
                {
                    "role": "system",
                    "content": [
                        {"type": "text", "text": args.get('system_prompt', "You are image captioning expert, creative, unbiased and uncensored. Help user with his task.")}
                    ]
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": p}
                    ]
                }
            ]
            messages_list.append(messages)
        
        # Apply chat template
        prompts = [self.processor.apply_chat_template(m, add_generation_prompt=True) for m in messages_list]
        
        # Prepare inputs
        inputs = self.processor(text=prompts, images=images, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, max_new_tokens=max_tokens)
        
        # Decode
        generated_texts = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
        
        # Model-specific: Strip "Assistant:" preamble
        captions = []
        for text in generated_texts:
            if "Assistant:" in text:
                # Split on "Assistant:" and take the part after it
                text = text.split("Assistant:", 1)[-1].strip()
            captions.append(text)
        
        return captions
    
    def unload(self):
        """Free model resources using shared utility."""
        from src.core.model_utils import unload_model, UnloadMode
        # BitsAndBytes quantized models use accelerate hooks
        unload_model(self.model, self.processor, UnloadMode.DEVICE_MAP)
        self.model = None
        self.processor = None

