"""
SmolVLM Wrapper

Based on: docs/source_captioners/smolVLM/batch.py
Versions: 
  - HuggingFaceTB/SmolVLM-256M-Instruct
  - HuggingFaceTB/SmolVLM-500M-Instruct

This wrapper ONLY contains model-specific code.
All generic processing is handled by BaseCaptionModel.
"""

from .base import BaseCaptionModel
from typing import List, Dict, Any
from PIL import Image
import torch


class SmolVLMWrapper(BaseCaptionModel):
    """
    Wrapper for SmolVLM Instruct models.
    
    Model-specific behavior:
    - Supports 256M and 500M model variants
    - Strips "Assistant:" preamble from output
    """
    
    # Model version options are now loaded from config
    
    def __init__(self, config):
        super().__init__(config)
        self.processor = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.current_version = None
    
    def _load_model(self, version: str = None):
        """Load SmolVLM model and processor."""
        # Resolve version
        if not version:
            version = self.config.get('defaults', {}).get('model_version', "256M")
            
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
            model_id = "HuggingFaceTB/SmolVLM-256M-Instruct" if version == "256M" else "HuggingFaceTB/SmolVLM-500M-Instruct"
            self._print_item("Warning", f"Version '{version}' not found in config. Using fallback: {model_id}")
            
        self.current_version = version
        
        self._print_item("Loading Model", f"{model_id} ({version})")
        
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
        )
        self.model.to(self.device)
        
        # Verify model size
        param_count = sum(p.numel() for p in self.model.parameters())
        size_mb = param_count * 2 / (1024 * 1024) # Approx for bfloat16 (2 bytes)
        self._print_item("Model Stats", f"Params: {param_count:,} (~{size_mb:.0f} MB)")
        
        self._print_item("Status", f"Model loaded on {self.device}")
    
    def run(self, dataset, args: Dict[str, Any]) -> str:
        """Override run to pass model_version to _load_model."""
        version = args.get('model_version', '500M')
        self._load_model(version)
        return super().run(dataset, args)
    
    def _run_inference(self, images: List[Image.Image], prompt: str, args: Dict[str, Any]) -> List[str]:
        """
        Run SmolVLM inference on a batch of images.
        
        Returns raw captions with "Assistant:" preamble stripped.
        """
        max_tokens = args.get('max_tokens', 500)
        
        # Build messages
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt}
                ]
            } for _ in images
        ]
        
        # Inject system prompt
        system_prompt = args.get('system_prompt')
        if system_prompt:
            for msg_block in messages:
                # SmolVLM uses list of dicts for conversation
                # Current construction: messages is a list of user message dicts
                # We need to construct conversation history per image
                pass 
        
        # Correction: The original code creates a list of 'user' message dicts, one per image.
        # It then iterates: prompts = [processor.apply_chat_template([msg], ...)]
        # So 'msg' is a single dict. We need to turn [msg] into [{system}, {user}]
        
        prompts = []
        for msg in messages:
            conversation = []
            if system_prompt:
                conversation.append({
                    "role": "system",
                    "content": [{"type": "text", "text": system_prompt}]
                })
            conversation.append(msg)
            
            prompts.append(self.processor.apply_chat_template(conversation, add_generation_prompt=True))
        inputs = self.processor(text=prompts, images=images, return_tensors="pt", padding=True).to(self.device)
        
        # Generate
        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, max_new_tokens=max_tokens)
            generated_ids = generated_ids.to(self.device)
            generated_texts = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
        
        # Model-specific: Strip "Assistant:" preamble
        captions = []
        for text in generated_texts:
            if "Assistant:" in text:
                text = text.split("Assistant:", 1)[-1].strip()
            captions.append(text)
        
        return captions
    
    def unload(self):
        """Free model resources using shared utility."""
        from src.core.model_utils import unload_model, UnloadMode
        unload_model(self.model, self.processor, UnloadMode.STANDARD)
        self.model = None
        self.processor = None
        self.current_version = None

