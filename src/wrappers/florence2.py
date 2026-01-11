"""
Florence2 Wrapper

Based on: docs/source_captioners/florence2/batch.py
Model: microsoft/Florence-2-large

This wrapper implements Florence-2 with task-based prompting for different detail levels.
"""

from .base import BaseCaptionModel
from typing import List, Dict, Any
from PIL import Image
import torch
from functools import partial
from unittest.mock import patch


class Florence2Wrapper(BaseCaptionModel):
    """
    Wrapper for Microsoft Florence-2-large model.
    
    Model-specific behavior:
    - Uses task prompts: <CAPTION>, <DETAILED_CAPTION>, <MORE_DETAILED_CAPTION>
    - Uses fixed_get_imports patch to remove flash_attn dependency
    - Uses bfloat16 precision
    - Supports batch processing
    - Uses num_beams=3, do_sample=False for generation
    """
    
    MODEL_ID = "microsoft/Florence-2-large"
    
    def __init__(self, config):
        super().__init__(config)
        self.processor = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    def _load_model(self):
        """Load Florence-2 model and processor with flash_attn patch."""
        if self.model is not None:
            return
        
        from transformers import AutoModelForCausalLM, AutoProcessor
        from transformers.dynamic_module_utils import get_imports
        
        def fixed_get_imports(filename):
            """Remove flash_attn import for compatibility."""
            imports = get_imports(filename)
            if str(filename).endswith("modeling_florence2.py"):
                imports = [imp for imp in imports if imp != "flash_attn"]
            return imports
        
        print(f"Loading Florence-2 model: {self.MODEL_ID}...")
        
        # Florence-2 often has issues with Flash Attention 2
        # Florence-2 remote code doesn't declare SDPA support, so we force eager to avoid crashes.
        # It also doesn't support Flash Attention 2.
        attn_implementation = "eager"
        print(f"Using attention implementation: {attn_implementation}")
        
        # Apply patch to prevent flash_attn import inside the model code if it tries to auto-detect
        model_id = self.config.get('model_id', self.MODEL_ID)
        with patch("transformers.dynamic_module_utils.get_imports", fixed_get_imports):
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                trust_remote_code=True,
                dtype=torch.bfloat16,
                attn_implementation=attn_implementation
            ).to(self.device)
        
        self.processor = AutoProcessor.from_pretrained(
            model_id,
            trust_remote_code=True
        )
        
        print(f"Florence-2 loaded on {self.device}")
    
    def _run_inference(self, images: List[Image.Image], prompt: List[str], args: Dict[str, Any]) -> List[str]:
        """Run florence-2 inference on batch of images - matches source implementation."""
        if self.processor is None:
            raise RuntimeError("Florence-2 processor is None - model may not have loaded correctly")
        
        inputs = {
            "input_ids": [],
            "pixel_values": []
        }
        
        for img, p in zip(images, prompt):
            if img is None or not isinstance(img, Image.Image):
                continue
            
            try:
                input_data = self.processor(
                    text=p,
                    images=img,
                    return_tensors="pt"
                )
                
                if "pixel_values" not in input_data or input_data["pixel_values"] is None:
                    print(f"Warning: processor returned no pixel_values")
                    continue
                    
                inputs["input_ids"].append(input_data["input_ids"])
                inputs["pixel_values"].append(input_data["pixel_values"])
            except Exception as e:
                print(f"Warning: Failed to process image: {e}")
                continue
        
        if not inputs["input_ids"]:
            return [""] * len(images)
        
        inputs["input_ids"] = torch.cat(inputs["input_ids"]).to(self.device)
        inputs["pixel_values"] = torch.cat(inputs["pixel_values"]).to(self.device).to(torch.bfloat16)
        
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=args.get('max_tokens', 1024),
                do_sample=False,
                num_beams=3,
                use_cache=False
            )
        
        results = self.processor.batch_decode(generated_ids, skip_special_tokens=False)
        
        cleaned_results = [
            result.replace('</s>', '').replace('<s>', '').replace('<pad>', '').strip()
            for result in results
        ]
        
        return cleaned_results
    
    def unload(self):
        """Free model resources using shared utility."""
        from src.core.model_utils import unload_model, UnloadMode
        unload_model(self.model, self.processor, UnloadMode.STANDARD)
        self.model = None
        self.processor = None

