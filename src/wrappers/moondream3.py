"""
Moondream3 Wrapper

Model: moondream/moondream3-preview
Supports: Advanced VLM with captioning, Q&A, detection, and pointing

Modes (via model_mode):
- Caption: Generate image descriptions
- Query: Ask questions about images (uses Task Prompt)
- Query with Reasoning: Ask questions with thinking process
- Detect: Find objects and output bounding boxes (uses Task Prompt as target)
- Point: Locate objects and output coordinates (uses Task Prompt as target)

Note: This model requires compilation for optimal performance.
      First run may take 30-60 seconds for compilation warmup.
"""

from .base import BaseCaptionModel
from src.core.console_kit import console, Fore
from typing import List, Dict, Any
from PIL import Image
import torch
import json


class Moondream3Wrapper(BaseCaptionModel):
    """
    Wrapper for Moondream3-preview vision-language model.
    
    Uses the HuggingFace transformers interface with trust_remote_code.
    Requires compilation (.compile()) for optimal FlexAttention performance.
    
    Modes are mutually exclusive:
    - Caption: Generate descriptions (supports caption_length setting)
    - Query: Answer questions (uses Task Prompt as the question)
    - Query with Reasoning: Answer with thinking process
    - Detect: Find objects (uses Task Prompt as target, e.g. "face")
    - Point: Locate objects (uses Task Prompt as target, e.g. "red car")
    """
    
    MODEL_ID = "moondream/moondream3-preview"
    
    def __init__(self, config):
        super().__init__(config)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._compiled = False
    
    def _load_model(self):
        """Load and compile Moondream3 model."""
        if self.model is not None:
            return
        
        import os
        import logging
        
        logging.getLogger("huggingface_hub.file_download").setLevel(logging.ERROR)
        os.environ["HF_HUB_DISABLE_XET"] = "1"
        
        from .moondream3_lib.hf_moondream import HfMoondream as MoondreamForCausalLM
        
        self._print_item("Loading", self.MODEL_ID)
        
        self.model = MoondreamForCausalLM.from_pretrained(
            self.MODEL_ID,
            dtype=torch.bfloat16,
            device_map={"": self.device}
        )
        
        self._print_item("Status", f"Model loaded on {self.device}")
        
        if not self._compiled:
            self._print_item("Compiling", "Running model.compile() for FlexAttention (this may take 30-60 seconds)...")
            try:
                self.model.compile()
                self._compiled = True
                self._print_item("Compiled", "Model compilation complete")
            except Exception as e:
                self._print_item("Warning", f"Compilation failed: {e}. Running without compilation.")
    
    def _run_inference(self, images: List[Image.Image], prompt: List[str], args: Dict[str, Any]) -> List[str]:
        """
        Run Moondream3 inference on images.
        
        Args:
            images: List of PIL Images
            prompt: List of Task Prompts (one per image)
            args: Dictionary containing:
                - model_mode: "Caption", "Query", "Query with Reasoning", "Detect", or "Point"
                - caption_length: "Short", "Normal", or "Long" (for Caption mode)
                - temperature: float - generation temperature
                - max_tokens: int - maximum tokens for generation
        
        Returns:
            List of results (captions, answers, or JSON detection/pointing data)
        """
        model_mode = args.get('model_mode', 'Caption')
        caption_length = args.get('caption_length', 'Normal').lower()
        temperature = args.get('temperature', 0.7)
        max_tokens = args.get('max_tokens', 512)
        
        # Derive reasoning flag from mode
        reasoning = False
        if model_mode == "Query with Reasoning":
            reasoning = True
            model_mode = "Query"  # Treat as Query mode internally
            
        # Validate caption_length (model expects lowercase)
        if caption_length not in ['short', 'normal', 'long']:
            caption_length = 'normal'
            
        settings = {
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
        results = []
        
        for image, p in zip(images, prompt):
            result_text = ""
            
            if model_mode == "Query":
                # Query mode: Answer a question about the image
                if not p:
                    result_text = "[Error: Query mode requires a Task Prompt with your question]"
                else:
                    result = self.model.query(
                        image=image, 
                        question=p,
                        reasoning=reasoning,
                        settings=settings
                    )
                    result_text = result.get("answer", "")
                    
                    if reasoning and "reasoning" in result:
                        reasoning_text = result.get("reasoning", {}).get("text", "")
                        if reasoning_text:
                            self._print_item("Reasoning", reasoning_text, Fore.YELLOW)
            
            elif model_mode == "Detect":
                # Detect mode: Find objects and output bounding boxes
                if not p:
                    result_text = "[Error: Detect mode requires a Task Prompt with target object (e.g. 'face', 'car')]"
                else:
                    try:
                        detect_result = self.model.detect(image, p)
                        objects = detect_result.get("objects", [])
                        result_text = json.dumps({
                            "target": prompt,
                            "objects": objects,
                            "count": len(objects)
                        }, indent=2)
                    except Exception as e:
                        result_text = f"[Detection Error: {e}]"
            
            elif model_mode == "Point":
                # Point mode: Locate objects and output coordinates
                if not p:
                    result_text = "[Error: Point mode requires a Task Prompt with target object (e.g. 'red car', 'person')]"
                else:
                    try:
                        point_result = self.model.point(image, p)
                        points = point_result.get("points", [])
                        result_text = json.dumps({
                            "target": p,
                            "points": points,
                            "count": len(points)
                        }, indent=2)
                    except Exception as e:
                        result_text = f"[Pointing Error: {e}]"
            
            else:
                # Caption mode (default): Generate image description
                # Note: Reasoning is currently not exposed for pure caption mode in GUI
                result = self.model.caption(
                    image, 
                    length=caption_length,
                    settings=settings
                )
                result_text = result.get("caption", "")
            
            results.append(result_text)
        
        return results
    
    def unload(self):
        """Free model resources."""
        from src.core.model_utils import unload_model, UnloadMode
        unload_model(self.model, None, UnloadMode.DEVICE_MAP)
        self.model = None
        self._compiled = False
