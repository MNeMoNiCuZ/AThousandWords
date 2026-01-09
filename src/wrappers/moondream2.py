"""
Moondream2 Wrapper

Model: vikhyatk/moondream2
Supports: Image captioning and visual querying

Features:
- caption_length: short, normal, long
- query_mode: when enabled, uses task_prompt as a question
"""

from .base import BaseCaptionModel
from typing import List, Dict, Any, Union
from PIL import Image
from pathlib import Path
import torch


class Moondream2Wrapper(BaseCaptionModel):
    """
    Wrapper for Moondream2 vision-language model.
    
    Uses the HuggingFace transformers interface with trust_remote_code.
    Supports two modes (via model_mode):
    - Caption: Generates image descriptions
    - Query: Answers questions about the image
    """
    
    MODEL_ID = "vikhyatk/moondream2"
    MODEL_REVISION = "2025-06-21"
    
    def __init__(self, config):
        super().__init__(config)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    def _load_model(self):
        """Load Moondream2 model."""
        if self.model is not None:
            return
        
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        self._print_item("Loading", f"{self.MODEL_ID} (revision: {self.MODEL_REVISION})")
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.MODEL_ID,
            revision=self.MODEL_REVISION,
            trust_remote_code=True,
            device_map={"": self.device}
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.MODEL_ID,
            revision=self.MODEL_REVISION
        )
        
        self._print_item("Status", f"Model loaded on {self.device}")
    
    def _run_inference(self, images: List[Image.Image], prompt: str, args: Dict[str, Any]) -> List[str]:
        """
        Run Moondream2 inference on images.
        
        Args:
            images: List of PIL Images
            prompt: The prompt/question (used in query mode)
            args: Dictionary containing:
                - caption_length: "short", "normal", or "long"
                - query_mode: bool - if True, use query() instead of caption()
                - max_tokens: Maximum tokens for generation
        
        Returns:
            List of captions/answers
        """
        model_mode = args.get('model_mode', 'Caption')
        max_tokens = args.get('max_tokens', 512)
        
        # Verify mode (default to Caption if invalid)
        if model_mode not in ["Caption", "Query"]:
            model_mode = "Caption"
        
        print(f"DEBUG: Moondream2 Prompt: '{prompt}' | Mode: {model_mode}")
        
        results = []
        
        for image in images:
            # Prepare image embeddings
            enc_image = self.model.encode_image(image)
            
            if model_mode == "Query" and prompt:
                # Query mode - answer the question
                answer = self.model.answer_question(enc_image, prompt, self.tokenizer, max_new_tokens=int(max_tokens))
                results.append(answer)
            else:
                # Caption mode - generate description
                # Note: Moondream2 uses answer_question for captioning too, typically with empty prompt or specific instructions
                # But here we pass the prompt as is (usually contains instruction)
                answer = self.model.answer_question(enc_image, prompt, self.tokenizer, max_new_tokens=int(max_tokens))
                results.append(answer)
        
        return results
    
    def unload(self):
        """Free model resources."""
        from src.core.model_utils import unload_model, UnloadMode
        unload_model(self.model, None, UnloadMode.DEVICE_MAP)
        self.model = None
