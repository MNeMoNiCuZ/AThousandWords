"""
Moondream Wrapper

Model: vikhyatk/moondream1
Supports: Visual question answering (1.6B parameters)

This is the original Moondream model, released for research purposes only.
Uses the older API with CodeGenTokenizerFast and answer_question method.

Features:
- None (Uses VQA approach by default)
"""

from .base import BaseCaptionModel
from typing import List, Dict, Any
from PIL import Image
import torch
import warnings

# Suppress transformers warning about PhiModel not inheriting from GenerationMixin
warnings.filterwarnings("ignore", message=r"(?s).*PhiModel has generative capabilities.*")



class Moondream1Wrapper(BaseCaptionModel):
    """
    Wrapper for Moondream1 vision-language model (1.6B parameters).
    
    Uses local code to avoid 'trust_remote_code=True' issues and patch generation bugs.
    This model always uses a VQA (question-answering) approach.
    
    Note: This model is for research purposes only.
    """
    
    MODEL_ID = "vikhyatk/moondream1"
    
    def __init__(self, config):
        super().__init__(config)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = None
    
    def _load_model(self):
        """Load Moondream1 model and tokenizer from local source."""
        if self.model is not None:
            return
        
        # Import local model classes
        try:
            from .moondream1_lib.moondream import Moondream
            from .moondream1_lib.configuration_moondream import MoondreamConfig
            from transformers import CodeGenTokenizerFast
        except ImportError as e:
             raise ImportError(f"Could not import generic Moondream1 classes. Ensure files are in src/wrappers/moondream1_lib/: {e}")

        self._print_item("Loading", self.MODEL_ID)
        
        # Load tokenizer
        self.tokenizer = CodeGenTokenizerFast.from_pretrained(self.MODEL_ID)

        # Load model using local class
        self.model = Moondream.from_pretrained(
            self.MODEL_ID,
            trust_remote_code=False, # We use local code now
            dtype=torch.float16 if self.device == "cuda" else torch.float32
        ).to(self.device)
        
        self._print_item("Status", f"Model loaded on {self.device}")

    
    def _run_inference(self, images: List[Image.Image], prompt: str, args: Dict[str, Any]) -> List[str]:
        """
        Run Moondream inference on images.
        
        Moondream always uses a Q&A approach. If no prompt is provided,
        default to "Describe this image."
        
        Args:
            images: List of PIL Images
            prompt: The question to ask about the image
            args: Dictionary containing:
                - max_tokens: Maximum tokens for generation
        
        Returns:
            List of answers
        """
        max_tokens = args.get('max_tokens', 512)
        
        # Default prompt if none provided
        if not prompt or not prompt.strip():
            prompt = "Describe this image."
        
        results = []
        
        for image in images:
            # Encode image
            enc_image = self.model.encode_image(image)
            
            # Answer question
            answer = self.model.answer_question(
                enc_image, 
                prompt, 
                self.tokenizer
            )
            
            results.append(answer)
        
        for image in images:
            # Drop from cache
            pass
        
        return results
    
    def unload(self):
        """Free model resources."""
        from src.core.model_utils import unload_model, UnloadMode
        unload_model(self.model, self.tokenizer, UnloadMode.STANDARD)
        self.model = None
        self.tokenizer = None
