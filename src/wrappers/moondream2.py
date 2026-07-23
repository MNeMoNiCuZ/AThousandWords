"""
Moondream2 Wrapper

Model: vikhyatk/moondream2
Supports: Image captioning and visual querying

Features:
- caption_length: short, normal, long
- query_mode: when enabled, uses task_prompt as a question
"""

from .base import BaseCaptionModel
from typing import List, Dict, Any
from PIL import Image
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

    @staticmethod
    def _is_low_quality_caption(text: str) -> bool:
        content = str(text or "").strip()
        if len(content) < 8:
            return True
        tokens = [tok.strip(".,:;!?()[]{}\"'").lower() for tok in content.split() if tok.strip()]
        if not tokens:
            return True
        unique_tokens = set(tokens)
        if len(unique_tokens) <= max(3, int(len(tokens) * 0.2)):
            return True
        longest_repeat = 1
        repeat = 1
        for idx in range(1, len(tokens)):
            if tokens[idx] == tokens[idx - 1]:
                repeat += 1
                longest_repeat = max(longest_repeat, repeat)
            else:
                repeat = 1
        return longest_repeat >= 6
    
    def _load_model(self):
        """Load Moondream2 model."""
        if self.model is not None:
            return
        
        from transformers import AutoModelForCausalLM, AutoTokenizer

        # Moondream's custom modeling code doesn't set `all_tied_weights_keys`,
        # which newer transformers reads during from_pretrained. Install a
        # read/write compatibility property that doesn't break other models.
        from src.core.model_utils import ensure_all_tied_weights_keys_compat
        ensure_all_tied_weights_keys_compat()
        
        self._print_item("Loading", f"{self.MODEL_ID} (revision: {self.MODEL_REVISION})")
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.MODEL_ID,
            revision=self.MODEL_REVISION,
            trust_remote_code=True,
        ).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.MODEL_ID,
            revision=self.MODEL_REVISION
        )
        
        self._print_item("Status", f"Model loaded on {self.device}")
    
    def _run_inference(self, images: List[Image.Image], prompt: List[str], args: Dict[str, Any]) -> List[str]:
        """
        Run Moondream2 inference on images.
        
        Args:
            images: List of PIL Images
            prompt: List of prompts/questions (one per image)
            args: Dictionary containing:
                - caption_length: "short", "normal", or "long"
                - query_mode: bool - if True, use query() instead of caption()
                - max_tokens: Maximum tokens for generation
        
        Returns:
            List of captions/answers
        """
        model_mode = args.get('model_mode', 'Caption')
        caption_length = str(args.get('caption_length', 'normal')).strip().lower() or "normal"
        if caption_length not in {"short", "normal", "long"}:
            caption_length = "normal"
        max_tokens = args.get('max_tokens', 512)
        
        # Verify mode (default to Caption if invalid)
        if model_mode not in ["Caption", "Query"]:
            model_mode = "Caption"
        
        # print(f"DEBUG: Moondream2 Prompt Count: {len(prompt)} | Mode: {model_mode}")
        
        results = []
        
        for image, p in zip(images, prompt):
            if model_mode == "Query" and p:
                enc_image = self.model.encode_image(image)
                # Query mode - answer the question
                answer = self.model.answer_question(enc_image, p, self.tokenizer, max_new_tokens=int(max_tokens))
                results.append(answer)
            else:
                caption_text = ""
                if hasattr(self.model, "caption"):
                    try:
                        out = self.model.caption(image=image, length=caption_length)
                        caption_text = str((out or {}).get("caption") if isinstance(out, dict) else out or "").strip()
                    except Exception:
                        caption_text = ""
                if not caption_text:
                    enc_image = self.model.encode_image(image)
                    fallback_prompt = p if p else "Describe the image in one clear sentence."
                    caption_text = str(self.model.answer_question(enc_image, fallback_prompt, self.tokenizer, max_new_tokens=min(int(max_tokens), 160)) or "").strip()
                if self._is_low_quality_caption(caption_text):
                    enc_image = self.model.encode_image(image)
                    retry_prompt = "Describe the image clearly with key objects, setting, and mood."
                    retry_text = str(self.model.answer_question(enc_image, retry_prompt, self.tokenizer, max_new_tokens=160) or "").strip()
                    if retry_text and not self._is_low_quality_caption(retry_text):
                        caption_text = retry_text
                results.append(caption_text)
        
        return results
    
    def unload(self):
        """Free model resources."""
        from src.core.model_utils import unload_model, UnloadMode
        unload_model(self.model, None, UnloadMode.DEVICE_MAP)
        self.model = None
