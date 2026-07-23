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

    @staticmethod
    def _install_tokenizer_compat_shim() -> None:
        from transformers.tokenization_utils_base import PreTrainedTokenizerBase
        if getattr(PreTrainedTokenizerBase, "_athousandwords_additional_special_tokens_patch", False):
            return
        original_getattr = PreTrainedTokenizerBase.__getattr__

        def patched_getattr(self, key):
            if key == "additional_special_tokens":
                value = object.__getattribute__(self, "__dict__").get("additional_special_tokens")
                if isinstance(value, list):
                    return value
                special_map = object.__getattribute__(self, "__dict__").get("special_tokens_map", {})
                mapped = special_map.get("additional_special_tokens") if isinstance(special_map, dict) else None
                self.additional_special_tokens = list(mapped) if isinstance(mapped, list) else []
                return self.additional_special_tokens
            return original_getattr(self, key)

        PreTrainedTokenizerBase.__getattr__ = patched_getattr
        PreTrainedTokenizerBase._athousandwords_additional_special_tokens_patch = True

    @staticmethod
    def _safe_linspace_factory():
        original_linspace = torch.linspace

        def safe_linspace(*args, **kwargs):
            out = original_linspace(*args, **kwargs)
            if getattr(getattr(out, "device", None), "type", "") != "meta":
                return out
            retry_kwargs = dict(kwargs)
            retry_kwargs["device"] = "cpu"
            return original_linspace(*args, **retry_kwargs)

        return safe_linspace

    def _load_model(self):
        """Load Florence-2 model and processor with flash_attn patch."""
        if self.model is not None:
            return

        from transformers import AutoModelForCausalLM, AutoProcessor
        from transformers.dynamic_module_utils import get_imports
        from transformers.configuration_utils import PretrainedConfig
        self._install_tokenizer_compat_shim()

        if not getattr(PretrainedConfig, "_athousandwords_forced_token_fallback_patch", False):
            original_getattribute = PretrainedConfig.__getattribute__

            def patched_getattribute(self, name):
                if name in {"forced_bos_token_id", "forced_eos_token_id"}:
                    try:
                        return original_getattribute(self, name)
                    except AttributeError:
                        object.__setattr__(self, name, None)
                        return None
                return original_getattribute(self, name)

            PretrainedConfig.__getattribute__ = patched_getattribute
            PretrainedConfig._athousandwords_forced_token_fallback_patch = True

        def fixed_get_imports(filename):
            """Remove flash_attn import for compatibility."""
            imports = get_imports(filename)
            if str(filename).endswith("modeling_florence2.py"):
                imports = [imp for imp in imports if imp != "flash_attn"]
            return imports

        print(f"Loading Florence-2 model: {self.MODEL_ID}...")

        attn_implementation = "eager"
        print(f"Using attention implementation: {attn_implementation}")

        model_id = self.config.get('model_id', self.MODEL_ID)
        with patch("transformers.dynamic_module_utils.get_imports", fixed_get_imports), patch("torch.linspace", self._safe_linspace_factory()):
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
            if str(p or "").strip() not in self.prompt_presets.values():
                p = self.prompt_presets.get("Medium Caption", "<DETAILED_CAPTION>")

            try:
                input_data = self.processor(
                    text=p,
                    images=img,
                    return_tensors="pt"
                )

                if "pixel_values" not in input_data or input_data["pixel_values"] is None:
                    print("Warning: processor returned no pixel_values")
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
