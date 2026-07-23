"""
MiaoshouAI Wrapper

Based on: docs/source_captioners/miaoshouai/batch.py
Model: MiaoshouAI/Florence-2-base-PromptGen-v1.5

This wrapper implements MiaoshouAI's Florence-2 variant with multiple prompt generation modes.
"""

from .base import BaseCaptionModel
from typing import List, Dict, Any
from PIL import Image
import torch
from unittest.mock import patch


class MiaoshouAIWrapper(BaseCaptionModel):
    """
    Wrapper for MiaoshouAI/Florence-2-base-PromptGen-v1.5 model.
    
    Model-specific behavior:
    - Supports multiple prompt types: <GENERATE_TAGS>, <CAPTION>, <DETAILED_CAPTION>, <MORE_DETAILED_CAPTION>, <MIXED_CAPTION>
    - Special mode: <EMPTY> returns empty captions
    - Uses post_process_generation for output parsing
    - Uses fixed_get_imports patch to remove flash_attn dependency
    - Supports batch processing and image resizing
    """
    
    MODEL_ID = "MiaoshouAI/Florence-2-base-PromptGen-v1.5"
    
    def __init__(self, config):
        super().__init__(config)
        self.processor = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    @staticmethod
    def _install_transformers_flash_shim() -> None:
        import transformers.utils as transformers_utils
        try:
            import transformers.utils.import_utils as transformers_import_utils
        except Exception:
            transformers_import_utils = None
        if not hasattr(transformers_utils, "is_flash_attn_greater_or_equal_2_10"):
            def _is_flash_attn_greater_or_equal_2_10():
                return False
            setattr(transformers_utils, "is_flash_attn_greater_or_equal_2_10", _is_flash_attn_greater_or_equal_2_10)
        names = getattr(transformers_utils, "__all__", None)
        if isinstance(names, list) and "is_flash_attn_greater_or_equal_2_10" not in names:
            names.append("is_flash_attn_greater_or_equal_2_10")
        if transformers_import_utils is not None and not hasattr(transformers_import_utils, "is_flash_attn_greater_or_equal_2_10"):
            setattr(
                transformers_import_utils,
                "is_flash_attn_greater_or_equal_2_10",
                getattr(transformers_utils, "is_flash_attn_greater_or_equal_2_10"),
            )

    @staticmethod
    def _install_tokenizer_compat_shim() -> None:
        from transformers.tokenization_utils_base import PreTrainedTokenizerBase
        if getattr(PreTrainedTokenizerBase, "_ai_launcher_additional_special_tokens_patch", False):
            return
        original_getattr = PreTrainedTokenizerBase.__getattr__
        def _patched_getattr(self, key):
            if key == "additional_special_tokens":
                value = object.__getattribute__(self, "__dict__").get("additional_special_tokens", None)
                if isinstance(value, list):
                    return value
                special_map = object.__getattribute__(self, "__dict__").get("special_tokens_map", {})
                if isinstance(special_map, dict):
                    mapped = special_map.get("additional_special_tokens")
                    if isinstance(mapped, list):
                        self.additional_special_tokens = list(mapped)
                        return self.additional_special_tokens
                self.additional_special_tokens = []
                return self.additional_special_tokens
            return original_getattr(self, key)
        PreTrainedTokenizerBase.__getattr__ = _patched_getattr
        PreTrainedTokenizerBase._ai_launcher_additional_special_tokens_patch = True

    @staticmethod
    def _install_tied_weights_compat_shim() -> None:
        from transformers.modeling_utils import PreTrainedModel
        if getattr(PreTrainedModel, "_ai_launcher_tied_weights_mapping_patch", False):
            return
        original = PreTrainedModel.get_expanded_tied_weights_keys
        def _as_mapping(value):
            if isinstance(value, dict):
                return {str(k): str(v) for k, v in value.items() if str(k)}
            if isinstance(value, (list, tuple, set)):
                return {str(k): str(k) for k in list(value) if str(k)}
            return {}
        def _patched(self, *args, **kwargs):
            tied = getattr(self, "all_tied_weights_keys", None)
            mapped = _as_mapping(tied)
            if mapped:
                self.all_tied_weights_keys = mapped
            private_tied = getattr(self, "_tied_weights_keys", None)
            private_mapped = _as_mapping(private_tied)
            if private_mapped:
                self._tied_weights_keys = private_mapped
            try:
                return original(self, *args, **kwargs)
            except AttributeError as exc:
                if "keys" not in str(exc):
                    raise
                fallback = _as_mapping(getattr(self, "_tied_weights_keys", None)) or _as_mapping(getattr(self, "all_tied_weights_keys", None))
                out = []
                for k, v in fallback.items():
                    if str(k):
                        out.append(str(k))
                    if str(v):
                        out.append(str(v))
                seen = set()
                ordered = []
                for key in out:
                    if key in seen:
                        continue
                    seen.add(key)
                    ordered.append(key)
                return ordered
        PreTrainedModel.get_expanded_tied_weights_keys = _patched
        PreTrainedModel._ai_launcher_tied_weights_mapping_patch = True
    
    def _load_model(self):
        """Load MiaoshouAI model and processor with flash_attn patch."""
        if self.model is not None:
            return
        
        import warnings
        # Suppress SyntaxWarning from upstream cached files
        warnings.filterwarnings("ignore", category=SyntaxWarning, message=r"invalid escape sequence")
        
        import transformers
        from transformers import AutoModelForCausalLM, AutoProcessor
        from transformers.dynamic_module_utils import get_imports
        from transformers.configuration_utils import PretrainedConfig
        self._install_transformers_flash_shim()
        self._install_tokenizer_compat_shim()
        self._install_tied_weights_compat_shim()

        if not getattr(PretrainedConfig, "_ai_launcher_forced_token_fallback_patch", False):
            original_getattribute = PretrainedConfig.__getattribute__
            def _patched_getattribute(self, name):
                if name in {"forced_bos_token_id", "forced_eos_token_id"}:
                    try:
                        return original_getattribute(self, name)
                    except AttributeError:
                        object.__setattr__(self, name, None)
                        return None
                return original_getattribute(self, name)
            PretrainedConfig.__getattribute__ = _patched_getattribute
            PretrainedConfig._ai_launcher_forced_token_fallback_patch = True
        
        def fixed_get_imports(filename):
            """Remove flash_attn import for compatibility."""
            if not str(filename).endswith("modeling_florence2.py"):
                return get_imports(filename)
            imports = get_imports(filename)
            try:
                imports.remove("flash_attn")
            except ValueError:
                pass
            return imports

        original_linspace = torch.linspace
        def _safe_linspace(*args, **kwargs):
            out = original_linspace(*args, **kwargs)
            if getattr(getattr(out, "device", None), "type", "") != "meta":
                return out
            retry_kwargs = dict(kwargs)
            retry_kwargs["device"] = "cpu"
            return original_linspace(*args, **retry_kwargs)
        
        print(f"Loading MiaoshouAI model: {self.MODEL_ID}...")
        print(f"  Transformers version: {transformers.__version__}")
        
        # Determine attention implementation based on transformers version
        # Use eager attention for transformers >= 4.51.0 to avoid cache issues
        use_eager = transformers.__version__ >= '4.51.0'
        attention = 'eager' if use_eager else 'sdpa'
        print(f"  Using attention: {attention}")
        
        def _load_with_auto_model(model_id: str):
            with patch("transformers.dynamic_module_utils.get_imports", fixed_get_imports), patch("torch.linspace", _safe_linspace):
                with torch.device("cpu"):
                    model = AutoModelForCausalLM.from_pretrained(
                        model_id,
                        trust_remote_code=True,
                        torch_dtype=torch.float16 if self.device != "cpu" else torch.float32,
                        attn_implementation=attention,
                        low_cpu_mem_usage=False,
                        device_map=None,
                        _fast_init=False,
                    ).to(self.device).eval()
                processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
            return model, processor

        model_id = self.config.get('model_id', self.MODEL_ID)
        restored_default_device = None
        if hasattr(torch, "get_default_device") and hasattr(torch, "set_default_device"):
            try:
                restored_default_device = torch.get_default_device()
                torch.set_default_device("cpu")
            except Exception:
                restored_default_device = None
        try:
            if transformers.__version__ >= '4.51.0':
                # For transformers >= 4.51.0, try local model files first.
                from .miaoshouai_florence2 import Florence2ForConditionalGeneration
                from .miaoshouai_florence2.modeling_florence2 import Florence2LanguageForConditionalGeneration
                from transformers import AutoProcessor
                print("  Using local Florence2ForConditionalGeneration (GenerationMixin fix)")
                Florence2LanguageForConditionalGeneration._tied_weights_keys = [
                    "model.encoder.embed_tokens.weight",
                    "model.decoder.embed_tokens.weight",
                    "lm_head.weight",
                ]
                Florence2ForConditionalGeneration._tied_weights_keys = [
                    "language_model.model.encoder.embed_tokens.weight",
                    "language_model.model.decoder.embed_tokens.weight",
                    "language_model.lm_head.weight",
                ]
                try:
                    with patch("torch.linspace", _safe_linspace):
                        with torch.device("cpu"):
                            self.model = Florence2ForConditionalGeneration.from_pretrained(
                                self.MODEL_ID,
                                torch_dtype=torch.bfloat16,
                                attn_implementation=attention,
                                low_cpu_mem_usage=False,
                                device_map=None,
                                _fast_init=False,
                            ).to(self.device).eval()
                    self.processor = AutoProcessor.from_pretrained(self.MODEL_ID, trust_remote_code=True)
                except RuntimeError as exc:
                    if "meta tensors" not in str(exc):
                        raise
                    print("  Local Florence2 load hit meta tensor path, retrying with AutoModel fallback...")
                    self.model, self.processor = _load_with_auto_model(model_id)
            else:
                self.model, self.processor = _load_with_auto_model(model_id)
        finally:
            if restored_default_device is not None and hasattr(torch, "set_default_device"):
                try:
                    torch.set_default_device(restored_default_device)
                except Exception:
                    pass
        text_cfg = getattr(getattr(self.model, "config", None), "text_config", None)
        if text_cfg is not None and not hasattr(text_cfg, "forced_bos_token_id"):
            text_cfg.forced_bos_token_id = getattr(text_cfg, "bos_token_id", None)
        if text_cfg is not None and not hasattr(text_cfg, "forced_eos_token_id"):
            text_cfg.forced_eos_token_id = getattr(text_cfg, "eos_token_id", None)
        
        if torch.cuda.is_available():
            print(f"  GPU detected: {torch.cuda.get_device_name(0)}")
            print(f"  Compute Capability: {torch.cuda.get_device_capability(0)}")
            print(f"  Initial VRAM: {torch.cuda.memory_allocated()/1024**3:.2f} GB / {torch.cuda.get_device_properties(0).total_memory/1024**3:.2f} GB")
            
        print(f"MiaoshouAI loaded on {self.device}")
    
    def _run_inference(self, images: List[Image.Image], prompt: List[str], args: Dict[str, Any]) -> List[str]:
        """
        Run MiaoshouAI inference on a batch of images.
        
        Special handling:
        - <EMPTY> prompt returns empty strings
        - Uses post_process_generation for output parsing
        
        Args:
            images: List of PIL Images
            prompt: List of prompts (one per image)
            args: Dictionary of generation parameters
            
        Returns:
            List of generated captions
        """
        # Handle <EMPTY> mode - Check if ALL prompts are empty marker
        # Optimization: If any prompt is <EMPTY>, we handle it, but technically per-image check is better.
        # But if the generic "task_prompt" was <EMPTY>, then all are <EMPTY>.
        # We'll just process normally, but if a prompt is <EMPTY>, we expect empty result.
        
        max_tokens = args.get('max_tokens', 1024)
        temperature = args.get('temperature', 0.7)
        top_k = args.get('top_k', 50)
        repetition_penalty = args.get('repetition_penalty', 1.3)
        
        captions = []
        
        # Batch preparation
        # prompts is now the input list
        prompts = prompt
        
        try:
            # Prepare inputs for the whole batch
            inputs = self.processor(
                text=prompts,
                images=images,
                return_tensors="pt",
                padding=True
            )
            
            # Move to device and convert pixel_values to bfloat16
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            if 'pixel_values' in inputs:
                inputs['pixel_values'] = inputs['pixel_values'].to(torch.bfloat16)

            # Generate batch with ALL configured parameters
            with torch.no_grad():
                generated_ids = self.model.generate(
                    input_ids=inputs["input_ids"],
                    pixel_values=inputs["pixel_values"],
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    top_k=top_k,
                    repetition_penalty=repetition_penalty,
                    do_sample=True,  # Required for temperature/top_k to take effect
                    use_cache=False
                )

            # Decode batch
            generated_texts = self.processor.batch_decode(
                generated_ids,
                skip_special_tokens=False
            )

            # Post-process results
            for text, image, task_prompt_str in zip(generated_texts, images, prompts):
                if task_prompt_str == "<EMPTY>":
                    captions.append("")
                    continue
                    
                cleaned_text = text.replace('<pad>', '').strip()
                try:
                    parsed_answer = self.processor.post_process_generation(
                        cleaned_text,
                        task=task_prompt_str,
                        image_size=(image.width, image.height)
                    )
                    cap = parsed_answer.get(task_prompt_str, "")
                except Exception as e:
                    print(f"Warning: post_process_generation failed: {e}, using raw output")
                    cap = cleaned_text
                captions.append(cap)
                
        except Exception as e:
            print(f"Error during batch inference: {e}")
            # Fallback to empty strings if batch fails
            captions = [""] * len(images)
            
        return captions
    
    def unload(self):
        """Free model resources using shared utility."""
        from src.core.model_utils import unload_model, UnloadMode
        unload_model(self.model, self.processor, UnloadMode.STANDARD)
        self.model = None
        self.processor = None

