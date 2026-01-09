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
        
        print(f"Loading MiaoshouAI model: {self.MODEL_ID}...")
        print(f"  Transformers version: {transformers.__version__}")
        
        # Determine attention implementation based on transformers version
        # Use eager attention for transformers >= 4.51.0 to avoid cache issues
        use_eager = transformers.__version__ >= '4.51.0'
        attention = 'eager' if use_eager else 'sdpa'
        print(f"  Using attention: {attention}")
        
        if transformers.__version__ >= '4.51.0':
            # For transformers >= 4.51.0, use local model files that properly inherit GenerationMixin
            from .miaoshouai_florence2 import Florence2ForConditionalGeneration
            from transformers import AutoProcessor
            print("  Using local Florence2ForConditionalGeneration (GenerationMixin fix)")
            
            self.model = Florence2ForConditionalGeneration.from_pretrained(
                self.MODEL_ID,
                dtype=torch.bfloat16,
                attn_implementation=attention
            ).to(self.device)
            
            # CRITICAL: Must also load the processor
            self.processor = AutoProcessor.from_pretrained(
                self.MODEL_ID,
                trust_remote_code=True
            )
        else:
            # For older transformers, use the standard AutoModelForCausalLM
            # Use config model_id
            model_id = self.config.get('model_id', self.MODEL_ID)
            
            with patch("transformers.dynamic_module_utils.get_imports", fixed_get_imports):
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    trust_remote_code=True,
                    dtype=torch.float16 if self.device != "cpu" else torch.float32,  # Use float16 for GPU
                    attn_implementation=attention
                ).to(self.device).eval()
                
                self.processor = AutoProcessor.from_pretrained(
                    model_id,
                    trust_remote_code=True
                )
        
        if torch.cuda.is_available():
            print(f"  GPU detected: {torch.cuda.get_device_name(0)}")
            print(f"  Compute Capability: {torch.cuda.get_device_capability(0)}")
            print(f"  Initial VRAM: {torch.cuda.memory_allocated()/1024**3:.2f} GB / {torch.cuda.get_device_properties(0).total_memory/1024**3:.2f} GB")
            
        print(f"MiaoshouAI loaded on {self.device}")
    
    def _run_inference(self, images: List[Image.Image], prompt: str, args: Dict[str, Any]) -> List[str]:
        """
        Run MiaoshouAI inference on a batch of images.
        
        Special handling:
        - <EMPTY> prompt returns empty strings
        - Uses post_process_generation for output parsing
        
        Args:
            images: List of PIL Images
            prompt: Task prompt
            args: Dictionary of generation parameters
            
        Returns:
            List of generated captions
        """
        # Handle <EMPTY> mode
        if prompt == "<EMPTY>":
            return [""] * len(images)
        
        max_tokens = args.get('max_tokens', 1024)
        temperature = args.get('temperature', 0.7)
        top_k = args.get('top_k', 50)
        repetition_penalty = args.get('repetition_penalty', 1.3)
        
        captions = []
        
        # Batch preparation
        prompts = [prompt] * len(images)
        
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
            for text, image in zip(generated_texts, images):
                cleaned_text = text.replace('<pad>', '').strip()
                try:
                    parsed_answer = self.processor.post_process_generation(
                        cleaned_text,
                        task=prompt,
                        image_size=(image.width, image.height)
                    )
                    cap = parsed_answer.get(prompt, "")
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

