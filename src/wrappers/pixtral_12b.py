"""
Pixtral 12B Wrapper

Based on: docs/source_captioners/pixtral_12b/batch.py
Model Versions:
  - SeanScripts/pixtral-12b-nf4 (quantized, low VRAM)
  - mistral-community/pixtral-12b (full model)

This wrapper implements the Mistral Pixtral 12B model using LlavaForConditionalGeneration.
"""

from .base import BaseCaptionModel
from typing import List, Dict, Any
from PIL import Image
import torch


class Pixtral12BWrapper(BaseCaptionModel):
    """
    Wrapper for Mistral Pixtral 12B model.
    
    Model-specific behavior:
    - Uses LlavaForConditionalGeneration architecture
    - Supports both quantized (nf4) and full model variants
    - Uses special prompt format: <s>[INST]{prompt}\n[IMG][/INST]
    - Strips user prompt from generated output
    - Supports temperature, top_k, repetition_penalty (but not used in generation per source)
    """
    
    
    # Model version options
    # versions are defined in config/models/pixtral_12b.yaml

    
    def __init__(self, config):
        super().__init__(config)
        self.processor = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.current_version = None
        self.user_prompt_text = None
    
    def _load_model(self, version: str = "Quantized (nf4)"):
        """Load Pixtral 12B model and processor."""
        # Check if we need to reload for different version
        if self.model is not None and self.current_version == version:
            return
        

        # Unload existing model if switching versions
        if self.model is not None and self.current_version != version:
            print(f"Switching model from {self.current_version} to {version}...")
            self.unload()
        
        from transformers import LlavaForConditionalGeneration, AutoProcessor
        
        # Get model ID from config versions
        versions = self.config.get('model_versions', {})
        # If config is missing, fallback to explicit strings (safe default)
        if not versions:
            versions = {
               "Quantized (nf4)": "SeanScripts/pixtral-12b-nf4",
               "Full Model": "mistral-community/pixtral-12b",
            }
            
        model_id = versions.get(version, versions.get("Quantized (nf4)"))
        self.current_version = version
        
        print(f"Loading Pixtral 12B ({version}): {model_id}...")
        
        try:
            if version == "Quantized (nf4)":
                # Quantized model - load efficiently with safetensors
                self.model = LlavaForConditionalGeneration.from_pretrained(
                    model_id,
                    use_safetensors=True,
                    device_map=self.device
                )
                print("Using quantized (nf4) model")
            else:
                # Full model
                self.model = LlavaForConditionalGeneration.from_pretrained(
                    model_id,
                    dtype=torch.bfloat16,
                    device_map=self.device
                )
                print("Using full model")
            
            self.processor = AutoProcessor.from_pretrained(model_id)
            # Set pad_token to eos_token for batch processing
            self.processor.tokenizer.pad_token = self.processor.tokenizer.eos_token
            print(f"Pixtral loaded on {self.device}")
        except Exception as e:
            print(f"Error loading Pixtral 12B model {model_id}: {e}")
            self.model = None
            self.processor = None
            self.current_version = None
            raise
    
    def run(self, dataset, args: Dict[str, Any]) -> str:
        """Override run to pass model_version to _load_model."""
        version = args.get('model_version', 'Quantized (nf4)')
        self._load_model(version)
        return super().run(dataset, args)
    
    def _run_inference(self, images: List[Image.Image], prompt: str, args: Dict[str, Any]) -> List[str]:
        """
        Run Pixtral inference on a batch of images.
        
        Note: Source script doesn't use temperature/top_k/repetition_penalty in generate.
        This follows the source script's implementation.
        
        Args:
            images: List of PIL Images
            prompt: Text prompt for captioning (with prefix/suffix already applied)
            args: Dictionary of generation parameters
            
        Returns:
            List of generated captions
        """
        # Extract ALL generation parameters from args
        max_tokens = args.get('max_tokens', 512)
        temperature = args.get('temperature', 0.7)
        top_k = args.get('top_k', 50)
        repetition_penalty = args.get('repetition_penalty', 1.5)
        
        # CRITICAL: Pixtral processor requires [IMG] token in prompt to insert image features
        # If missing, processor won't know where to place image and generate() will fail
        if '[IMG]' not in prompt:
            # Add [IMG] in standard Pixtral format
            if prompt.startswith("<s>[INST]"):
                # Has structure, add [IMG] before closing
                prompt = prompt.replace("[/INST]", "\n[IMG][/INST]")
            elif "[INST]" in prompt and "[/INST]" in prompt:
                # Has INST tags, add [IMG] before closing
                prompt = prompt.replace("[/INST]", "\n[IMG][/INST]")
            else:
                # Plain text prompt - wrap in full format
                prompt = f"<s>[INST]{prompt}\n[IMG][/INST]"
        
        # Extract the user prompt text from the prompt for stripping later
        # The prompt comes formatted as: <s>[INST]{USER_PROMPT}\n[IMG][/INST]
        # We need to extract USER_PROMPT to strip it from output
        user_prompt_text = prompt
        if prompt.startswith("<s>[INST]"):
            user_prompt_text = prompt.replace("<s>[INST]", "").replace("\n[IMG][/INST]", "").strip()
        elif "[INST]" in prompt:
            user_prompt_text = prompt.split("[INST]")[1].split("[/INST]")[0].replace("\n[IMG]", "").strip()
        
        self.user_prompt_text = user_prompt_text
        
        # Handle system prompt manually for Pixtral (no standard system role in simple templates)
        # Note: Mistral/Pixtral usually puts system prompt inside [INST] or before it.
        # We will prepend it to the text prompts.
        system_prompt = args.get('system_prompt')
        final_prompt = prompt
        
        if system_prompt:
             # If using [INST] format, insert system prompt inside? 
             # Common practice: [INST] System Prompt\n\nUser Prompt [/INST]
             if "[INST]" in final_prompt:
                 # Check if it starts with <s>[INST]
                 if final_prompt.startswith("<s>[INST]"):
                     final_prompt = final_prompt.replace("<s>[INST]", f"<s>[INST]{system_prompt}\n\n", 1)
                 elif final_prompt.startswith("[INST]"):
                     final_prompt = final_prompt.replace("[INST]", f"[INST]{system_prompt}\n\n", 1)
             else:
                 # Plain text, just prepend
                 final_prompt = f"{system_prompt}\n\n{final_prompt}"
        
        # Helper list of prompts (one per image)
        prompts = [final_prompt] * len(images)
        
        try:
            # Prepare inputs - allow processor to handle batching
            inputs = self.processor(images=images, text=prompts, padding=True, return_tensors="pt")
            
            # Move to device and convert to model's dtype (float16)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            if 'pixel_values' in inputs:
                inputs['pixel_values'] = inputs['pixel_values'].to(self.model.dtype)
            
            # Generate with ALL configured parameters
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs, 
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    top_k=top_k,
                    repetition_penalty=repetition_penalty,
                    do_sample=True,  # Required for temperature/top_k to take effect
                    pad_token_id=self.processor.tokenizer.eos_token_id
                )
            
            # Decode the output
            output_texts = self.processor.batch_decode(
                generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            
            # Strip the user prompt from the output
            captions = []
            for output_text in output_texts:
                if self.user_prompt_text and output_text.startswith(self.user_prompt_text):
                    output_text = output_text[len(self.user_prompt_text):].strip()
                captions.append(output_text)

            return captions


        except Exception as e:
            # Detect OOM errors
            is_oom = "out of memory" in str(e).lower()
            
            if is_oom:
                # Print non-fatal OOM in light red
                from src.core.console_kit import console, Fore
                console.print("\n⚠️  BATCH OOM ERROR (Non-fatal - continuing with next batch)", color=Fore.LIGHTRED_EX, force=True)
                console.print(f"   {str(e)}", color=Fore.LIGHTRED_EX, force=True)
                console.print("   💡 Try reducing batch size to avoid this error\n", color=Fore.YELLOW, force=True)
            else:
                print(f"Error during Pixtral batched inference: {e}")
            
            # Fallback to empty strings for batch size to maintain alignment (imperfect)
            return [""] * len(images)
    
    def unload(self):
        """Free model resources using shared utility."""
        from src.core.model_utils import unload_model, UnloadMode
        
        # Select mode based on version
        if self.current_version == "Quantized (nf4)":
            mode = UnloadMode.DEVICE_MAP
        else:
            mode = UnloadMode.STANDARD
        
        unload_model(self.model, self.processor, mode)
        self.model = None
        self.processor = None
        self.current_version = None
        self.user_prompt_text = None

