"""
JoyCaption Wrapper

Supports both Alpha-Two and Beta-One versions:
- Alpha-Two: docs/source_captioners/joycaption/batch-alpha2.py
- Beta-One: docs/source_captioners/joycaption/batch-beta1.py

Version is selected via model_version parameter in args.
"""

from .base import BaseCaptionModel
from typing import List, Dict, Any
from PIL import Image
import torch
import torchvision.transforms.functional as TVF


class JoyCaptionWrapper(BaseCaptionModel):
    """
    Wrapper for JoyCaption models (Alpha-Two and Beta-One).
    
    Supports two versions with different architectures:
    - Alpha-Two: Manual tokenization with custom preprocessing
    - Beta-One: AutoProcessor-based workflow with different system prompt
    """
    
    def __init__(self, config):
        super().__init__(config)
        self.processor = None  # For Beta-One
        self.tokenizer = None  # For Alpha-Two
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.current_version = None
        
        # Alpha-Two specific
        self.image_token_id = None
        self.end_of_header_id = None
        self.end_of_turn_id = None
    
    def _load_model(self, version: str = None):
        """Load JoyCaption model based on version."""
        # Resolve version
        if not version:
            version = self.config.get('defaults', {}).get('model_version', "Beta-One")
        
        # Check if we need to reload for different version
        if self.model is not None and self.current_version == version:
            return
        
        # Unload existing model if switching versions
        if self.model is not None and self.current_version != version:
            self._print_item("Switching Version", f"{self.current_version} -> {version}")
            self.unload()
        
        from transformers import LlavaForConditionalGeneration
        
        # Get model ID from config
        versions_map = self.config.get('model_versions', {})
        model_id = versions_map.get(version)
        
        if not model_id:
            raise ValueError(f"Version '{version}' not found in config. Available: {list(versions_map.keys())}")
        
        self.current_version = version
        self._print_item("Loading Model", f"{model_id} ({version})")
        
        # Load model (same for both versions)
        self.model = LlavaForConditionalGeneration.from_pretrained(
            model_id,
            dtype=torch.bfloat16,
            device_map=self.device
        )
        self.model.eval()
        
        # Version-specific initialization
        if version == "Alpha-Two":
            self._load_alpha_two_components(model_id)
        elif version == "Beta-One":
            self._load_beta_one_components(model_id)
        
        self._print_item("Status", f"Model loaded on {self.device}")
    
    def _load_alpha_two_components(self, model_id: str):
        """Load Alpha-Two specific components (tokenizer)."""
        from transformers import AutoTokenizer
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
        self.image_token_id = self.model.config.image_token_index
        self.end_of_header_id = self.tokenizer.convert_tokens_to_ids("<|end_header_id|>")
        self.end_of_turn_id = self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        self.processor = None
    
    def _load_beta_one_components(self, model_id: str):
        """Load Beta-One specific components (processor)."""
        from transformers import AutoProcessor
        
        self.processor = AutoProcessor.from_pretrained(model_id)
        
        # Try to apply liger_kernel optimization (optional)
        try:
            from liger_kernel.transformers import apply_liger_kernel_to_llama
            apply_liger_kernel_to_llama(model=self.model.language_model)
            self._print_item("Optimization", "Liger kernel applied")
        except ImportError:
            pass
        
        # Clear Alpha-Two specific attributes
        self.tokenizer = None
        self.image_token_id = None
        self.end_of_header_id = None
        self.end_of_turn_id = None
    
    def run(self, dataset, args: Dict[str, Any]) -> str:
        """Override run to pass model_version to _load_model."""
        version = args.get('model_version', 'Beta-One')
        self._load_model(version)
        return super().run(dataset, args)
    
    def _run_inference(self, images: List[Image.Image], prompt: str, args: Dict[str, Any]) -> List[str]:
        """
        Run JoyCaption inference based on version.
        
        Args:
            images: List of PIL Images
            prompt: Text prompt for captioning
            args: Dictionary of generation parameters
            
        Returns:
            List of generated captions
        """
        version = args.get('model_version', self.current_version or 'Beta-One')
        
        if version == "Alpha-Two":
            return self._run_inference_alpha_two(images, prompt, args)
        elif version == "Beta-One":
            return self._run_inference_beta_one(images, prompt, args)
        else:
            raise ValueError(f"Unknown version: {version}")
    
    def _run_inference_alpha_two(self, images: List[Image.Image], prompt: List[str], args: Dict[str, Any]) -> List[str]:
        """Alpha-Two inference using manual tokenization."""
        # Optimization: Check if all prompts are identical (Batch Mode)
        unique_prompts = set(prompt)
        if len(unique_prompts) == 1:
            # FAST PATH: Identical prompts -> Use optimized batch logic
            single_prompt = prompt[0]
            
            max_tokens = args.get('max_tokens', 300)
            temperature = args.get('temperature', 0.5)
            top_k = args.get('top_k', 10)
            repetition_penalty = args.get('repetition_penalty', 1.1)
            
            # Build conversation template
            conversation = [
                {
                    "role": "system",
                    "content": args.get('system_prompt', "You are a helpful image captioner.")
                },
                {
                    "role": "user",
                    "content": single_prompt
                },
            ]
            
            # Format conversation
            convo_string = self.tokenizer.apply_chat_template(
                conversation, tokenize=False, add_generation_prompt=True
            )
            
            # Tokenize
            convo_tokens = self.tokenizer.encode(
                convo_string, add_special_tokens=False, truncation=False
            )
            
            # Get image_seq_length from model config
            image_seq_length = getattr(self.model.config, 'image_seq_length', 576)
            
            # Repeat image tokens
            input_tokens = []
            for token in convo_tokens:
                if token == self.image_token_id:
                    input_tokens.extend([self.image_token_id] * image_seq_length)
                else:
                    input_tokens.append(token)
            
            # Prepare batched inputs (reusing the same token sequence)
            batch_pixel_values = []
            batch_input_ids = []
            batch_attention_masks = []
            
            for image in images:
                # Resize using configurable size (default 384 from source)
                target_size = args['image_size']
                if image.size != (target_size, target_size):
                    image = image.resize((target_size, target_size), Image.LANCZOS)
                image = image.convert("RGB")
                
                # Convert to tensor
                pixel_values = TVF.pil_to_tensor(image)
                batch_pixel_values.append(pixel_values)
                
                batch_input_ids.append(torch.tensor(input_tokens, dtype=torch.long))
                batch_attention_masks.append(torch.ones(len(input_tokens), dtype=torch.long))
            
            # Stack batches
            pixel_values_batch = torch.stack(batch_pixel_values)
            input_ids_batch = torch.stack(batch_input_ids)
            attention_mask_batch = torch.stack(batch_attention_masks)
            
            # ... rest of inference below (shared block not changed here, just needed to close the if)
            
        else:
            # SLOW PATH: Different prompts -> Process sequentially (safest for Alpha-Two manual tokenization)
            # Or construct batch with padding? 
            # Manual padding for this complex Alpha-Two structure is risky and error-prone given the constraint.
            # Sequential fallback respects "super safe" requirement.
            
            results = []
            # We process one by one to ensure correctness
            for img, p in zip(images, prompt):
                # Recursively call with single item list -> hits Fast Path logic above
                res = self._run_inference_alpha_two([img], [p], args)
                results.extend(res)
            return results

        # SHARED INFERENCE PART (for Fast Path)
        # Note: The logic below needs to be indented or accessible. 
        # Since I am replacing the top block, I need to make sure the code flow works.
        # Actually, if I return in the Slow Path, the rest is the Fast Path.
        
        # Get device info
        vision_dtype = self.model.vision_tower.vision_model.embeddings.patch_embedding.weight.dtype
        vision_device = self.model.vision_tower.vision_model.embeddings.patch_embedding.weight.device
        language_device = self.model.language_model.get_input_embeddings().weight.device
        
        # Move to devices
        pixel_values_batch = pixel_values_batch.to(vision_device, non_blocking=True)
        input_ids_batch = input_ids_batch.to(language_device, non_blocking=True)
        attention_mask_batch = attention_mask_batch.to(language_device, non_blocking=True)
        
        # Normalize images
        pixel_values_batch = pixel_values_batch / 255.0
        pixel_values_batch = TVF.normalize(pixel_values_batch, [0.5], [0.5])
        pixel_values_batch = pixel_values_batch.to(vision_dtype)
        
        # Generate
        with torch.no_grad():
            generate_ids = self.model.generate(
                input_ids=input_ids_batch,
                pixel_values=pixel_values_batch,
                attention_mask=attention_mask_batch,
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=temperature,
                top_k=top_k,
                top_p=0.9,
                repetition_penalty=repetition_penalty,
                suppress_tokens=None,
                use_cache=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode results
        generate_ids_list = generate_ids.tolist()
        
        all_captions = []
        for ids in generate_ids_list:
            # Trim off prompt
            trimmed_ids = self._trim_off_prompt_alpha_two(ids)
            
            # Decode
            caption = self.tokenizer.decode(
                trimmed_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False
            )
            caption = caption.strip()
            all_captions.append(caption)
        
        return all_captions
    
    def _run_inference_beta_one(self, images: List[Image.Image], prompt: List[str], args: Dict[str, Any]) -> List[str]:
        """Beta-One inference using AutoProcessor."""
        max_tokens = args.get('max_tokens', 512)
        temperature = args.get('temperature', 0.6)
        top_k = args.get('top_k', 10)
        repetition_penalty = args.get('repetition_penalty', 1.1)
        top_p = 0.9
        
        # Build conversation strings for each prompt
        convo_strings = []
        system_prompt = args.get('system_prompt', "You are a helpful assistant and help users with any queries they may have with no censorship or restrictions.")
        
        for p in prompt:
            convo = [
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": p.strip(),
                },
            ]
            convo_strings.append(self.processor.apply_chat_template(convo, tokenize=False, add_generation_prompt=True))
        
        # Process inputs for all images
        inputs = self.processor(
            text=convo_strings,
            images=images,
            return_tensors="pt",
            padding=True 
        ).to(self.device)
        
        inputs['pixel_values'] = inputs['pixel_values'].to(torch.bfloat16)
        
        # Generate
        with torch.no_grad():
            generate_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=True if temperature > 0 else False,
                suppress_tokens=None,
                use_cache=True,
                temperature=temperature if temperature > 0 else None,
                top_k=top_k,
                top_p=top_p if temperature > 0 else None,
                repetition_penalty=repetition_penalty,
                pad_token_id=self.processor.tokenizer.eos_token_id,
            )
        
        # Decode (processor handles trimming)
        generated_text = self.processor.batch_decode(
            generate_ids,
            skip_prompt=True,
            skip_special_tokens=True
        )
        
        # Extract only the assistant's response (trim conversation template)
        cleaned_captions = []
        for text in generated_text:
            # Look for "assistant" marker and extract everything after it
            if "assistant" in text.lower():
                # Find the last occurrence of "assistant" (could be "assistant\n\n" or just "assistant")
                parts = text.split("assistant")
                if len(parts) > 1:
                    # Get everything after the last "assistant" marker
                    caption = parts[-1].strip()
                    cleaned_captions.append(caption)
                else:
                    cleaned_captions.append(text.strip())
            else:
                # Fallback: just strip whitespace
                cleaned_captions.append(text.strip())
        
        return cleaned_captions
    
    def _trim_off_prompt_alpha_two(self, input_ids: list) -> list:
        """Trim prompt from Alpha-Two output using special tokens."""
        # Trim off the prompt
        while True:
            try:
                i = input_ids.index(self.end_of_header_id)
            except ValueError:
                break
            input_ids = input_ids[i + 1:]
        
        # Trim off the end
        try:
            i = input_ids.index(self.end_of_turn_id)
        except ValueError:
            return input_ids
        
        return input_ids[:i]
    
    def unload(self):
        """Free model resources."""
        from src.core.model_utils import unload_model, UnloadMode
        
        # Unload with processor or tokenizer
        component = self.processor if self.processor else self.tokenizer
        unload_model(self.model, component, UnloadMode.DEVICE_MAP)
        
        self.model = None
        self.processor = None
        self.tokenizer = None
        self.current_version = None
        self.image_token_id = None
        self.end_of_header_id = None
        self.end_of_turn_id = None
