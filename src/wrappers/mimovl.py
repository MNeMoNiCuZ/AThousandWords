"""
MimoVL Wrapper

Based on: docs/source_captioners/mimovl/batch.py
Model: XiaomiMiMo/MiMo-VL-7B-RL

This wrapper implements the MimoVL captioner using Qwen2.5-VL architecture.
"""

from .base import BaseCaptionModel
from typing import List, Dict, Any
from PIL import Image
import torch
import re
from src.core.console_kit import console, Fore


class MimoVLWrapper(BaseCaptionModel):
    """
    Wrapper for MiMo-VL 7B RL model.
    
    Model-specific behavior:
    - Uses Qwen2_5_VLForConditionalGeneration
    - Handles thinking/reasoning tags (model produces <think>...</think>)
    - Supports temperature, top_k, repetition_penalty
    """
    
    MODEL_ID = "XiaomiMiMo/MiMo-VL-7B-RL"
    
    def __init__(self, config):
        super().__init__(config)
        self.processor = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    def _load_model(self):
        """Load MimoVL model and processor."""
        if self.model is not None:
            return
        
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        
        console.print(f"Loading MimoVL model: {self.MODEL_ID}...", color=Fore.CYAN, force=True)
        
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.MODEL_ID,
            dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
            device_map="auto"
        )
        
        self.processor = AutoProcessor.from_pretrained(self.MODEL_ID, use_fast=True)
        self.processor.tokenizer.padding_side = 'left'
        
        console.print(f"MimoVL loaded on {self.device}", color=Fore.GREEN, force=True)
    
    def _run_inference(self, images: List[Image.Image], prompt: str, args: Dict[str, Any]) -> List[str]:
        """
        Run MimoVL inference on a batch of images.
        
        Returns raw captions (thinking tags stripped by default), or JSON string if output_json is True.
        """
        # Get model-specific parameters
        temperature = args.get('temperature', 0.8)
        top_k = args.get('top_k', 50)
        repetition_penalty = args.get('repetition_penalty', 1.2)
        max_tokens = args.get('max_tokens', 4096)
        system_prompt = args.get('system_prompt', 'You are a helpful image captioning model.')
        strip_thinking = args.get('strip_thinking_tags', True)
        output_json = args.get('output_json', False)
        
        # Build conversations for each image
        conversations = []
        for _ in images:
            conversation = [
                {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
                {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}
            ]
            conversations.append(conversation)
        
        # Prepare text prompts
        text_prompts = [self.processor.apply_chat_template(conv, add_generation_prompt=True) for conv in conversations]
        
        # Prepare inputs
        inputs = self.processor(
            text=text_prompts,
            images=images,
            padding=True,
            return_tensors="pt",
        ).to(self.device)
        
        # Generate
        with torch.no_grad():
            with torch.autocast(device_type=self.device, dtype=torch.bfloat16):
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    do_sample=True,
                    temperature=temperature,
                    use_cache=True,
                    top_k=top_k,
                    repetition_penalty=repetition_penalty,
                )
        
        # Decode
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, output_ids)
        ]
        output_texts = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        
        # Process outputs
        captions = []
        for output_text in output_texts:
            # Extract thinking text and caption
            think_match = re.search(r'<think>(.*?)</think>', output_text, re.DOTALL)
            if think_match:
                thinking_text = think_match.group(1).strip()
                caption = re.sub(r'<think>.*?</think>', '', output_text).strip()
            else:
                thinking_text = ""
                caption = re.sub(r'<think>.*$', '', output_text).strip()
            
            # Clean text (remove markdown formatting)
            caption = self._clean_text(caption)
            thinking_text = self._clean_text(thinking_text)
            
            if output_json:
                import json
                data = {
                    "thinking_text": thinking_text,
                    "caption_text": caption
                }
                final_caption = json.dumps(data, indent=4, ensure_ascii=False)
            else:
                # Include thinking if requested
                if not strip_thinking and thinking_text:
                    final_caption = f"<think>{thinking_text}</think>{caption}"
                else:
                    # If caption too short and we have thinking, use thinking instead
                    if len(caption) < 10 and thinking_text:
                        final_caption = thinking_text
                    else:
                        final_caption = caption
            
            captions.append(final_caption)
        
        return captions
    
    def _clean_text(self, text: str) -> str:
        """Clean markdown formatting from text."""
        text = re.sub(r'^#{1,6}\s*', '', text, flags=re.MULTILINE)
        text = re.sub(r'^[-*]\s+', '', text, flags=re.MULTILINE)
        text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
        text = re.sub(r'\*(.*?)\*', r'\1', text)
        text = re.sub(r'_(.*?)_', r'\1', text)
        text = re.sub(r'\[(.*?)\]\(.*?\)', r'\1', text)
        text = re.sub(r'`(.*?)`', r'\1', text)
        text = re.sub(r'^>\s*(.*?)$', r'\1', text, flags=re.MULTILINE)
        text = re.sub(r'^\d+\.\s*(.*?)$', r'\1', text, flags=re.MULTILINE)
        text = re.sub(r'[–—]', '-', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def unload(self):
        """Free model resources using shared utility."""
        from src.core.model_utils import unload_model, UnloadMode
        unload_model(self.model, self.processor, UnloadMode.DEVICE_MAP)
        self.model = None
        self.processor = None

