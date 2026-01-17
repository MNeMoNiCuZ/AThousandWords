"""
Metadata Extraction Tool

Extracts metadata (User Comment, PNG Info, etc.) from images to create captions.
Implements BaseTool for auto-discovery and self-contained GUI.
"""

import os
import re
import logging
import gradio as gr
import piexif
from PIL import Image
from typing import Dict, Any

from .base import BaseTool, ToolConfig


class MetadataTool(BaseTool):
    """Tool for extracting image metadata and creating captions."""
    
    @property
    def config(self) -> ToolConfig:
        return ToolConfig(
            name="metadata_extractor",
            display_name="Metadata Extractor",
            description="### Extract Metadata to Captions\nExtract prompts from PNG Info, EXIF, etc. to create caption files.",
            icon=""
        )
    
    def apply_to_dataset(self, dataset, source_type: str = "all", update_caption: bool = True,
                         prefix: str = "", suffix: str = "", 
                         clean: bool = False, collapse: bool = False, normalize: bool = False,
                         output_dir: str = None, extension: str = ".txt") -> str:
        """
        Iterates through the dataset and extracts metadata.
        
        Args:
            dataset: Dataset object containing images
            source_type: 'png_info', 'exif', or 'all'
            update_caption: If True, sets the image caption to the extracted 'positive_prompt'
            prefix: Text to add at the start of each caption
            suffix: Text to add at the end of each caption
            clean: Remove extra spaces
            collapse: Merge paragraphs (collapse newlines)
            normalize: Fix punctuation
            output_dir: Optional path to save captions to
            extension: File extension for saved captions
            
        Returns:
            str: Status message
        """
        from src.features.core.text_cleanup import CleanTextFeature, CollapseNewlinesFeature
        from src.features.core.normalize_text import NormalizeTextFeature
        from pathlib import Path
        
        # ANSI colors
        CYAN = "\033[96m"
        GREEN = "\033[92m"
        YELLOW = "\033[93m"
        RED = "\033[91m"
        RESET = "\033[0m"
        DIM = "\033[2m"
        
        count = 0
        saved_count = 0
        no_meta_count = 0
        error_count = 0
        
        # Log settings
        print(f"{CYAN}Settings:{RESET}")
        print(f"  Source: {source_type}")
        print(f"  Output Dir: {output_dir or '(same as source)'}")
        print(f"  Extension: {extension}")
        print(f"  Overwrite Captions: {update_caption}")
        opts = []
        if clean: opts.append("Clean")
        if collapse: opts.append("Collapse")
        if normalize: opts.append("Normalize")
        print(f"  Text Processing: {', '.join(opts) if opts else '(none)'}")
        if prefix or suffix:
            print(f"  Prefix: '{prefix}' | Suffix: '{suffix}'")
        print(f"")
        
        out_path_obj = Path(output_dir) if output_dir else None
        if out_path_obj and not out_path_obj.exists():
            out_path_obj.mkdir(parents=True, exist_ok=True)

        for img_obj in dataset.images:
            try:
                meta = self._extract_metadata_from_file(str(img_obj.path))
                img_obj.metadata.update(meta.get("metadata", {}))
                img_obj.metadata.update(meta.get("parsed_params", {}))
                
                pos_prompt = meta.get("positive_prompt", "")
                if pos_prompt:
                    if normalize:
                        pos_prompt = NormalizeTextFeature.apply(pos_prompt)
                    if collapse:
                        pos_prompt = CollapseNewlinesFeature.apply(pos_prompt)
                    if clean:
                        pos_prompt = CleanTextFeature.apply(pos_prompt)
                        
                    if prefix:
                        pos_prompt = f"{prefix}{pos_prompt}"
                    if suffix:
                        pos_prompt = f"{pos_prompt}{suffix}"
                    
                    if update_caption:
                        img_obj.update_caption(pos_prompt)
                        if extension and not extension.startswith('.'):
                            extension = '.' + extension
                        img_obj.save_caption(extension=extension, output_dir=out_path_obj)
                        saved_count += 1
                        
                    count += 1
                    preview = pos_prompt[:60] + "..." if len(pos_prompt) > 60 else pos_prompt
                    print(f"  {GREEN}✓{RESET} {Path(img_obj.path).name}: {DIM}{preview}{RESET}")
                else:
                    no_meta_count += 1
                    print(f"  {YELLOW}○{RESET} {Path(img_obj.path).name}: {DIM}(no metadata found){RESET}")
            except Exception as e:
                error_count += 1
                print(f"  {RED}✗{RESET} {Path(img_obj.path).name}: {DIM}{e}{RESET}")
                logging.error(f"Error extracting metadata from {img_obj.path}: {e}")
                img_obj.error = f"Metadata error: {e}"
        
        print(f"")
        print(f"{CYAN}Summary:{RESET} {GREEN}Found: {count}{RESET} | {YELLOW}No Metadata: {no_meta_count}{RESET} | {RED}Errors: {error_count}{RESET} | Saved: {saved_count}")
        
        result = f"Processed {len(dataset)} images. Found metadata for {count}. Saved {saved_count} captions."
        return result
    
    def create_gui(self, app) -> tuple:
        """Create the Metadata tool UI. Returns (run_button, inputs) for later event wiring."""
        
        gr.Markdown(self.config.description)
        
        with gr.Row():
            meta_src = gr.Dropdown(
                ["all", "png_info", "exif"], 
                label="Source", 
                value="all", 
                info="Which metadata field to search."
            )
            meta_out_dir = gr.Textbox(
                label="Output Directory", 
                placeholder="Optional. Leave empty to save in same folder.", 
                info="Path to save text files."
            )
            meta_ext = gr.Textbox(
                label="Output Extension", 
                value="txt", 
                placeholder="txt", 
                info="Extension for caption files."
            )
            meta_upd = gr.Checkbox(
                label="Overwrite existing captions", 
                value=True
            )
        
        with gr.Row():
            meta_clean = gr.Checkbox(label="Clean Text", value=True, info="Remove extra spaces")
            meta_collapse = gr.Checkbox(label="Collapse Newlines", value=True, info="Merge paragraphs")
            meta_norm = gr.Checkbox(label="Normalize Text", value=True, info="Fix punctuation")
            
        with gr.Row():
            meta_pre = gr.Textbox(label="Prefix", placeholder="Added to start...", lines=1)
            meta_suf = gr.Textbox(label="Suffix", placeholder="Added to end...", lines=1)

        meta_run = gr.Button("Extract Metadata", variant="primary", elem_id="metadata_tool_btn")
        
        # Return components for later event wiring
        inputs = [meta_src, meta_upd, meta_pre, meta_suf, meta_clean, meta_collapse, meta_norm, meta_out_dir, meta_ext]
        return (meta_run, inputs)
    
    def _parse_png_parameters(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Parse PNG parameters string into structured data."""
        parsed_data = {"positive_prompt": "", "negative_prompt": "", "parsed_params": {}}
        params_str = metadata.get('metadata', {}).get('parameters', '')
        if not isinstance(params_str, str):
            metadata.update(parsed_data)
            return metadata

        neg_prompt_index = params_str.find('Negative prompt:')
        steps_index = params_str.find('Steps:')
        
        if neg_prompt_index != -1:
            parsed_data['positive_prompt'] = params_str[:neg_prompt_index].strip()
            end_index = steps_index if steps_index != -1 else len(params_str)
            parsed_data['negative_prompt'] = params_str[neg_prompt_index + len('Negative prompt:'):end_index].strip()
        elif steps_index != -1:
            parsed_data['positive_prompt'] = params_str[:steps_index].strip()
        else:
            parsed_data['positive_prompt'] = params_str.strip()

        parsed_data['parsed_params']['positive_prompt'] = parsed_data['positive_prompt']
        parsed_data['parsed_params']['negative_prompt'] = parsed_data['negative_prompt']

        param_patterns = {
            'steps': r'Steps: (.*?)(?:,|$)', 'sampler': r'Sampler: (.*?)(?:,|$)',
            'cfg scale': r'CFG scale: (.*?)(?:,|$)', 'seed': r'Seed: (.*?)(?:,|$)',
            'size': r'Size: (.*?)(?:,|$)', 'model': r'Model: (.*?)(?:,|$)',
            'denoising strength': r'Denoising strength: (.*?)(?:,|$)', 'clip skip': r'Clip skip: (.*?)(?:,|$)',
            'hires upscale': r'Hires upscale: (.*?)(?:,|$)', 'hires steps': r'Hires steps: (.*?)(?:,|$)',
            'hires upscaler': r'Hires upscaler: (.*?)(?:,|$)', 'lora hashes': r'Lora hashes: "(.*?)"(?:,|$)'
        }
        for key, pattern in param_patterns.items():
            match = re.search(pattern, params_str, re.IGNORECASE)
            if match:
                parsed_data['parsed_params'][key] = match.group(1).strip()
        
        metadata.update(parsed_data)
        return metadata

    def _extract_metadata_from_file(self, file_path: str) -> Dict[str, Any]:
        """Extract metadata from image file."""
        ext = os.path.splitext(file_path)[1].lower()
        try:
            if ext == '.png':
                with Image.open(file_path) as img: 
                    info = dict(img.info)
                    
                metadata = {"file_path": file_path, "metadata": info}
                return self._parse_png_parameters(metadata)
            else:
                try:
                    exif_dict = piexif.load(file_path)
                    readable_exif = {}
                    for ifd, tags in exif_dict.items():
                        if ifd == "thumbnail":
                            continue
                        readable_exif[ifd] = {}
                        for tag, value in tags.items():
                            try:
                                tag_name = piexif.TAGS.get(ifd, {}).get(tag, {}).get("name", tag)
                                readable_exif[ifd][tag_name] = value.decode('utf-8', 'ignore') if isinstance(value, bytes) else value
                            except:
                                pass
                    return {"file_path": file_path, "metadata": readable_exif, "parsed_params": {}, "positive_prompt": "", "negative_prompt": ""}
                except Exception:
                    return {"file_path": file_path, "metadata": {}, "parsed_params": {}, "positive_prompt": "", "negative_prompt": ""}
        except Exception as e:
            return {"file_path": file_path, "error": str(e), "parsed_params": {}, "positive_prompt": "", "negative_prompt": ""}
