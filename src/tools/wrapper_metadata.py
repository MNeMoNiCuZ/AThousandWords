import os
import re
import json
import logging
import piexif
from PIL import Image
from typing import Dict, Any, Optional

class MetadataTool:
    def __init__(self):
        pass

    def apply_to_dataset(self, dataset, source_type: str = "all", update_caption: bool = True,
                         prefix: str = "", suffix: str = "", 
                         clean: bool = False, collapse: bool = False, normalize: bool = False,
                         output_dir: str = None, extension: str = ".txt"):
        """
        Iterates through the dataset and extracts metadata.
        source_type: 'png_info', 'exif', or 'all'
        update_caption: If True, sets the image caption to the extracted 'positive_prompt'.
        output_dir: Optional path to save captions to.
        extension: File extension for saved captions.
        """
        # Import core feature logic to avoid code duplication
        from src.features.core.text_cleanup import CleanTextFeature, CollapseNewlinesFeature
        from src.features.core.normalize_text import NormalizeTextFeature
        from pathlib import Path
        
        count = 0
        saved_count = 0
        
        # Prepare output directory if specified
        out_path_obj = Path(output_dir) if output_dir else None
        if out_path_obj and not out_path_obj.exists():
            out_path_obj.mkdir(parents=True, exist_ok=True)

        for img_obj in dataset.images:
            try:
                meta = self._extract_metadata_from_file(str(img_obj.path))
                img_obj.metadata.update(meta.get("metadata", {}))
                img_obj.metadata.update(meta.get("parsed_params", {}))
                
                # Check for prompts
                pos_prompt = meta.get("positive_prompt", "")
                if pos_prompt:
                    # Apply Text Cleaning
                    if normalize:
                        pos_prompt = NormalizeTextFeature.apply(pos_prompt)
                    if collapse:
                        pos_prompt = CollapseNewlinesFeature.apply(pos_prompt)
                    if clean:
                        pos_prompt = CleanTextFeature.apply(pos_prompt)
                        
                    # Apply Prefix/Suffix
                    if prefix:
                        pos_prompt = f"{prefix}{pos_prompt}"
                    if suffix:
                        pos_prompt = f"{pos_prompt}{suffix}"
                    
                    if update_caption:
                        img_obj.update_caption(pos_prompt)
                        # Ensure extension starts with dot
                        if extension and not extension.startswith('.'):
                            extension = '.' + extension
                            
                        # Save to disk
                        img_obj.save_caption(extension=extension, output_dir=out_path_obj)
                        saved_count += 1
                        
                    count += 1
            except Exception as e:
                logging.error(f"Error extracting metadata from {img_obj.path}: {e}")
                img_obj.error = f"Metadata error: {e}"
        
        return f"Processed {len(dataset)} images. Found metadata for {count}. Saved {saved_count} captions."

    def _parse_png_parameters(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
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
                        if ifd == "thumbnail": continue
                        readable_exif[ifd] = {}
                        for tag, value in tags.items():
                            try:
                                tag_name = piexif.TAGS.get(ifd, {}).get(tag, {}).get("name", tag)
                                readable_exif[ifd][tag_name] = value.decode('utf-8', 'ignore') if isinstance(value, bytes) else value
                            except:
                                pass
                    return {"file_path": file_path, "metadata": readable_exif, "parsed_params": {}, "positive_prompt": "", "negative_prompt": ""}
                except Exception:
                    # Non-exif image or fail
                    return {"file_path": file_path, "metadata": {}, "parsed_params": {}, "positive_prompt": "", "negative_prompt": ""}
        except Exception as e:
            return {"file_path": file_path, "error": str(e), "parsed_params": {}, "positive_prompt": "", "negative_prompt": ""}
