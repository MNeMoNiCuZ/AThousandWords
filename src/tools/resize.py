"""
Image Resize Tool

Batch resize images to a maximum dimension while preserving aspect ratio.
Implements BaseTool for auto-discovery and self-contained GUI.
"""

import logging
import gradio as gr
from PIL import Image
from pathlib import Path

from .base import BaseTool, ToolConfig


class ResizeTool(BaseTool):
    """Tool for batch resizing images."""
    
    @property
    def config(self) -> ToolConfig:
        return ToolConfig(
            name="resize",
            display_name="Resize",
            description="""### Resize Images
Batch resize all loaded images to a maximum dimension while preserving aspect ratio.
Images smaller than the target are NOT upscaled.

> [!CAUTION]
> **Irreversible Action**: Resizing images is a destructive operation if you overwrite the originals. 
> It is highly recommended to use a separate Output Directory or Prefix/Suffix.""",
            icon=""
        )
    
    def apply_to_dataset(self, dataset, max_dim: int, output_dir: str = None,
                         prefix: str = "", suffix: str = "", extension: str = "",
                         overwrite: bool = True) -> str:
        """
        Resize all images in dataset to max dimension.
        
        Args:
            dataset: Dataset object containing images
            max_dim: Maximum dimension (width or height) in pixels
            output_dir: Optional path to save resized images
            prefix: Text to add at the start of each filename
            suffix: Text to add at the end of each filename (before extension)
            extension: Output file extension (empty = keep original)
            overwrite: Whether to overwrite existing files
            
        Returns:
            str: Status message
        """
        # ANSI colors
        CYAN = "\033[96m"
        GREEN = "\033[92m"
        YELLOW = "\033[93m"
        RED = "\033[91m"
        RESET = "\033[0m"
        DIM = "\033[2m"
        
        count = 0
        skipped = 0
        errors = 0
        
        # Log settings
        print(f"{CYAN}Settings:{RESET}")
        print(f"  Max Dimension: {max_dim}px")
        print(f"  Output Dir: {output_dir or '(same as source)'}")
        print(f"  Prefix: '{prefix}' | Suffix: '{suffix}' | Extension: {extension or '(keep original)'}")
        print(f"  Overwrite: {overwrite}")
        print(f"")
        
        out_path_obj = Path(output_dir) if output_dir else None
        if out_path_obj and not out_path_obj.exists():
            out_path_obj.mkdir(parents=True, exist_ok=True)
            
        for img_obj in dataset.images:
            src_path = Path(img_obj.path)
            base_name = src_path.stem
            
            if extension and not extension.startswith('.'):
                ext = '.' + extension
            else:
                ext = extension or src_path.suffix
                
            new_name = f"{prefix}{base_name}{suffix}{ext}"
            
            if out_path_obj:
                target_path = str(out_path_obj / new_name)
            else:
                target_path = str(src_path.with_name(new_name))
            
            success, msg = self._resize_image_file(str(src_path), int(max_dim), target_path, overwrite)
            
            if success:
                count += 1
                print(f"  {GREEN}✓{RESET} {src_path.name} → {new_name} {DIM}({msg}){RESET}")
                logging.info(f"Resized: {src_path.name} -> {new_name} ({msg})")
            elif "Skipped" in msg:
                skipped += 1
                print(f"  {YELLOW}○{RESET} {src_path.name} {DIM}({msg}){RESET}")
                logging.debug(f"Skipped: {src_path.name} ({msg})")
            else:
                errors += 1
                print(f"  {RED}✗{RESET} {src_path.name} {DIM}({msg}){RESET}")
                logging.error(f"Error: {src_path.name} ({msg})")
        
        print(f"")
        print(f"{CYAN}Summary:{RESET} {GREEN}Resized: {count}{RESET} | {YELLOW}Skipped: {skipped}{RESET} | {RED}Errors: {errors}{RESET}")
                
        result = f"Processed {len(dataset)} images. Resized/Saved: {count}, Skipped: {skipped}, Errors: {errors}"
        return result
    
    def create_gui(self, app) -> tuple:
        """Create the Resize tool UI. Returns (run_button, inputs) for later event wiring."""
        
        gr.Markdown(self.config.description)
        
        with gr.Row():
            resize_out_dir = gr.Textbox(
                label="Output Directory", 
                placeholder="Optional. Leave empty to save in same folder.", 
                info="Path to save resized images."
            )
            resize_pre = gr.Textbox(
                label="Output Filename Prefix", 
                placeholder="Added to start...", 
                lines=1
            )
            resize_suf = gr.Textbox(
                label="Output Filename Suffix", 
                placeholder="Added to end...", 
                lines=1
            )
            resize_ext = gr.Textbox(
                label="Output Extension", 
                value="", 
                placeholder="Keep Original", 
                info="Ext (jpg, png). Empty = Keep Original."
            )
            
        with gr.Row():
            resize_px = gr.Number(label="Max Dimension (px)", value=1024, precision=0)
            resize_overwrite = gr.Checkbox(label="Overwrite", value=True, info="Overwrite if file exists")
            
        resize_run = gr.Button("Resize Images", variant="primary", elem_id="resize_tool_btn")
        
        # Return components for later event wiring
        inputs = [resize_px, resize_out_dir, resize_pre, resize_suf, resize_ext, resize_overwrite]
        return (resize_run, inputs)
    
    def _resize_image_file(self, image_path: str, max_dimension: int, output_path: str = None, 
                           overwrite: bool = True) -> tuple:
        """
        Resize a single image.
        
        Args:
            image_path: Path to source image
            max_dimension: Maximum dimension in pixels
            output_path: Path to save resized image (None = overwrite source)
            overwrite: Whether to overwrite existing files
            
        Returns:
            tuple: (success: bool, message: str)
        """
        import os
        try:
            target = output_path if output_path else image_path
            if not overwrite and os.path.exists(target) and target != image_path:
                return False, "Skipped (Exists)"

            with Image.open(image_path) as img:
                width, height = img.size
                if max(width, height) <= max_dimension:
                    if output_path and output_path != image_path:
                        img.save(target, quality=95)
                        return True, "Copied (No Resize)"
                    return False, "Skipped (Small Enough)"

                if width > height:
                    new_width = max_dimension
                    new_height = int(height * (max_dimension / width))
                else:
                    new_height = max_dimension
                    new_width = int(width * (max_dimension / height))

                img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
                img.save(target, quality=95)
                return True, "Resized"
        except Exception as e:
            logging.error(f"Failed to resize {image_path}: {e}")
            return False, str(e)
