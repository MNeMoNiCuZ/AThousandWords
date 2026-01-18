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

# Colorama for console colors
try:
    from colorama import Fore, Style
except ImportError:
    class Fore:
        CYAN = YELLOW = GREEN = RED = ''
    class Style:
        RESET_ALL = DIM = ''


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
    
    def _get_defaults(self) -> dict:
        """Return default values for all settings."""
        return {
            "max_dim": 1024,
            "output_dir": "",
            "prefix": "",
            "suffix": "",
            "extension": "",
            "overwrite": True,
        }
    
    def get_loaded_values(self, app) -> list:
        """Load saved settings from user config."""
        import gradio as gr
        
        defaults = self._get_defaults()
        saved = {}
        
        try:
            tool_settings = app.config_mgr.user_config.get("tool_settings", {})
            saved = tool_settings.get("resize", {})
        except Exception:
            pass
        
        values = {**defaults, **saved}
        
        # Order must match create_gui inputs list
        return [
            gr.update(value=values["max_dim"]),
            gr.update(value=values["output_dir"]),
            gr.update(value=values["prefix"]),
            gr.update(value=values["suffix"]),
            gr.update(value=values["extension"]),
            gr.update(value=values["overwrite"]),
        ]
    
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
        count = 0
        skipped = 0
        errors = 0
        
        # Log settings
        print(f"{Fore.CYAN}Settings:{Style.RESET_ALL}")
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
                print(f"  {Fore.GREEN}[OK]{Style.RESET_ALL} {src_path.name} -> {new_name} ({msg})")
                logging.info(f"Resized: {src_path.name} -> {new_name} ({msg})")
            elif "Skipped" in msg:
                skipped += 1
                print(f"  {Fore.YELLOW}[SKIP]{Style.RESET_ALL} {src_path.name} ({msg})")
                logging.debug(f"Skipped: {src_path.name} ({msg})")
            else:
                errors += 1
                print(f"  {Fore.RED}[ERR]{Style.RESET_ALL} {src_path.name} ({msg})")
                logging.error(f"Error: {src_path.name} ({msg})")
        
        print(f"")
        print(f"{Fore.CYAN}Summary:{Style.RESET_ALL} {Fore.GREEN}Resized: {count}{Style.RESET_ALL} | {Fore.YELLOW}Skipped: {skipped}{Style.RESET_ALL} | {Fore.RED}Errors: {errors}{Style.RESET_ALL}")
                
        result = f"Processed {len(dataset)} images. Resized/Saved: {count}, Skipped: {skipped}, Errors: {errors}"
        return result
    
    def create_gui(self, app) -> tuple:
        """Create the Resize tool UI. Returns (run_button, inputs) for later event wiring."""
        
        gr.Markdown(self.config.description)
        
        with gr.Accordion("Resize Settings", open=True):
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
                resize_px = gr.Number(label="Max Dimension (px)", value=1024, precision=0, info="Maximum width or height")
                resize_overwrite = gr.Checkbox(label="Overwrite", value=True, info="Overwrite if file exists")
        
        with gr.Row():
            save_btn = gr.Button("Save Settings", variant="secondary", scale=0)
            resize_run = gr.Button("Resize Images", variant="primary", scale=1, elem_id="resize_tool_btn")
        
        # Store for wire_events
        self._save_btn = save_btn
        
        # Return components for later event wiring
        inputs = [resize_px, resize_out_dir, resize_pre, resize_suf, resize_ext, resize_overwrite]
        return (resize_run, inputs)
    
    def wire_events(self, app, run_button, inputs: list, gallery_output, limit_count=None) -> None:
        """Wire events with save settings support."""
        from src.gui.constants import filter_user_overrides
        
        save_btn = self._save_btn
        
        def save_settings(*args):
            settings = {
                "max_dim": args[0],
                "output_dir": args[1],
                "prefix": args[2],
                "suffix": args[3],
                "extension": args[4],
                "overwrite": args[5],
            }
            
            try:
                if "tool_settings" not in app.config_mgr.user_config:
                    app.config_mgr.user_config["tool_settings"] = {}
                if "resize" not in app.config_mgr.user_config["tool_settings"]:
                    app.config_mgr.user_config["tool_settings"]["resize"] = {}
                
                defaults = self._get_defaults()
                
                for key, value in settings.items():
                    default_val = defaults.get(key)
                    if value != default_val:
                        app.config_mgr.user_config["tool_settings"]["resize"][key] = value
                    elif key in app.config_mgr.user_config["tool_settings"]["resize"]:
                        del app.config_mgr.user_config["tool_settings"]["resize"][key]
                
                if not app.config_mgr.user_config["tool_settings"]["resize"]:
                    del app.config_mgr.user_config["tool_settings"]["resize"]
                if not app.config_mgr.user_config.get("tool_settings"):
                    if "tool_settings" in app.config_mgr.user_config:
                        del app.config_mgr.user_config["tool_settings"]
                
                filtered = filter_user_overrides(app.config_mgr.user_config)
                app.config_mgr._save_yaml(app.config_mgr.user_config_path, filtered)
                gr.Info("Resize settings saved!")
            except Exception as e:
                gr.Warning(f"Failed to save settings: {e}")
        
        save_btn.click(save_settings, inputs=inputs, outputs=[])
        
        # Call base class wire_events for run button
        super().wire_events(app, run_button, inputs, gallery_output, limit_count)
    
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
