"""
Augment Tool

Expand a small image dataset by generating randomized augmentations.
Supports crop jitter, rotation, flip, color jitter, blur/sharpen, and noise.

CREDITS:
    Original script by a-l-e-x-d-s-9
    https://github.com/a-l-e-x-d-s-9/stable_diffusion_tools/blob/main/images_dataset_expand.py
    
    Adapted for A Thousand Words by integration into the tool system.
"""

import math
import random
import time
import gradio as gr
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional, Tuple

from PIL import Image, ImageEnhance, ImageFilter, ImageOps

# Colorama for console colors
try:
    from colorama import Fore, Style
except ImportError:
    class Fore:
        CYAN = YELLOW = GREEN = RED = ''
    class Style:
        RESET_ALL = ''

from .base import BaseTool, ToolConfig


@dataclass
class AugConfig:
    """Configuration for augmentation parameters."""
    # Crop jitter
    crop_jitter_prob: float
    crop_min_scale: float
    crop_max_scale: float
    max_translate_frac: float

    # Rotation
    rotate_prob: float
    max_rotate_deg: float

    # Flip
    hflip_prob: float

    # Color jitter
    color_jitter_prob: float
    brightness_range: float
    contrast_range: float
    saturation_range: float

    # Blur / sharpen
    blur_prob: float
    blur_max_radius: float
    sharpen_prob: float
    sharpen_max_percent: int

    # Noise
    noise_prob: float
    noise_std_range: float

    # Output
    out_format: str
    jpg_quality: int
    png_compress_level: int
    webp_quality: int

    # Captions
    copy_captions: bool
    
    # Seed
    seed: int


class AugmentTool(BaseTool):
    """
    Tool for expanding datasets through image augmentation.
    
    Credits: Original script by a-l-e-x-d-s-9
    https://github.com/a-l-e-x-d-s-9/stable_diffusion_tools
    """
    
    @property
    def config(self) -> ToolConfig:
        return ToolConfig(
            name="augment",
            display_name="Augment",
            description="### Augment Dataset\nGenerate augmented copies of images to expand your training dataset. Supports crop jitter, rotation, flip, color adjustments, blur/sharpen, and noise. Credits: [a-l-e-x-d-s-9/stable_diffusion_tools](https://github.com/a-l-e-x-d-s-9/stable_diffusion_tools)",
            icon=""
        )
    
    def _get_defaults(self) -> dict:
        """Return default values for all settings."""
        return {
            "target_count": 100,
            "output_dir": "",
            "prefix": "",
            "suffix": "",
            "out_format": "same",
            "jpg_quality": 100,
            "png_compress": 6,
            "webp_quality": 100,
            "copy_captions": True,
            "overwrite": True,
            "seed": 0,
            "crop_prob": 0.95,
            "crop_min": 0.78,
            "crop_max": 1.0,
            "translate": 0.06,
            "rotate_prob": 0.50,
            "rotate_max": 10.0,
            "flip_prob": 0.50,
            "color_prob": 0.55,
            "brightness": 0.12,
            "contrast": 0.12,
            "saturation": 0.10,
            "blur_prob": 0.12,
            "blur_radius": 0.9,
            "sharpen_prob": 0.18,
            "sharpen_pct": 140,
            "noise_prob": 0.12,
            "noise_std": 0.02,
            "force_width": 0,
            "force_height": 0,
            "min_width": 0,
            "min_height": 0,
            "max_width": 0,
            "max_height": 0,
        }
    
    def get_loaded_values(self, app) -> list:
        """Load saved settings from user config and return values for GUI components."""
        import gradio as gr
        
        defaults = self._get_defaults()
        saved = {}
        
        try:
            tool_settings = app.config_mgr.user_config.get("tool_settings", {})
            saved = tool_settings.get("augment", {})
        except Exception:
            pass
        
        # Merge saved over defaults
        values = {**defaults, **saved}
        
        # Return gr.update for each input in order matching create_gui inputs list
        return [
            gr.update(value=values["target_count"]),
            gr.update(value=values["output_dir"]),
            gr.update(value=values["prefix"]),
            gr.update(value=values["suffix"]),
            gr.update(value=values["out_format"]),
            gr.update(value=values["jpg_quality"]),
            gr.update(value=values["png_compress"]),
            gr.update(value=values["webp_quality"]),
            gr.update(value=values["copy_captions"]),
            gr.update(value=values["overwrite"]),
            gr.update(value=values["seed"]),
            gr.update(value=values["crop_prob"]),
            gr.update(value=values["crop_min"]),
            gr.update(value=values["crop_max"]),
            gr.update(value=values["translate"]),
            gr.update(value=values["rotate_prob"]),
            gr.update(value=values["rotate_max"]),
            gr.update(value=values["flip_prob"]),
            gr.update(value=values["color_prob"]),
            gr.update(value=values["brightness"]),
            gr.update(value=values["contrast"]),
            gr.update(value=values["saturation"]),
            gr.update(value=values["blur_prob"]),
            gr.update(value=values["blur_radius"]),
            gr.update(value=values["sharpen_prob"]),
            gr.update(value=values["sharpen_pct"]),
            gr.update(value=values["noise_prob"]),
            gr.update(value=values["noise_std"]),
            gr.update(value=values["force_width"]),
            gr.update(value=values["force_height"]),
            gr.update(value=values["min_width"]),
            gr.update(value=values["min_height"]),
            gr.update(value=values["max_width"]),
            gr.update(value=values["max_height"]),
        ]
    
    def apply_to_dataset(self, dataset, target_count: int, output_dir: str,
                         prefix: str, suffix: str,
                         out_format: str, jpg_quality: int, png_compress: int, webp_quality: int,
                         copy_captions: bool, overwrite: bool, seed: int,
                         crop_prob: float, crop_min: float, crop_max: float, translate: float,
                         rotate_prob: float, rotate_max: float,
                         flip_prob: float,
                         color_prob: float, brightness: float, contrast: float, saturation: float,
                         blur_prob: float, blur_radius: float,
                         sharpen_prob: float, sharpen_pct: int,
                         noise_prob: float, noise_std: float,
                         force_width: int, force_height: int,
                         min_width: int, min_height: int, max_width: int, max_height: int,
                         app_input_path: str = None) -> Tuple[str, List[str]]:
        """
        Expand dataset by generating augmented copies.
        
        Returns:
            Tuple[str, List[str]]: (status message, list of generated file paths)
        """
        # Use colorama for console colors
        
        if not dataset or not dataset.images:
            gr.Warning("No images loaded.")
            return "Error: No images in dataset.", []
        
        if target_count <= 0:
            gr.Warning("Target count must be greater than 0.")
            return "Error: Invalid target count.", []
        
        # Determine output directory with fallback logic
        if not output_dir or not output_dir.strip():
            if app_input_path and Path(app_input_path).exists() and Path(app_input_path).is_dir():
                output_dir = app_input_path
                print(f"{Fore.YELLOW}No output directory specified, using input path: {output_dir}{Style.RESET_ALL}")
            else:
                output_dir = "output"
                print(f"{Fore.YELLOW}No output directory specified, using fallback: {output_dir}{Style.RESET_ALL}")
        
        # Handle "same" format option
        same_format = (out_format == "same")
        
        # Build config
        cfg = AugConfig(
            crop_jitter_prob=crop_prob,
            crop_min_scale=crop_min,
            crop_max_scale=crop_max,
            max_translate_frac=translate,
            rotate_prob=rotate_prob,
            max_rotate_deg=rotate_max,
            hflip_prob=flip_prob,
            color_jitter_prob=color_prob,
            brightness_range=brightness,
            contrast_range=contrast,
            saturation_range=saturation,
            blur_prob=blur_prob,
            blur_max_radius=blur_radius,
            sharpen_prob=sharpen_prob,
            sharpen_max_percent=sharpen_pct,
            noise_prob=noise_prob,
            noise_std_range=noise_std,
            out_format=out_format,
            jpg_quality=jpg_quality,
            png_compress_level=png_compress,
            webp_quality=webp_quality,
            copy_captions=copy_captions,
            seed=seed
        )
        
        # Log settings
        print(f"{Fore.CYAN}Settings:{Style.RESET_ALL}")
        print(f"  Target Count: {target_count}")
        print(f"  Output Dir: {output_dir}")
        print(f"  Prefix: '{prefix}' | Suffix: '{suffix}'")
        print(f"  Format: {out_format} | Overwrite: {overwrite}")
        print(f"  Copy Captions: {copy_captions} | Seed: {seed}")
        print(f"  Crop: {crop_prob:.0%} (scale {crop_min:.2f}-{crop_max:.2f})")
        print(f"  Rotate: {rotate_prob:.0%} (max {rotate_max} deg)")
        print(f"  Flip: {flip_prob:.0%} | Color: {color_prob:.0%}")
        print(f"  Blur: {blur_prob:.0%} | Sharpen: {sharpen_prob:.0%} | Noise: {noise_prob:.0%}")
        if force_width and force_height:
            print(f"  Force Size: {force_width}x{force_height}")
        if any([min_width, min_height, max_width, max_height]):
            print(f"  Constraints: min({min_width or '-'}, {min_height or '-'}) max({max_width or '-'}, {max_height or '-'})")
        print(f"")
        
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        
        input_files = [img.path for img in dataset.images]
        if not input_files:
            return "Error: No valid input images.", []
        
        rng = random.Random(seed)
        np_rng = np.random.default_rng(seed)
        
        generated = 0
        errors = 0
        counter = 0
        t0 = time.time()
        generated_files = []
        
        print(f"{Fore.CYAN}Input images: {len(input_files)}{Style.RESET_ALL}")
        print(f"Generating augmented images...")
        print(f"")
        
        while generated < target_count:
            src = input_files[rng.randrange(0, len(input_files))]
            
            try:
                img = self._safe_open_image(src)
                img = self._to_rgb(img)
                
                base_w, base_h = img.size
                if force_width and force_height:
                    base_w, base_h = int(force_width), int(force_height)
                
                # Apply augmentations
                if rng.random() < cfg.crop_jitter_prob:
                    img = self._random_crop_jitter(img, rng, cfg)
                    img = self._resize_exact(img, base_w, base_h)
                elif force_width and force_height:
                    img = self._resize_exact(img, base_w, base_h)
                
                if rng.random() < cfg.rotate_prob and cfg.max_rotate_deg > 0:
                    img = self._apply_rotation(img, rng, cfg, base_w, base_h)
                
                if rng.random() < cfg.hflip_prob:
                    img = ImageOps.mirror(img)
                
                if rng.random() < cfg.color_jitter_prob:
                    img = self._apply_color_jitter(img, rng, cfg)
                
                if cfg.blur_max_radius > 0 and rng.random() < cfg.blur_prob:
                    radius = self._rand_uniform(rng, 0.1, cfg.blur_max_radius)
                    img = img.filter(ImageFilter.GaussianBlur(radius=radius))
                
                if cfg.sharpen_max_percent > 0 and rng.random() < cfg.sharpen_prob:
                    percent = int(round(self._rand_uniform(rng, 10.0, float(cfg.sharpen_max_percent))))
                    img = img.filter(ImageFilter.UnsharpMask(radius=1.0, percent=percent, threshold=3))
                
                if cfg.noise_std_range > 0 and rng.random() < cfg.noise_prob:
                    img = self._apply_noise_numpy(img, np_rng, cfg)
                
                # Apply size constraints if not forcing exact size
                if not (force_width and force_height):
                    img = self._apply_constraints_keep_aspect(
                        img, min_width or None, min_height or None,
                        max_width or None, max_height or None
                    )
                
                counter += 1
                
                # Determine output extension
                if same_format:
                    out_ext = src.suffix.lower()
                    if out_ext not in (".jpg", ".jpeg", ".png", ".webp"):
                        out_ext = ".jpg"
                else:
                    out_ext = "." + out_format.lower().replace("jpeg", "jpg")
                
                out_name = f"{prefix}{src.stem}_aug{counter:04d}_{rng.getrandbits(24):06x}{suffix}{out_ext}"
                out_file = out_path / out_name
                
                if out_file.exists() and not overwrite:
                    continue
                
                self._save_image(img, out_file, cfg, out_ext)
                generated_files.append(str(out_file.absolute()))
                
                if cfg.copy_captions:
                    self._maybe_copy_caption(src, out_file)
                
                generated += 1
                
                if generated % 50 == 0 or generated == target_count:
                    elapsed = time.time() - t0
                    rate = generated / elapsed if elapsed > 0 else 0.0
                    print(f"  {Fore.GREEN}Progress:{Style.RESET_ALL} {generated}/{target_count} ({rate:.2f} img/s)")
                    
            except Exception as e:
                errors += 1
                if errors <= 5:
                    print(f"  {Fore.RED}Error:{Style.RESET_ALL} {src.name}: {e}")
                if errors >= 50:
                    print(f"{Fore.RED}Too many errors ({errors}). Aborting.{Style.RESET_ALL}")
                    break
        
        elapsed = time.time() - t0
        print(f"")
        print(f"{Fore.CYAN}Summary:{Style.RESET_ALL}")
        print(f"  {Fore.GREEN}Generated: {generated}{Style.RESET_ALL}")
        print(f"  {Fore.RED if errors else Fore.CYAN}Errors: {errors}{Style.RESET_ALL}")
        if elapsed > 0:
            print(f"  Time: {elapsed:.2f}s ({generated / elapsed:.2f} img/s)")
        
        return f"Generated {generated} augmented images to {output_dir}", generated_files
    
    def create_gui(self, app, is_server_mode=False) -> tuple:
        """Create the Augment tool UI."""
        self._is_server_mode = is_server_mode
        self._app = app
        
        gr.Markdown(self.config.description)
        
        with gr.Accordion("Augmentation Settings", open=True):
            # Output Settings
            gr.Markdown("**Output Settings**")
            with gr.Row():
                target_count = gr.Number(label="Target Count", value=100, precision=0, minimum=1,
                                         info="Number of augmented images to generate")
                output_dir = gr.Textbox(label="Output Directory", placeholder="output",
                                        info="Leave empty to use input path or 'output'",
                                        visible=not is_server_mode)
                prefix = gr.Textbox(label="Prefix", placeholder="aug_", info="Added before filename")
                suffix = gr.Textbox(label="Suffix", placeholder="_v1", info="Added after filename")
            
            with gr.Row():
                out_format = gr.Dropdown(["same", "jpg", "png", "webp"], value="same", label="Format",
                                         info="Output format (same = keep original)")
                jpg_quality = gr.Number(label="JPG Quality", value=100, precision=0, minimum=1, maximum=100,
                                        info="JPEG compression quality")
                png_compress = gr.Number(label="PNG Compress", value=6, precision=0, minimum=0, maximum=9,
                                         info="PNG compression level")
                webp_quality = gr.Number(label="WebP Quality", value=100, precision=0, minimum=1, maximum=100,
                                         info="WebP compression quality")
            
            with gr.Row():
                copy_captions = gr.Checkbox(label="Copy Captions", value=True,
                                            info="Copy matching .txt caption files for augmented images")
                overwrite = gr.Checkbox(label="Overwrite", value=True, 
                                        info="Overwrite existing files")
                seed = gr.Number(label="Seed", value=0, precision=0, 
                                info="RNG seed (0 is valid)")
            
            # Size Settings
            gr.Markdown("**Size Settings**")
            with gr.Row():
                force_width = gr.Number(label="Force Width", value=0, precision=0,
                                        info="Exact output width (0 = preserve)")
                force_height = gr.Number(label="Force Height", value=0, precision=0,
                                         info="Exact output height (0 = preserve)")
                min_width = gr.Number(label="Min Width", value=0, precision=0,
                                      info="Minimum width constraint")
                min_height = gr.Number(label="Min Height", value=0, precision=0,
                                       info="Minimum height constraint")
                max_width = gr.Number(label="Max Width", value=0, precision=0, 
                                      info="Maximum width (0 = no limit)")
                max_height = gr.Number(label="Max Height", value=0, precision=0, 
                                       info="Maximum height (0 = no limit)")
            
            # Crop & Transform
            gr.Markdown("**Crop & Transform**")
            with gr.Row():
                crop_prob = gr.Slider(0, 1, value=0.95, step=0.05, label="Crop Prob",
                                      info="Probability of crop jitter")
                crop_min = gr.Slider(0.5, 1, value=0.78, step=0.01, label="Crop Min Scale",
                                     info="Minimum crop scale")
                crop_max = gr.Slider(0.5, 1, value=1.0, step=0.01, label="Crop Max Scale",
                                     info="Maximum crop scale")
                translate = gr.Slider(0, 0.2, value=0.06, step=0.01, label="Translate Frac",
                                      info="Max center shift fraction")
            
            with gr.Row():
                rotate_prob = gr.Slider(0, 1, value=0.50, step=0.05, label="Rotate Prob",
                                        info="Probability of rotation")
                rotate_max = gr.Slider(0, 45, value=10.0, step=0.5, label="Rotate Max Deg",
                                       info="Maximum rotation degrees")
                flip_prob = gr.Slider(0, 1, value=0.50, step=0.05, label="H-Flip Prob",
                                      info="Probability of horizontal flip")
            
            # Color Adjustments
            gr.Markdown("**Color Adjustments**")
            with gr.Row():
                color_prob = gr.Slider(0, 1, value=0.55, step=0.05, label="Color Prob",
                                       info="Probability of color jitter")
                brightness = gr.Slider(0, 0.5, value=0.12, step=0.01, label="Brightness Range",
                                       info="Brightness adjustment range")
                contrast = gr.Slider(0, 0.5, value=0.12, step=0.01, label="Contrast Range",
                                     info="Contrast adjustment range")
                saturation = gr.Slider(0, 0.5, value=0.10, step=0.01, label="Saturation Range",
                                       info="Saturation adjustment range")
            
            # Blur, Sharpen, Noise
            gr.Markdown("**Blur, Sharpen, Noise**")
            with gr.Row():
                blur_prob = gr.Slider(0, 1, value=0.12, step=0.05, label="Blur Prob",
                                      info="Probability of blur")
                blur_radius = gr.Slider(0, 3, value=0.9, step=0.1, label="Blur Max Radius",
                                        info="Maximum blur radius")
                sharpen_prob = gr.Slider(0, 1, value=0.18, step=0.05, label="Sharpen Prob",
                                         info="Probability of sharpen")
                sharpen_pct = gr.Number(label="Sharpen Max %", value=140, precision=0, minimum=0, maximum=500,
                                        info="Maximum sharpen percentage")
            
            with gr.Row():
                noise_prob = gr.Slider(0, 1, value=0.12, step=0.05, label="Noise Prob",
                                       info="Probability of noise")
                noise_std = gr.Slider(0, 0.1, value=0.02, step=0.005, label="Noise Std Range",
                                      info="Noise standard deviation (fraction of 255)")
        
        # Control buttons
        with gr.Row():
            save_btn = gr.Button("Save Settings", variant="secondary", scale=0)
            run_btn = gr.Button("Augment Dataset", variant="primary", scale=1, elem_id="augment_tool_btn")
            with gr.Column(visible=False, scale=0, min_width=80, elem_classes="download-btn-wrapper") as download_btn_group:
                download_btn = gr.DownloadButton(
                    label="", 
                    icon=str(Path(__file__).parent.parent / "core" / "download_white.svg"),
                    visible=True, variant="primary", scale=0, elem_classes="download-btn"
                )
        
        # Store for wire_events
        self._app = app
        self._download_btn = download_btn
        self._download_btn_group = download_btn_group
        self._save_btn = save_btn
        
        inputs = [
            target_count, output_dir,
            prefix, suffix,
            out_format, jpg_quality, png_compress, webp_quality,
            copy_captions, overwrite, seed,
            crop_prob, crop_min, crop_max, translate,
            rotate_prob, rotate_max,
            flip_prob,
            color_prob, brightness, contrast, saturation,
            blur_prob, blur_radius,
            sharpen_prob, sharpen_pct,
            noise_prob, noise_std,
            force_width, force_height,
            min_width, min_height, max_width, max_height
        ]
        
        return (run_btn, inputs)
    
    def wire_events(self, app, run_button, inputs: list, gallery_output: gr.Gallery,
                    limit_count=None) -> None:
        """Custom wire_events with button state management and download button."""
        import copy as copy_module
        from src.gui.inference import tool_start_processing, tool_finish_processing
        import tempfile
        import os
        
        tool_name = self.config.display_name
        
        download_btn = self._download_btn
        download_btn_group = self._download_btn_group
        save_btn = self._save_btn
        
        all_inputs = inputs.copy() if inputs else []
        if limit_count is not None:
            all_inputs.append(limit_count)
        
        def run_handler(*args):
            if not app.dataset or not app.dataset.images:
                gr.Warning("No images loaded. Please load a folder in Input Source first.")
                return (
                    gr.update(value="Augment Dataset", interactive=True),
                    gr.update(visible=False),
                    gr.update()
                )
            
            limit_val = None
            tool_args = list(args)
            if limit_count is not None and len(args) > len(inputs):
                limit_val = args[-1]
                tool_args = list(args[:-1])
            
            run_dataset = app.dataset
            total_count = len(app.dataset.images)
            
            if limit_val:
                try:
                    limit = int(limit_val)
                    if limit > 0 and total_count > limit:
                        run_dataset = copy_module.copy(app.dataset)
                        run_dataset.images = app.dataset.images[:limit]
                        print(f"{Fore.YELLOW}Limiting to first {limit} images (from {total_count} loaded).{Style.RESET_ALL}")
                except (ValueError, TypeError):
                    pass
            
            # Server Mode Handling
            temp_dir_obj = None
            if self._is_server_mode:
                temp_dir_obj = tempfile.TemporaryDirectory(dir=os.environ.get("GRADIO_TEMP_DIR"), prefix="augment_")
                tool_args[1] = temp_dir_obj.name # Override output_dir
                print(f"{Fore.CYAN}Server Mode: Generating to temp dir {tool_args[1]}{Style.RESET_ALL}")

            image_count = len(run_dataset.images)
            print(f"")
            print(f"{Fore.CYAN}--- Running {tool_name} Tool on {image_count} images ---{Style.RESET_ALL}")
            
            # Pass app_input_path as the last argument for fallback logic
            app_input_path = app.current_input_path if not app.is_drag_and_drop else None
            tool_args.append(app_input_path)
            
            try:
                result, generated_files = self.apply_to_dataset(run_dataset, *tool_args)
                
                print(f"{Fore.GREEN}Result: {result}{Style.RESET_ALL}")
                print(f"{Fore.CYAN}--- {tool_name} Tool Complete ---{Style.RESET_ALL}")
                print(f"")
                
                gr.Info(result)
                
                # Cleanup if we used a temp dir (tool_finish_processing creates zip from files, effectively copying them out of here)
                # Wait, tool_finish_processing creates zip from paths. 
                # If we cleanup temp_dir_obj immediately, files disappear.
                # But tool_finish_processing runs synchronously here before returning.
                # Actually, tool_finish_processing just returns updates. 
                # It calls file_loader.create_zip internally.
                # So we can cleanup afterwards.
                
                ui_updates = tool_finish_processing("Augment Dataset", generated_files, zip_prefix="augmented_images")
                
                if temp_dir_obj:
                    # We should delay cleanup or ignore it?
                    # The zip is created in tool_finish_processing.
                    # So we can clean up now.
                    # But checking implementation of tool_finish_processing:
                    # it calls file_loader.create_zip(generated_files).
                    # So zip is created NOW.
                    temp_dir_obj.cleanup()
                    
                return ui_updates
            
            except Exception as e:
                # Cleanup on error
                if temp_dir_obj:
                    temp_dir_obj.cleanup()
                raise e
        
        def save_settings(*args):
            """Save tool settings to user config using existing pattern."""
            from src.gui.constants import filter_user_overrides
            
            settings = {
                "target_count": args[0],
                "output_dir": args[1],
                "prefix": args[2],
                "suffix": args[3],
                "out_format": args[4],
                "jpg_quality": args[5],
                "png_compress": args[6],
                "webp_quality": args[7],
                "copy_captions": args[8],
                "overwrite": args[9],
                "seed": args[10],
                "crop_prob": args[11],
                "crop_min": args[12],
                "crop_max": args[13],
                "translate": args[14],
                "rotate_prob": args[15],
                "rotate_max": args[16],
                "flip_prob": args[17],
                "color_prob": args[18],
                "brightness": args[19],
                "contrast": args[20],
                "saturation": args[21],
                "blur_prob": args[22],
                "blur_radius": args[23],
                "sharpen_prob": args[24],
                "sharpen_pct": args[25],
                "noise_prob": args[26],
                "noise_std": args[27],
            }
            
            try:
                # Initialize tool_settings if needed
                if "tool_settings" not in app.config_mgr.user_config:
                    app.config_mgr.user_config["tool_settings"] = {}
                if "augment" not in app.config_mgr.user_config["tool_settings"]:
                    app.config_mgr.user_config["tool_settings"]["augment"] = {}
                
                # Get defaults for diff comparison
                defaults = self._get_defaults()
                
                # Only save values that differ from defaults
                for key, value in settings.items():
                    default_val = defaults.get(key)
                    if value != default_val:
                        app.config_mgr.user_config["tool_settings"]["augment"][key] = value
                    elif key in app.config_mgr.user_config["tool_settings"]["augment"]:
                        # Remove if it matches default
                        del app.config_mgr.user_config["tool_settings"]["augment"][key]
                
                # Clean up empty dict
                if not app.config_mgr.user_config["tool_settings"]["augment"]:
                    del app.config_mgr.user_config["tool_settings"]["augment"]
                if not app.config_mgr.user_config["tool_settings"]:
                    del app.config_mgr.user_config["tool_settings"]
                
                # Use filter_user_overrides to clean up before saving
                filtered = filter_user_overrides(app.config_mgr.user_config)
                app.config_mgr._save_yaml(app.config_mgr.user_config_path, filtered)
                gr.Info("Augment settings saved!")
            except Exception as e:
                gr.Warning(f"Failed to save settings: {e}")
        
        # Wire save button
        save_btn.click(
            save_settings,
            inputs=inputs[:28],
            outputs=[]
        )
        
        # Wire run button with state management
        run_button.click(
            tool_start_processing,
            inputs=[],
            outputs=[run_button, download_btn_group, download_btn]
        ).then(
            run_handler,
            inputs=all_inputs,
            outputs=[run_button, download_btn_group, download_btn],
            show_progress="hidden"
        )
    
    # --- Helper Methods (adapted from original script) ---
    
    def _safe_open_image(self, path: Path) -> Image.Image:
        """Open image with EXIF transpose."""
        img = Image.open(path)
        img = ImageOps.exif_transpose(img)
        return img
    
    def _to_rgb(self, img: Image.Image) -> Image.Image:
        """Convert to RGB, handling alpha by compositing on white."""
        if img.mode == "RGB":
            return img
        if img.mode in ("RGBA", "LA") or ("transparency" in img.info):
            rgba = img.convert("RGBA")
            bg = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
            comp = Image.alpha_composite(bg, rgba).convert("RGB")
            return comp
        return img.convert("RGB")
    
    def _rand_uniform(self, rng: random.Random, lo: float, hi: float) -> float:
        """Random float between lo and hi."""
        return lo + (hi - lo) * rng.random()
    
    def _resize_exact(self, img: Image.Image, w: int, h: int) -> Image.Image:
        """Resize to exact dimensions."""
        return img.resize((w, h), resample=Image.LANCZOS)
    
    def _random_crop_jitter(self, img: Image.Image, rng: random.Random, cfg: AugConfig) -> Image.Image:
        """Apply random crop jitter."""
        w, h = img.size
        if w < 2 or h < 2:
            return img
        
        s = self._rand_uniform(rng, cfg.crop_min_scale, cfg.crop_max_scale)
        s = max(0.10, min(1.0, s))
        
        crop_w = max(2, int(round(w * s)))
        crop_h = max(2, int(round(h * s)))
        
        max_shift = cfg.max_translate_frac * float(min(w, h))
        dx = self._rand_uniform(rng, -max_shift, max_shift)
        dy = self._rand_uniform(rng, -max_shift, max_shift)
        
        cx = (w / 2.0) + dx
        cy = (h / 2.0) + dy
        
        left = int(round(cx - crop_w / 2.0))
        top = int(round(cy - crop_h / 2.0))
        right = left + crop_w
        bottom = top + crop_h
        
        # Clamp to bounds
        if left < 0:
            right -= left
            left = 0
        if top < 0:
            bottom -= top
            top = 0
        if right > w:
            left -= (right - w)
            right = w
        if bottom > h:
            top -= (bottom - h)
            bottom = h
        
        left = max(0, left)
        top = max(0, top)
        right = min(w, right)
        bottom = min(h, bottom)
        
        if right - left < 2 or bottom - top < 2:
            return img
        
        return img.crop((left, top, right, bottom))
    
    def _apply_rotation(self, img: Image.Image, rng: random.Random, 
                        cfg: AugConfig, base_w: int, base_h: int) -> Image.Image:
        """Apply rotation with border crop and resize back."""
        deg = self._rand_uniform(rng, -cfg.max_rotate_deg, cfg.max_rotate_deg)
        angle_rad = math.radians(deg)
        
        pre_w, pre_h = img.size
        rotated = img.rotate(deg, resample=Image.BICUBIC, expand=True, fillcolor=(255, 255, 255))
        
        crop_w, crop_h = self._rotated_rect_with_max_area(pre_w, pre_h, angle_rad)
        crop_w = int(round(crop_w))
        crop_h = int(round(crop_h))
        
        if crop_w >= 2 and crop_h >= 2:
            rotated = self._center_crop(rotated, crop_w, crop_h)
        
        return self._resize_exact(rotated, base_w, base_h)
    
    def _rotated_rect_with_max_area(self, w: int, h: int, angle_rad: float) -> Tuple[float, float]:
        """Compute largest axis-aligned rectangle in rotated rectangle."""
        if w <= 0 or h <= 0:
            return 0.0, 0.0
        
        angle = abs(angle_rad) % (math.pi / 2.0)
        if angle < 1e-12:
            return float(w), float(h)
        
        sin_a = abs(math.sin(angle))
        cos_a = abs(math.cos(angle))
        
        width_is_longer = w >= h
        side_long = float(w if width_is_longer else h)
        side_short = float(h if width_is_longer else w)
        
        if side_short <= 2.0 * sin_a * cos_a * side_long or abs(sin_a - cos_a) < 1e-12:
            x = 0.5 * side_short
            if width_is_longer:
                crop_w = x / sin_a
                crop_h = x / cos_a
            else:
                crop_w = x / cos_a
                crop_h = x / sin_a
        else:
            cos_2a = (cos_a * cos_a) - (sin_a * sin_a)
            crop_w = (w * cos_a - h * sin_a) / cos_2a
            crop_h = (h * cos_a - w * sin_a) / cos_2a
        
        return abs(float(crop_w)), abs(float(crop_h))
    
    def _center_crop(self, img: Image.Image, crop_w: int, crop_h: int) -> Image.Image:
        """Center crop to specified dimensions."""
        w, h = img.size
        crop_w = max(1, min(w, crop_w))
        crop_h = max(1, min(h, crop_h))
        left = int(round((w - crop_w) / 2.0))
        top = int(round((h - crop_h) / 2.0))
        return img.crop((left, top, left + crop_w, top + crop_h))
    
    def _apply_color_jitter(self, img: Image.Image, rng: random.Random, cfg: AugConfig) -> Image.Image:
        """Apply brightness, contrast, saturation jitter."""
        b = self._rand_uniform(rng, 1.0 - cfg.brightness_range, 1.0 + cfg.brightness_range)
        c = self._rand_uniform(rng, 1.0 - cfg.contrast_range, 1.0 + cfg.contrast_range)
        s = self._rand_uniform(rng, 1.0 - cfg.saturation_range, 1.0 + cfg.saturation_range)
        
        img = ImageEnhance.Brightness(img).enhance(b)
        img = ImageEnhance.Contrast(img).enhance(c)
        img = ImageEnhance.Color(img).enhance(s)
        return img
    
    def _apply_noise_numpy(self, img: Image.Image, np_rng: np.random.Generator, cfg: AugConfig) -> Image.Image:
        """Apply Gaussian noise using NumPy (much faster than pixel-by-pixel)."""
        std = np_rng.uniform(0.0, cfg.noise_std_range) * 255.0
        if std < 0.01:
            return img
        
        arr = np.array(img, dtype=np.float32)
        noise = np_rng.normal(0.0, std, arr.shape).astype(np.float32)
        arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(arr, mode="RGB")
    
    def _apply_constraints_keep_aspect(self, img: Image.Image,
                                        min_w: Optional[int], min_h: Optional[int],
                                        max_w: Optional[int], max_h: Optional[int]) -> Image.Image:
        """Apply min/max size constraints, preserving aspect ratio."""
        if min_w is None and min_h is None and max_w is None and max_h is None:
            return img
        
        w, h = img.size
        
        # Calculate scale factors
        s_min = 1.0
        if min_w is not None and min_w > 0:
            s_min = max(s_min, min_w / float(w))
        if min_h is not None and min_h > 0:
            s_min = max(s_min, min_h / float(h))
        
        s_max = float("inf")
        if max_w is not None and max_w > 0:
            s_max = min(s_max, max_w / float(w))
        if max_h is not None and max_h > 0:
            s_max = min(s_max, max_h / float(h))
        
        if s_max == float("inf"):
            s_max = s_min
        
        # Pick final scale
        if s_min > s_max:
            s_final = (s_min + s_max) / 2.0
        elif s_min <= 1.0 <= s_max:
            s_final = 1.0
        else:
            s_final = s_min if s_min > 1.0 else s_max
        
        if abs(s_final - 1.0) < 1e-9:
            return img
        
        new_w = max(1, int(round(w * s_final)))
        new_h = max(1, int(round(h * s_final)))
        return img.resize((new_w, new_h), resample=Image.LANCZOS)
    
    def _save_image(self, img: Image.Image, out_path: Path, cfg: AugConfig, out_ext: str = None) -> None:
        """Save image in specified format."""
        out_path.parent.mkdir(parents=True, exist_ok=True)
        
        ext = out_ext.lower() if out_ext else "." + cfg.out_format.lower()
        ext = ext.lstrip(".")
        
        if ext in ("jpg", "jpeg"):
            img.save(out_path, format="JPEG", quality=cfg.jpg_quality, optimize=True, subsampling=0)
        elif ext == "png":
            img.save(out_path, format="PNG", compress_level=cfg.png_compress_level, optimize=True)
        elif ext == "webp":
            img.save(out_path, format="WEBP", quality=cfg.webp_quality, method=6)
        else:
            img.save(out_path, format="JPEG", quality=cfg.jpg_quality)
    
    def _maybe_copy_caption(self, src_path: Path, out_img_path: Path) -> None:
        """Copy caption sidecar file if it exists."""
        src_txt = src_path.with_suffix(".txt")
        if not src_txt.exists() or not src_txt.is_file():
            return
        out_txt = out_img_path.with_suffix(".txt")
        try:
            out_txt.write_text(src_txt.read_text(encoding="utf-8", errors="replace"), encoding="utf-8")
        except Exception:
            pass
