"""
Inference helpers for argument building and validation.
"""

from pathlib import Path
import gradio as gr


def build_inference_args(app, model_id, model_ver, batch, tokens, dynamic_settings, 
                         pre, suf, over, rec, con, unload, clean, collapse, norm, rm_cn, loop, 
                         w, h, limit, out_dir_glob, out_fmt):
    """
    Construct arguments dictionary for runs and CLI generation.
    """
    # Get defaults
    model_config = app.config_mgr.get_model_config(model_id)
    args_dict = model_config.get('defaults', {}).copy()
    
    # Update with dynamic settings
    if dynamic_settings:
        clean_dynamic = {k: v for k, v in dynamic_settings.items() if v is not None}
        args_dict.update(clean_dynamic)
    
    # Static overrides
    if model_ver: 
        args_dict['model_version'] = model_ver
    
    if batch: args_dict['batch_size'] = int(batch)
    if tokens: args_dict['max_tokens'] = int(tokens)
    
    # Text Processing
    if pre: args_dict['prefix'] = pre
    if suf: args_dict['suffix'] = suf
    if clean is not None: args_dict['clean_text'] = clean
    if collapse is not None: args_dict['collapse_newlines'] = collapse
    if norm is not None: args_dict['normalize_text'] = norm
    if rm_cn is not None: args_dict['remove_chinese'] = rm_cn
    if loop is not None: args_dict['strip_loop'] = loop
    
    # Image Processing
    args_dict['max_width'] = int(w) if w else None
    args_dict['max_height'] = int(h) if h else None
    
    # Execution Flags
    if over is not None: args_dict['overwrite'] = over
    if rec is not None: args_dict['recursive'] = rec
    if con is not None: args_dict['print_console'] = con
    if unload is not None: args_dict['unload_model'] = unload
    
    # Output Settings
    if out_dir_glob: args_dict['output_dir'] = out_dir_glob
    if out_fmt: args_dict['output_format'] = out_fmt
    if limit: args_dict['limit_count'] = limit

    # Global Settings
    g_set = app.config_mgr.get_global_settings()
    args_dict['gpu_vram'] = g_set.get('gpu_vram', 24)
    
    return args_dict


def validate_run_state(app, model_id) -> bool:
    """
    Validate run requirements.
    Results in Warning if invalid.
    """
    if not app.dataset or not app.dataset.images:
        gr.Warning("No media found. Please load a folder or add images to the 'Input Source'.")
        return False
    
    # Check media compatibility
    config = app.config_mgr.get_model_config(model_id)
    supported_media = config.get('media_type', ["Image"])
    if isinstance(supported_media, str):
        supported_media = [supported_media]

    has_images = any(img.media_type == "image" for img in app.dataset.images)
    has_videos = any(img.media_type == "video" for img in app.dataset.images)
    
    if not has_images and not has_videos:
        gr.Warning("Dataset contains no valid image or video files.")
        return False

    supports_image = "Image" in supported_media
    supports_video = "Video" in supported_media

    if supports_video and not supports_image and not has_videos and has_images:
        gr.Warning(f"Model '{model_id}' only supports Video, but dataset contains only Images.")
        return False
    
    return True


def validate_dataset_only(app) -> bool:
    """
    Validate dataset is loaded.
    """
    if not app.dataset or not app.dataset.images:
        gr.Warning("No media found. Please load a folder first.")
        return False
    return True


def start_processing(is_valid):
    """
    Update UI on run start:
    1. Run Button -> "Processing...", Disabled
    2. Download Button -> Show wrapper, set button to Processing state (spinner)
    """
    if not is_valid:
        return (
            gr.update(value="Run Captioning", interactive=True),
            gr.update(visible=False),
            gr.update()
        )

    return (
        gr.update(value="Processing...", interactive=False),
        gr.update(visible=True),
        gr.update(
            value=None,
            variant="secondary",
            interactive=False,
            elem_classes=["processing-btn"]
        )
    )


def run_with_dynamic_state(app, build_args_fn, *args):
    """
    Execute inference with dynamic state and return UI updates.
    
    Args:
        app: CaptioningApp instance
        build_args_fn: Function to build inference args
        *args: Input arguments (last one is validation state)
    """
    is_valid = args[-1]
    if not is_valid:
        return gr.update(), gr.update(value="Run Captioning", interactive=True), gr.update(visible=False), gr.update(visible=False)

    input_args = args[:-1]
    args_dict = build_args_fn(*input_args)
    
    result = app.run_inference(args_dict.get('model_id') or app.current_model_id, args_dict)
    
    stats = {}
    if isinstance(result, tuple) and len(result) == 3:
        gallery_data, download_update, stats = result
    elif isinstance(result, tuple) and len(result) == 2:
        gallery_data, download_update = result
    else:
        gallery_data = result
        download_update = gr.update(visible=False)
    
    if stats:
        processed = stats.get('processed', 0)
        skipped = stats.get('skipped', 0)
        time_val = stats.get('time', 0)
        empty = stats.get('empty_count', 0)
        peak_vram = stats.get('peak_vram', 0)
        vram_str = f"{peak_vram:.2f} GB" if peak_vram else "N/A"
        
        if processed > 0:
            model_name = stats.get('model_name', 'Unknown')
            msg = (
                f"Model: {model_name}<br>"
                f"Captioned: {processed} files<br>"
                f"Time: {time_val:.2f}s<br>"
                f"Peak VRAM: {vram_str}"
            )
            if skipped > 0:
                msg += f"<br>Skipped: {skipped}"
            gr.Info(msg, title="Success")
        elif skipped > 0:
            gr.Info(f"Skipped all {skipped} files (already exist).<br>Check 'Overwrite' to re-caption.", title="No Changes")
        else:
            gr.Warning("No files were processed. The dataset may be empty.")

        if empty > 0:
            gr.Warning(f"{empty} captions were empty!")
    else:
        gr.Warning("Processing completed but no statistics were returned.")
    
    # Download Button Logic
    dl_grp = gr.update(visible=False)
    dl_btn = gr.update(visible=False)
    if isinstance(download_update, dict) and download_update.get('value'):
        dl_grp = gr.update(visible=True)
        dl_btn = gr.update(
            value=download_update.get('value'),
            visible=True,
            interactive=True,
            variant="primary",
            icon=str(Path(__file__).parent.parent / "core" / "download_white.svg"),
            elem_classes="download-btn"
        )

    return gallery_data, gr.update(value="Run Captioning", interactive=True), dl_grp, dl_btn


# ============================================================================
# REUSABLE TOOL BUTTON STATE FUNCTIONS
# Use these in tools for consistent button behavior across the app.
# ============================================================================

def tool_start_processing(button_text: str = "Run Tool"):
    """
    Return UI updates for when a tool starts processing.
    
    Returns tuple for: (run_button, download_btn_group, download_btn)
    - Run button shows "Processing..." and is disabled
    - Download button wrapper is visible but button shows processing state
    
    Usage in wire_events:
        run_button.click(
            lambda: tool_start_processing("Augment Dataset"),
            inputs=[],
            outputs=[run_button, download_btn_group, download_btn]
        ).then(...)
    """
    return (
        gr.update(value="Processing...", interactive=False),
        gr.update(visible=True),
        gr.update(
            value=None,
            variant="secondary",
            interactive=False,
            elem_classes=["processing-btn"]
        )
    )


def tool_finish_processing(button_text: str, generated_files: list = None, zip_prefix: str = "tool_output"):
    """
    Return UI updates for when a tool finishes processing.
    
    Args:
        button_text: Text to restore on the run button (e.g., "Augment Dataset")
        generated_files: List of file paths. If provided and non-empty, creates zip and shows download.
        zip_prefix: Prefix for the generated zip filename (e.g., "augmented_images").
        
    Returns tuple for: (run_button, download_btn_group, download_btn)
    """
    from src.gui import file_loader
    
    # Default: hide download
    dl_grp = gr.update(visible=False)
    dl_btn = gr.update(visible=False)
    
    # If files were generated, create zip and show download button
    if generated_files:
        zip_path = file_loader.create_zip(generated_files, base_name=zip_prefix)
        if zip_path:
            dl_grp = gr.update(visible=True)
            dl_btn = gr.update(
                value=zip_path,
                visible=True,
                interactive=True,
                variant="primary",
                icon=str(Path(__file__).parent.parent / "core" / "download_white.svg"),
                elem_classes=["download-btn"]
            )
    
    return (
        gr.update(value=button_text, interactive=True),
        dl_grp,
        dl_btn
    )
