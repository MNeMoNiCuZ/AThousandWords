"""
Inference helpers for argument building and validation.
"""

import gradio as gr


def build_inference_args(app, model_id, model_ver, batch, tokens, dynamic_settings, 
                         pre, suf, over, rec, con, unload, clean, collapse, norm, rm_cn, loop, 
                         w, h, limit, out_dir_glob, out_fmt):
    """
                         Builds an inference arguments dictionary by merging model defaults with dynamic and explicit overrides.
                         
                         Parameters:
                             model_id (str): Identifier of the model whose defaults are used.
                             model_ver (str | None): Explicit model version to set, if provided.
                             batch (int | None): Batch size override.
                             tokens (int | None): Maximum tokens override.
                             dynamic_settings (dict | None): Mapping of option names to values; entries with value None are ignored.
                             pre (str | None): Text prefix to set for generated captions.
                             suf (str | None): Text suffix to set for generated captions.
                             over (bool | None): Whether to overwrite existing output files.
                             rec (bool | None): Whether to recurse directories when scanning input.
                             con (bool | None): Whether to print progress to the console.
                             unload (bool | None): Whether to unload the model after inference.
                             clean (bool | None): Whether to enable text cleaning.
                             collapse (bool | None): Whether to collapse consecutive newlines in text.
                             norm (bool | None): Whether to normalize text.
                             rm_cn (bool | None): Whether to remove Chinese characters from text.
                             loop (bool | None): Whether to strip loop markers from filenames/text.
                             w (int | None): Maximum image width; if falsy, `max_width` is set to None.
                             h (int | None): Maximum image height; if falsy, `max_height` is set to None.
                             limit (int | None): Limit on processed items (sets `limit_count`).
                             out_dir_glob (str | None): Output directory path or glob to set.
                             out_fmt (str | None): Output format string to set.
                         
                         Returns:
                             dict: Assembled arguments dictionary ready for inference or CLI use, including merged defaults, overrides, image/text processing settings, execution flags, output settings, and a `gpu_vram` entry from global settings (default 24).
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
    Validate that a dataset with compatible media is loaded for the specified model.
    
    Emits a Gradio warning describing the problem when validation fails (no media, no valid image/video files, or model-media incompatibility).
    
    Parameters:
        model_id (str): Identifier of the target model whose supported media types will be checked.
    
    Returns:
        `true` if the dataset contains media compatible with the model, `false` otherwise.
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
    Check that a dataset containing at least one image is loaded.
    
    Displays a Gradio warning if no dataset or no images are present.
    
    Returns:
        True if a dataset with at least one image is loaded, False otherwise.
    """
    if not app.dataset or not app.dataset.images:
        gr.Warning("No media found. Please load a folder first.")
        return False
    return True


def start_processing(is_valid):
    """
    Set UI state for starting or aborting a run.
    
    Parameters:
        is_valid (bool): Whether the run inputs are valid; if False the UI is reset to the idle state, if True the UI is set to a processing state.
    
    Returns:
        tuple: A 3-tuple of Gradio update objects in the order (run_button_update, download_wrapper_visibility_update, download_button_update).
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
            elem_classes="processing-btn"
        )
    )


def run_with_dynamic_state(app, build_args_fn, *args):
    """
    Run inference using dynamically built arguments and produce UI update objects for the Gradio interface.
    
    Parameters:
        app: CaptioningApp-like object providing run_inference and current_model_id.
        build_args_fn: Function that accepts the provided input arguments (excluding the final validation flag) and returns an args dictionary for inference.
        *args: Positional inputs where the final element is a boolean validation flag; preceding elements are passed to build_args_fn.
    
    Returns:
        tuple: (gallery_data, run_button_update, download_group_update, download_button_update)
            - gallery_data: Data to populate the results gallery.
            - run_button_update: Gradio update for the Run button (resets label and interactivity).
            - download_group_update: Gradio update controlling visibility of the download group.
            - download_button_update: Gradio update configuring the download button (visibility, label, styling).
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
            icon="src/core/download_white.svg",
            elem_classes="download-btn"
        )

    return gallery_data, gr.update(value="Run Captioning", interactive=True), dl_grp, dl_btn
