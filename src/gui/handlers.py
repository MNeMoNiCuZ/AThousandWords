# gui/handlers.py
"""
Event handlers for the Gradio interface.
Includes auto-save, inference wrapper, model switching, and settings persistence.
"""

import gradio as gr
import logging
from .constants import GLOBAL_DEFAULTS, filter_user_overrides
import src.features as feature_registry

logger = logging.getLogger("GUI")

# Universal features that are ALWAYS shown for ALL models regardless of model config
# These are core functionality features that every model uses
UNIVERSAL_FEATURES = {'batch_size', 'max_tokens'}


def get_system_ram_gb():
    """Get total system RAM in GB with smart rounding to common sizes."""
    try:
        import psutil
        raw_gb = psutil.virtual_memory().total / (1024 ** 3)
        
        # Smart rounding: Snap to multiples of 8 if within 5% tolerance
        # Common RAM sizes: 8, 16, 24, 32, 48, 64, 96, 128, 192, 256, etc.
        rounded = round(raw_gb)
        
        # Find nearest multiple of 8
        nearest_mult_8 = round(rounded / 8) * 8
        
        # If within 5% of a multiple of 8, snap to it
        tolerance = 0.05
        if abs(raw_gb - nearest_mult_8) / raw_gb < tolerance:
            return nearest_mult_8
        else:
            return rounded
    except ImportError:
        return 16  # Fallback if psutil not available



def get_model_description_html(app, model_id):
    """
    Generates the HTML for the model description and metadata.
    Refactored out of the handler to allow dynamic rendering in main.py.
    """
    if not model_id:
        return ""
        
    model_info = app.config_mgr.get_model_config(model_id)
    if not model_info:
        return ""

    description = model_info.get("description", "")
    model_path = model_info.get("model_path", "")
    
    # Metadata Line (Path + License)
    meta_parts = []
    
    # 1. Path
    if model_path:
        hf_url = f"https://huggingface.co/{model_path}"
        meta_parts.append(f"<a href='{hf_url}' target='_blank' style='text-decoration: none;'><code style='font-size: 0.85em; background: none; color: inherit; cursor: pointer;'>{model_path}</code></a>")
    
    # 2. Speed
    speed_val = model_info.get('caption_speed')
    if speed_val:
        # Add separator if needed
        if meta_parts:
            meta_parts.append("<span style='opacity: 0.5; margin: 0 8px;'>|</span>")
        
        meta_parts.append(f"<span style='font-size: 0.85em; opacity: 0.8;'>Speed: {speed_val} it/s</span>")
    
    # 3. License
    licence = model_info.get('licence', '')
    licence_url = model_info.get('licence-url', '')
    
    if licence:
        # Check for Non-Commercial/Restrictive to apply red styling
        container_style = 'font-size: 0.85em; opacity: 0.8;'
        
        # Triggers for warning color (Red)
        warning_triggers = ["non-commercial", "unknown", "business source license", "bsl"]
        is_warning = any(trigger in licence.lower() for trigger in warning_triggers)
        
        if is_warning:
            # Warning: Red & Bold (Value Only)
            warn_style = 'color: #ff5555; font-weight: bold;'
            container_style = 'font-size: 0.85em; opacity: 1.0;' # Full opacity for visibility
            
            if licence_url:
                lic_str = f"<a href='{licence_url}' target='_blank' style='text-decoration: underline; {warn_style}'>{licence}</a>"
            else:
                lic_str = f"<span style='{warn_style}'>{licence}</span>"
        else:
            # Standard: Inherit Color
            if licence_url:
                lic_str = f"<a href='{licence_url}' target='_blank' style='text-decoration: underline; color: inherit;'>{licence}</a>"
            else:
                lic_str = licence
        
        # Add separator if path exists
        if meta_parts:
            meta_parts.append("<span style='opacity: 0.5; margin: 0 8px;'>|</span>")
            
        meta_parts.append(f"<span style='{container_style}'>License: {lic_str}</span>")
        
    meta_html = "".join(meta_parts)
    if meta_html:
        meta_html = f"<div style='margin-bottom: 6px;'>{meta_html}</div>"
        
    # Construct HTML
    html_val = (
        f"<div style='width: 100%; white-space: normal; padding: 5px;'>"
        f"<h2 style='margin: 0 0 5px 0; padding: 0; font-size: 1.4em;'>{model_id}</h2>"
        f"{meta_html}"
        f"<span style='font-size: 0.9em; font-weight: normal; display: block; opacity: 0.9;'>{description}</span>"
        f"</div>"
    )
    return html_val


def create_update_model_settings_handler(app, static_components, model_description_component=None):
    """
    Updates STATIC model settings (Version, Batch Size, Max Tokens).
    Dynamic settings AND Description are now handled by @gr.render in main.py.
    """
    def update_model_settings_ui(model_id):
        # Force reload user config
        app.config_mgr.user_config = app.config_mgr._load_yaml(app.config_mgr.user_config_path)
        
        # Default hidden updates
        updates = [gr.update(visible=False) for _ in static_components.values()]
        desc_update = gr.update(value="", visible=False)
        
        if not model_id:
            return [desc_update] + updates
            
        app.save_last_model(model_id)
        model_info = app.config_mgr.get_model_config(model_id)
        if not model_info:
            return [desc_update] + updates

        defaults = model_info.get("defaults", {})
        saved_settings = app.config_mgr.user_config.get('model_settings', {}).get(model_id, {})
        merged_values = {**defaults, **saved_settings}
        
        # Update Description
        html_val = get_model_description_html(app, model_id)
        desc_update = gr.update(value=html_val, visible=True)
        
        # Update Static Components
        comp_updates = []
        for name, comp in static_components.items():
            is_visible = False
            update_kwargs = {}
            
            # Use universal or merged value
            val = merged_values.get(name)
            
            if name == 'batch_size':
                # Check if model config explicitly hides batch_size 
                # (for models like Moondream that don't benefit from batching)
                if model_info.get('hide_batch_size'):
                    is_visible = False
                    update_kwargs = {'value': 1}  # visible is added at the end
                else:
                    # VRAM Logic for Batch Size (reused from version handler)
                    vram = app.config_mgr.get_global_settings().get('gpu_vram', 24)
                    version_key = merged_values.get('model_version')
                    rec_batch = app.config_mgr.get_recommended_batch_size(model_id, vram, variant=version_key)
                    
                    # Retrieve VRAM info for tooltip
                    full_recs = model_info.get("vram_recommendations") or model_info.get("vram_table") or {}
                    
                    # Correctly navigate nested tables if present
                    recs = {}
                    if full_recs:
                        first_key = next(iter(full_recs))
                        if isinstance(first_key, str):
                            # Nested - use version or default from config
                            display_version = version_key
                            if not display_version:
                                 display_version = model_info.get('defaults', {}).get('model_version')
                            
                            if display_version and display_version in full_recs:
                                recs = full_recs[display_version]
                        else:
                            recs = full_recs
                    
                    # Find user tier
                    user_tier = None
                    if recs:
                        sorted_tiers = sorted([k for k in recs.keys() if isinstance(k, int)])
                        for tier in sorted_tiers:
                            if vram >= tier:
                                user_tier = tier

                    # Format tooltip
                    if recs:
                        min_tier = sorted_tiers[0] if sorted_tiers else None
                        rec_parts = []
                        for k, v in sorted(recs.items()):
                            # Common class for JS click handling
                            rec_item_class = "vram-rec-item"
                            # Use double quotes for attribute to ensure better HTML compliance
                            data_attr = f'data-bs="{v}"'
                            
                            if k == user_tier:
                                rec_parts.append(f"<span class='{rec_item_class}' {data_attr} style='color: green; font-weight: bold; font-size: 1.25em; text-decoration: underline;'>{k}GB:{v}</span>")
                            elif user_tier is None and k == min_tier:
                                rec_parts.append(f"<span class='{rec_item_class}' {data_attr} style='color: red; font-weight: bold; font-size: 1.25em; text-decoration: underline;'>{k}GB:{v}</span>")
                            else:
                                rec_parts.append(f"<span class='{rec_item_class}' {data_attr}>{k}GB:{v}</span>")
                        rec_list = " │ ".join(rec_parts)
                    else:
                        rec_list = "No data"
                    
                    # Use "RAM" label for ONNX models that use system RAM instead of GPU VRAM
                    mem_label = "RAM" if model_info.get('uses_system_ram') else "VRAM"
                    info = f"{mem_label}: {rec_list}"

                    # Auto-fill with recommendation when switching models
                    val = rec_batch
                    
                    is_visible = True
                    update_kwargs = {'label': 'Batch Size', 'value': val, 'info': info, 'elem_id': 'batch_size'}
                
            elif name == 'max_tokens':
                if val is None: val = 512
                is_visible = True
                update_kwargs = {'value': val}
                
            elif name == 'model_version':
                if 'model_versions' in model_info and model_info['model_versions']:
                    choices = list(model_info['model_versions'].keys())
                    if val not in choices and choices:
                        val = choices[0]
                    is_visible = True
                    update_kwargs = {'choices': choices, 'value': val}
            
            comp_updates.append(gr.update(visible=is_visible, **update_kwargs))

        return [desc_update] + comp_updates

    return update_model_settings_ui


def create_auto_save_handler(app):
    """Creates the ui_auto_save function bound to the app instance."""
    def ui_auto_save(pre, suf, o_dir, o_fmt, over, rec, con, clean, collapse, normalize, remove_cn, strip_loop, max_w, max_h):
        app.config_mgr.user_config.update({
            'prefix': pre, 'suffix': suf, 'output_dir': o_dir, 'output_format': o_fmt,
            'overwrite': over, 'recursive': rec, 'print_console': con,
            'clean_text': clean, 'collapse_newlines': collapse,
            'normalize_text': normalize, 'remove_chinese': remove_cn, 'strip_loop': strip_loop,
            'max_width': int(max_w) if max_w else None, 'max_height': int(max_h) if max_h else None
        })
        # Filter to only save user overrides
        filtered_config = filter_user_overrides(app.config_mgr.user_config)
        app.config_mgr._save_yaml(app.config_mgr.user_config_path, filtered_config)
    
    return ui_auto_save


def create_inference_wrapper(app, settings_state):
    """
    Creates inference wrapper that reads directly from settings_state (gr.State)
    and static inputs passed at click-time.
    """
    def run_inference_wrapper(model_id, image_input, folder_input, dragging_files, 
                              output_folder_input, 
                              # Static settings passed from UI values
                              model_version, batch_size, max_tokens,
                              # Dynamic settings from State
                              dynamic_settings):
        
        # Build Args from Static + Dynamic
        args = dynamic_settings.copy() if dynamic_settings else {}
        args['model_version'] = model_version
        args['batch_size'] = int(batch_size) if batch_size else 1
        args['max_tokens'] = int(max_tokens) if max_tokens else 512
        
        # Add universal settings
        settings = app.config_mgr.get_global_settings()
        args["gpu_vram"] = settings['gpu_vram']

        # VALIDATION: Check for "From File" / "From Metadata" compatibility
        # These modes require stable local paths (not temp files from drag-and-drop)
        prompt_source = dynamic_settings.get("prompt_source", "Prompt Presets")
        if prompt_source in ["From File", "From Metadata"]:
             # Check if we are potentially using drag-and-drop results or non-local inputs
             # If folder_input is empty and we have images, it's likely Drag & Drop or legacy single image
             if not folder_input and (image_input or dragging_files):
                 # We can't easily rely on per-file validation here without more complex logic,
                 # but generally D&D uploads end up in temp dirs which might not have the companion files.
                 # However, app.load_files moves them to 'user_data/uploads'.
                 # If the user dragged a PAIR (img + txt), it might be there.
                 # But "From Metadata" definitely needs the original file.
                 
                 # Strict Safety: Abort to prevent confusion
                 gr.Warning(f"⚠️ '{prompt_source}' mode requires selecting a Local Folder input. Drag & Drop is not supported for this mode.")
                 return []
        
        # Call App Inference
        # We assume app.run_inference handles 'model_settings_override' or similar,
        # OR we modify run_inference to accept args?
        # The original code called app.run_inference(model_id, args).
        # And app.run_inference handled the file resolution?
        # WAIT: The original wrapper in THIS FILE handled file resolution (lines 278-313 in old file).
        # But wait, lines 278-313 in old file DID NOT handle files, it just merged args.
        # Main.py called `run_inference_wrapper` with `inputs=[model_sel, image_input, ...]`
        # So `app.run_inference` MUST handle file logic.
        
        return app.run_inference(model_id, args, image_input, folder_input, dragging_files, output_folder_input)

    return run_inference_wrapper


def create_gallery_cols_saver(app):
    """Creates the save_gallery_cols function bound to the app instance."""
    def save_gallery_cols(cols):
        app.config_mgr.user_config['gallery_columns'] = int(cols)
        app.config_mgr._save_yaml(app.config_mgr.user_config_path, app.config_mgr.user_config)

    
    return save_gallery_cols


def create_version_change_handler(app):
    """Creates a handler to update batch size recommendations when version changes."""
    def update_batch_size_on_version_change(model_id, version):
        if not model_id:
            return gr.update()
        
        # Get model config first to check if it uses system RAM
        model_info = app.config_mgr.get_model_config(model_id)
        
        # Validate version belongs to this model (prevents stale values from model switch)
        valid_versions = list(model_info.get('model_versions', {}).keys()) if model_info.get('model_versions') else []
        if valid_versions and version not in valid_versions:
            # Stale version from previous model, use default or first available
            version = model_info.get('defaults', {}).get('model_version') or (valid_versions[0] if valid_versions else None)
        
        # Use system RAM for ONNX models, VRAM for GPU models
        if model_info.get('uses_system_ram'):
            # Use saved system_ram from config, fallback to auto-detection
            memory_gb = app.config_mgr.get_global_settings().get('system_ram', get_system_ram_gb())
        else:
            memory_gb = app.config_mgr.get_global_settings().get('gpu_vram', 24)
        
        # Get recommended batch size for this specific version
        rec_batch = app.config_mgr.get_recommended_batch_size(model_id, memory_gb, variant=version)
        
        # Get config for tooltip
        full_recs = model_info.get("vram_recommendations") or model_info.get("vram_table") or {}
        
        # Get specific table for this variant
        recs = {}
        if full_recs:
            first_key = next(iter(full_recs))
            if isinstance(first_key, str):
                # Nested
                display_version = version
                if not display_version:
                     display_version = model_info.get('defaults', {}).get('model_version')
                
                if display_version and display_version in full_recs:
                    recs = full_recs[display_version]
            else:
                # Flat
                recs = full_recs
        
        # Find user tier
        user_tier = None
        if recs:
            sorted_tiers = sorted([k for k in recs.keys() if isinstance(k, int)])
            for tier in sorted_tiers:
                if memory_gb >= tier:
                    user_tier = tier
        
        # Format tooltip with clickable spans
        if recs:
            min_tier = sorted_tiers[0] if sorted_tiers else None
            rec_parts = []
            for k, v in sorted(recs.items()):
                # Common class for JS click handling
                rec_item_class = "vram-rec-item"
                # Use double quotes for attribute to ensure better HTML compliance
                data_attr = f'data-bs="{v}"'
                
                # Non-clickable but styled (WAIT - User wants clickable! Make ALL clickable)
                # Logic: Even if it's the current tier, user might want to re-click or change back to it.
                if k == user_tier:
                    rec_parts.append(f"<span class='{rec_item_class}' {data_attr} style='color: green; font-weight: bold; font-size: 1.25em; text-decoration: underline;'>{k}GB:{v}</span>")
                elif user_tier is None and k == min_tier:
                    rec_parts.append(f"<span class='{rec_item_class}' {data_attr} style='color: red; font-weight: bold; font-size: 1.25em; text-decoration: underline;'>{k}GB:{v}</span>")
                else:
                    rec_parts.append(f"<span class='{rec_item_class}' {data_attr}>{k}GB:{v}</span>")
            rec_list = " │ ".join(rec_parts)
        else:
            rec_list = "No data"
        
        # Use "RAM" label for ONNX models that use system RAM instead of GPU VRAM
        mem_label = "System RAM" if model_info.get('uses_system_ram') else "VRAM"
        info = f"{mem_label}: {rec_list}"
        
        return gr.update(value=rec_batch, info=info, elem_id='batch_size')
    
    return update_batch_size_on_version_change
