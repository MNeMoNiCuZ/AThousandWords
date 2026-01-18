"""Settings logic - all settings-related functions extracted from app.py."""

import math
import logging
import gradio as gr

from .constants import filter_user_overrides
import src.features as feature_registry

logger = logging.getLogger("GUI")


def calc_gallery_height(app):
    """Calculate dynamic gallery height based on rows/cols to maintain aspect ratio."""
    if app.gallery_rows == 0:
        return None
        
    BASE_ROW_HEIGHT = 245
    REF_COLS = 6
    
    scale = REF_COLS / max(1, app.gallery_columns)
    return int(app.gallery_rows * BASE_ROW_HEIGHT * scale)


def save_last_model(app, mod_id):
    """Saves the last used model to user config."""
    if mod_id:
        app.current_model_id = mod_id
        app.config_mgr.user_config['last_model'] = mod_id
        app.config_mgr._save_yaml(app.config_mgr.user_config_path, app.config_mgr.user_config)


def move_model_up(app, selected_model, current_order_state):
    """Move selected model up in the order list."""
    if not selected_model:
        gr.Warning("Please select a model first")
        return gr.update(), gr.update(), current_order_state
    
    current_order = list(current_order_state)
    
    if selected_model not in current_order:
        gr.Warning(f"Model {selected_model} not found in order list")
        return gr.update(), gr.update(), current_order_state
    
    idx = current_order.index(selected_model)
    if idx == 0:
        gr.Info("Model is already at the top")
        return gr.update(), gr.update(), current_order_state
    
    current_order[idx], current_order[idx - 1] = current_order[idx - 1], current_order[idx]
    
    return (
        gr.update(choices=current_order, value=selected_model), 
        gr.update(value="\n".join(current_order)),
        current_order
    )


def move_model_down(app, selected_model, current_order_state):
    """Move selected model down in the order list."""
    if not selected_model:
        gr.Warning("Please select a model first")
        return gr.update(), gr.update(), current_order_state
    
    current_order = list(current_order_state)
    
    if selected_model not in current_order:
        gr.Warning(f"Model {selected_model} not found in order list")
        return gr.update(), gr.update(), current_order_state
    
    idx = current_order.index(selected_model)
    if idx >= len(current_order) - 1:
        gr.Info("Model is already at the bottom")
        return gr.update(), gr.update(), current_order_state
    
    current_order[idx], current_order[idx + 1] = current_order[idx + 1], current_order[idx]
    
    return (
        gr.update(choices=current_order, value=selected_model), 
        gr.update(value="\n".join(current_order)),
        current_order
    )


def move_tool_up(app, selected_tool, current_order_state):
    """Move selected tool up in the order list."""
    if not selected_tool:
        gr.Warning("Please select a tool first")
        return gr.update(), gr.update(), current_order_state
    
    current_order = list(current_order_state)
    
    if selected_tool not in current_order:
        gr.Warning(f"Tool {selected_tool} not found in order list")
        return gr.update(), gr.update(), current_order_state
    
    idx = current_order.index(selected_tool)
    if idx == 0:
        gr.Info("Tool is already at the top")
        return gr.update(), gr.update(), current_order_state
    
    current_order[idx], current_order[idx - 1] = current_order[idx - 1], current_order[idx]
    
    return (
        gr.update(choices=current_order, value=selected_tool), 
        gr.update(value="\n".join(current_order)),
        current_order
    )


def move_tool_down(app, selected_tool, current_order_state):
    """Move selected tool down in the order list."""
    if not selected_tool:
        gr.Warning("Please select a tool first")
        return gr.update(), gr.update(), current_order_state
    
    current_order = list(current_order_state)
    
    if selected_tool not in current_order:
        gr.Warning(f"Tool {selected_tool} not found in order list")
        return gr.update(), gr.update(), current_order_state
    
    idx = current_order.index(selected_tool)
    if idx >= len(current_order) - 1:
        gr.Info("Tool is already at the bottom")
        return gr.update(), gr.update(), current_order_state
    
    current_order[idx], current_order[idx + 1] = current_order[idx + 1], current_order[idx]
    
    return (
        gr.update(choices=current_order, value=selected_tool), 
        gr.update(value="\n".join(current_order)),
        current_order
    )


def save_settings(app, vram, models_checked, gal_cols, gal_rows, limit_cnt, o_dir, o_fmt, over, rec, con, unload, pre, suf, clean, collapse, normalize, remove_cn, strip_loop, max_w, max_h, current_mod_id, model_ver, batch_sz, max_tok, settings_state, items_per_page, tools_checked, tool_order):
    """Save settings from the main UI directly to user_config.yaml."""
    
    app.config_mgr.user_config.update({
        'gpu_vram': int(vram) if vram is not None else 24,
        'gallery_columns': int(gal_cols),
        'gallery_rows': int(gal_rows),
        'gallery_items_per_page': int(items_per_page) if items_per_page else 50,
        'limit_count': int(limit_cnt) if limit_cnt and str(limit_cnt).strip() else 0,
        'disabled_models': list(set(app.models) - set(models_checked)),
        'output_dir': o_dir,
        'output_format': o_fmt,
        'overwrite': over,
        'recursive': rec,
        'print_console': con,
        'prefix': pre,
        'suffix': suf,
        'clean_text': clean,
        'collapse_newlines': collapse,
        'normalize_text': normalize,
        'remove_chinese': remove_cn,
        'strip_loop': strip_loop,
        'max_width': int(max_w) if max_w else None,
        'max_height': int(max_h) if max_h else None,
        'max_width': int(max_w) if max_w else None,
        'max_height': int(max_h) if max_h else None,
        'unload_model': unload,
        'disabled_tools': list(set(app.tools) - set(tools_checked)),
        'tool_order': tool_order.split('\n') if isinstance(tool_order, str) else tool_order
    })
    
    if current_mod_id and isinstance(current_mod_id, str):
        clean_config = app.config_mgr.get_model_defaults(current_mod_id)
        clean_defaults_raw = clean_config.get('defaults', {})
        model_defaults = app.config_mgr._resolve_version_specific(clean_defaults_raw, model_ver)
        
        model_config = app.config_mgr.get_model_config(current_mod_id)
        supported_features = set(model_config.get('features', []))
        
        model_versions = model_config.get('model_versions', {})
        model_supports_versions = bool(model_versions)
        valid_versions = list(model_versions.keys()) if model_supports_versions else []
        
        if model_ver:
            if not model_supports_versions:
                logging.warning(f"Cannot save model_version '{model_ver}' for '{current_mod_id}' - model does not support versions")
                gr.Warning(f"Model '{current_mod_id}' does not support versions. Version setting ignored.")
                model_ver = None
            elif model_ver not in valid_versions:
                logging.warning(f"Cannot save invalid model_version '{model_ver}' for '{current_mod_id}'. Valid: {valid_versions}")
                gr.Warning(f"Invalid version '{model_ver}' for model '{current_mod_id}'. Valid versions: {', '.join(valid_versions)}")
                model_ver = None
        
        all_feature_values = {
            'batch_size': batch_sz,
            'max_tokens': max_tok
        }
        
        if model_ver:
            all_feature_values['model_version'] = model_ver
        
        if settings_state:
            all_feature_values.update(settings_state)
        
        if 'prompt_presets' in all_feature_values:
            if all_feature_values['prompt_presets'] != "Custom":
                if 'task_prompt' in all_feature_values:
                    del all_feature_values['task_prompt']
        
        UNIVERSAL_FEATURES = {'batch_size', 'max_tokens', 'custom_task_prompt'}
        
        if 'model_settings' not in app.config_mgr.user_config:
            app.config_mgr.user_config['model_settings'] = {}
        
        existing_root = app.config_mgr.user_config['model_settings'].get(current_mod_id, {})
        
        if model_ver:
            existing_versions = existing_root.get('versions', {})
            
            new_root = {
                'model_version': model_ver,
                'versions': existing_versions
            }
            
            if model_ver not in new_root['versions']:
                 new_root['versions'][model_ver] = {}
            
            target_dict = new_root['versions'][model_ver]
            
            for feature_name, value in all_feature_values.items():
                if feature_name in supported_features or feature_name in UNIVERSAL_FEATURES or feature_name == 'model_version':
                    default_val = model_defaults.get(feature_name)
                    if default_val is None:
                        feature = feature_registry.get_feature(feature_name)
                        if feature: default_val = feature.get_default()
                    
                    if value != default_val:
                        target_dict[feature_name] = value
                    else:
                        if feature_name in target_dict:
                            target_dict.pop(feature_name)
                            
            app.config_mgr.user_config['model_settings'][current_mod_id] = new_root
            
        else:
            if current_mod_id not in app.config_mgr.user_config['model_settings']:
                app.config_mgr.user_config['model_settings'][current_mod_id] = {}
                
            target_dict = app.config_mgr.user_config['model_settings'][current_mod_id]
            
            for feature_name, value in all_feature_values.items():
                if feature_name in supported_features or feature_name in UNIVERSAL_FEATURES or feature_name == 'model_version':
                    default_val = model_defaults.get(feature_name)
                    if default_val is None:
                        feature = feature_registry.get_feature(feature_name)
                        if feature: default_val = feature.get_default()
                        
                    if value != default_val:
                        target_dict[feature_name] = value
                    else:
                        if feature_name in target_dict:
                            target_dict.pop(feature_name)
    
    filtered_config = filter_user_overrides(app.config_mgr.user_config)
    
    app.config_mgr.user_config = filtered_config
    
    app.config_mgr._save_yaml(app.config_mgr.user_config_path, filtered_config)
    
    if current_mod_id and isinstance(current_mod_id, str) and current_mod_id in app.enabled_models:
        save_last_model(app, current_mod_id)

    app.refresh_models()
    app.gallery_columns = int(gal_cols)
    app.gallery_rows = int(gal_rows)
    app.gallery_items_per_page = int(items_per_page) if items_per_page else 50
    

    gr.Info("Settings saved successfully!")
    return []


def save_settings_simple(app, vram, system_ram, models_checked, gal_cols, gal_rows, unload, model_order_text, items_per_page, theme_mode, tools_checked, tool_order_text):
    """Save settings from the Settings tab (simplified version)."""
    model_order_lines = [line.strip() for line in model_order_text.split('\n') if line.strip()]
    tool_order_lines = [line.strip() for line in tool_order_text.split('\n') if line.strip()]
    
    all_models_set = set(app.models)
    invalid_models = [m for m in model_order_lines if m not in all_models_set]
    if invalid_models:
        gr.Warning(f"Unknown models in order list (ignored): {', '.join(invalid_models)}")
    
    valid_model_order = [m for m in model_order_lines if m in all_models_set]
    
    app.config_mgr.user_config.update({
        'gpu_vram': vram,
        'system_ram': system_ram,
        'gallery_columns': gal_cols,
        'gallery_rows': gal_rows,
        'gallery_items_per_page': int(items_per_page) if items_per_page else 50,
        'disabled_models': list(set(app.models) - set(models_checked)),
        'unload_model': unload,
        'model_order': valid_model_order,
        'theme_mode': theme_mode,
        'disabled_tools': list(set(app.tools) - set(tools_checked)),
        'tool_order': tool_order_lines
    })
    
    app.gallery_columns = int(gal_cols)
    app.gallery_rows = int(gal_rows)
    app.gallery_items_per_page = int(items_per_page) if items_per_page else 50
    
    filtered_config = filter_user_overrides(app.config_mgr.user_config)
    app.config_mgr._save_yaml(app.config_mgr.user_config_path, filtered_config)
    
    app._model_mgr.all_models = app.config_mgr.list_models()
    app._model_mgr.enabled_models = app.config_mgr.get_enabled_models()
    
    model_dd_update, model_chk_update = app.refresh_models()
    
    multi_updates = []
    for model_id in app.models:
        is_visible = model_id in app.enabled_models
        multi_updates.append(gr.update(visible=is_visible))
        
    for model_id in app.models:
        is_visible = model_id in app.enabled_models
        multi_updates.append(gr.update(visible=is_visible))
        
    if gal_rows == 0:
        gr.Info("Gallery hidden - refresh the page (F5) to apply")
    else:
        gr.Info("Settings saved successfully!")
        
    tool_updates = []
    # Make sure to iterate in creation order (startup_tool_order) to match fixed Tab components
    for tool_name in app.startup_tool_order:
        is_visible = tool_name in app.enabled_tools
        tool_updates.append(gr.update(visible=is_visible))

    radio_update = gr.update(choices=app.models, value=None)
    tool_radio_update = gr.update(choices=tool_order_lines, value=None)
    
    return [model_dd_update, model_chk_update, radio_update, tool_radio_update] + multi_updates + tool_updates


def reset_to_defaults(app):
    """Delete user_config.yaml to reset all settings to defaults."""
    import os
    
    user_config_path = app.config_mgr.user_config_path
    
    if user_config_path.exists():
        try:
            os.remove(user_config_path)
            return True, "Settings reset to defaults. Please refresh the page manually."
        except Exception as e:
            logger.error(f"Failed to delete user config: {e}")
            return False, f"Error: Failed to reset settings - {e}"
    else:
        return True, "No custom settings found. Already using defaults."


def load_settings(app):
    """Force re-read from disk (reload) and return UI values."""
    app.config_mgr.user_config = app.config_mgr._load_yaml(app.config_mgr.user_config_path)
    
    cfg = app.config_mgr.get_global_settings()
    
    app.gallery_columns = cfg.get('gallery_columns', 6)
    app.gallery_rows = cfg.get('gallery_rows', 3)
    if 'gallery_rows' not in cfg and 'gallery_height' in cfg:
         app.gallery_rows = max(1, int(cfg['gallery_height']) // 210)
    
    if app.current_model_id not in app.enabled_models and app.enabled_models:
        app.current_model_id = app.enabled_models[0]
    
    
    current_order = app.sorted_models
    
    # Tool settings
    all_tools = app.tools
    full_tool_order = app.sorted_tools
    
    disabled_tools = app.config_mgr.user_config.get('disabled_tools', [])
    enabled_tools = [t for t in full_tool_order if t not in disabled_tools]
    
    try:
         presets_data = app.get_user_presets_dataframe()
         presets_choices = ["All Models"] + app.get_preset_eligible_models()
    except Exception as e:
         logger.error(f"Error loading presets: {e}")
         presets_data = []
         presets_choices = ["All Models"]

    return [
        cfg['gpu_vram'], 
        app.config_mgr.get_enabled_models(),
        app.gallery_columns,
        app.gallery_rows,
        "" if not cfg.get('limit_count') else cfg['limit_count'],
        cfg['output_dir'], cfg['output_format'], cfg['overwrite'],
        cfg['recursive'], cfg['print_console'], cfg.get('unload_model', True), cfg['clean_text'], cfg['collapse_newlines'],
        cfg['normalize_text'], cfg['remove_chinese'], cfg['strip_loop'],
        cfg['max_width'], cfg['max_height'],
        cfg['prefix'], cfg['suffix'],
        app.current_model_id if app.current_model_id else (app.enabled_models[0] if app.enabled_models else ""),
        "\n".join(current_order),
        gr.update(choices=current_order, value=None),
        current_order,  # Update model_order_state
        app.gallery_items_per_page,
        app._get_pagination_vis(),
        enabled_tools,
        "\n".join(full_tool_order),
        gr.update(choices=full_tool_order, value=None),
        full_tool_order  # Update tool_order_state
    ]


def auto_save_setting(app, key, value):
    """Saves a UI setting to user_config.yaml automatically."""
    from .constants import UI_CONFIG_MAP
    if key in UI_CONFIG_MAP:
        config_key = UI_CONFIG_MAP[key]
        app.config_mgr.user_config[config_key] = value
        app.config_mgr._save_yaml(app.config_mgr.user_config_path, app.config_mgr.user_config)


def save_model_defaults(app, mod_id, t, k, mt, rp):
    """Saves current generation settings as user defaults for the model."""
    if not mod_id:
        return "No model selected."
    data = {
        "defaults": {
            "temperature": t,
            "top_k": k,
            "max_tokens": mt,
            "repetition_penalty": rp
        }
    }
    app.config_mgr.save_user_model_config(mod_id, data)
    return f"Saved defaults for {mod_id}"


def reset_to_global(app, key):
    """Returns the global default for a specific key."""
    return app.config_mgr.global_config.get(key, "")
