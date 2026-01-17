"""Settings logic - all settings-related functions extracted from app.py."""

import math
import logging
import gradio as gr

from .constants import filter_user_overrides
import src.features as feature_registry

logger = logging.getLogger("GUI")


def calc_gallery_height(app):
    """
    Compute a scaled gallery height to preserve aspect ratio from app.gallery_rows and app.gallery_columns.
    
    Returns:
        height (int or None): Calculated pixel height for the gallery in pixels; `None` if `app.gallery_rows` is 0.
    """
    if app.gallery_rows == 0:
        return None
        
    BASE_ROW_HEIGHT = 245
    REF_COLS = 6
    
    scale = REF_COLS / max(1, app.gallery_columns)
    return int(app.gallery_rows * BASE_ROW_HEIGHT * scale)


def save_last_model(app, mod_id):
    """
    Persist the specified model identifier as the last-used model in the user's configuration.
    
    Parameters:
        app: Application context providing `current_model_id` and `config_mgr`.
        mod_id (str): Model identifier to save; if falsy, no changes are made.
    
    """
    if mod_id:
        app.current_model_id = mod_id
        app.config_mgr.user_config['last_model'] = mod_id
        app.config_mgr._save_yaml(app.config_mgr.user_config_path, app.config_mgr.user_config)


def move_model_up(app, selected_model, current_order_state):
    """
    Move the selected model one position earlier in the model order.
    
    Parameters:
        selected_model (str): The model identifier chosen to move; if falsy, no action is taken.
        current_order_state (Iterable[str] or list): The current ordered sequence of model identifiers.
    
    Returns:
        tuple: A three-element tuple containing:
            - choices_update: UI update for the model choices/list reflecting the new order.
            - order_text_update: UI update for the multiline order text set to the joined order.
            - new_order (list): The updated list of model identifiers after the move.
    """
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
    """
    Move the selected model one position down within the provided model order.
    
    Parameters:
        selected_model (str): The model identifier chosen to move.
        current_order_state (Sequence[str]): Current ordered sequence of model identifiers.
    
    Returns:
        tuple: A 3-tuple containing:
            - A Gradio update object for the model-order chooser with updated `choices` and the same `value`.
            - A Gradio update object for the textual representation of the order (`value` set to the joined order lines).
            - The new list of model identifiers in their updated order.
    """
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


def save_settings(app, vram, models_checked, gal_cols, gal_rows, limit_cnt, o_dir, o_fmt, over, rec, con, unload, pre, suf, clean, collapse, normalize, remove_cn, strip_loop, max_w, max_h, current_mod_id, model_ver, batch_sz, max_tok, settings_state, items_per_page):
    """
    Persist UI settings to the user's configuration and apply them to the running app.
    
    Updates app.config_mgr.user_config with top-level UI settings (gallery layout, VRAM, disabled models, output options, text-processing flags, image size caps, unload behavior, etc.), records or prunes model-specific feature overrides for the optionally provided model and model version, filters and saves the resulting config to disk, refreshes model state, updates gallery dimensions and pagination, and displays a confirmation message.
    
    Parameters:
        current_mod_id (str|None): Identifier of the currently selected model. When provided, model-specific defaults and supported features are consulted and per-model settings are saved or pruned accordingly.
        model_ver (str|None): Selected model version to persist for the current model. Ignored if the model does not support versions or if the version is invalid.
        settings_state (dict|None): Additional per-request feature values (e.g., generation-related settings) to consider for model-specific persistence.
        items_per_page (int|None): Number of gallery items per page; also stored in the user configuration.
    
    Returns:
        list: An empty list.
    """
    
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
        'unload_model': unload
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


def save_settings_simple(app, vram, system_ram, models_checked, gal_cols, gal_rows, unload, model_order_text, items_per_page):
    """
    Persist a simplified set of Settings-tab preferences to the user configuration, update persisted model order, and refresh model-related UI state.
    
    Parameters:
        app: The application context containing config manager, model manager, and UI helpers.
        vram: GPU VRAM setting value to save.
        system_ram: System RAM setting value to save.
        models_checked: Iterable of model IDs currently enabled/checked in the UI.
        gal_cols: Number of gallery columns to save.
        gal_rows: Number of gallery rows to save (0 hides the gallery).
        unload: Boolean indicating whether models should be unloaded when not in use.
        model_order_text: New model order as a newline-separated string; unknown model IDs are ignored.
        items_per_page: Number of items per gallery page; defaults to 50 when empty or falsy.
    
    Returns:
        A list of Gradio update objects to apply to the UI (model dropdown update, model checkbox update, radio update, followed by per-model visibility updates).
    """
    model_order_lines = [line.strip() for line in model_order_text.split('\n') if line.strip()]
    
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
        'model_order': valid_model_order
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
        gr.Info("💡 Gallery hidden - refresh the page (F5) to apply")
    else:
        gr.Info("Settings saved successfully!")
        
    radio_update = gr.update(choices=app.models, value=None)
    return [model_dd_update, model_chk_update, radio_update] + multi_updates


def reset_to_defaults(app):
    """
    Reset user settings to the application's defaults by removing the user configuration file if present.
    
    Returns:
        tuple: (success, message)
            - success (bool): `True` if defaults are in use (file removed or no file existed), `False` if an error occurred while attempting reset.
            - message (str): Human-readable status or error message describing the outcome.
    """
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
    """
    Reload user and global settings from disk and return the current values needed to populate the settings UI.
    
    This forces the configuration manager to re-read the user YAML, applies fallbacks from global settings, ensures a valid current model selection, attempts to load user presets, and computes gallery layout defaults if needed.
    
    Returns:
        A list containing the UI values in the following order:
        - GPU VRAM setting value
        - enabled models list
        - gallery columns
        - gallery rows
        - limit count (empty string if not set)
        - output directory
        - output format
        - overwrite flag
        - recursive flag
        - print to console flag
        - unload model flag
        - clean text flag
        - collapse newlines flag
        - normalize text flag
        - remove Chinese characters flag
        - strip loop flag
        - max width
        - max height
        - prefix string
        - suffix string
        - current model id (or empty string if none)
        - model order as a single newline-joined string
        - a Gradio update object for the model order choices
        - gallery items per page
        - pagination visibility update handle
    """
    app.config_mgr.user_config = app.config_mgr._load_yaml(app.config_mgr.user_config_path)
    
    cfg = app.config_mgr.get_global_settings()
    
    app.gallery_columns = cfg.get('gallery_columns', 6)
    app.gallery_rows = cfg.get('gallery_rows', 3)
    if 'gallery_rows' not in cfg and 'gallery_height' in cfg:
         app.gallery_rows = max(1, int(cfg['gallery_height']) // 210)
    
    if app.current_model_id not in app.enabled_models and app.enabled_models:
        app.current_model_id = app.enabled_models[0]
    
    current_order = app.config_mgr.user_config.get('model_order', app.config_mgr.global_config.get('model_order', app.models))
    
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
        app.gallery_items_per_page,
        app._get_pagination_vis()
    ]


def auto_save_setting(app, key, value):
    """
    Persist a single UI setting into the user's configuration using the UI-to-config key map.
    
    If `key` is present in the module's `UI_CONFIG_MAP`, the function writes `value` to the mapped configuration key and saves the user configuration to disk; if `key` is not mapped, no changes are made.
    
    Parameters:
        app: Application object providing access to `config_mgr`.
        key (str): UI setting identifier to be mapped into the user configuration.
        value: Value to store for the mapped configuration key.
    """
    from .constants import UI_CONFIG_MAP
    if key in UI_CONFIG_MAP:
        config_key = UI_CONFIG_MAP[key]
        app.config_mgr.user_config[config_key] = value
        app.config_mgr._save_yaml(app.config_mgr.user_config_path, app.config_mgr.user_config)


def save_model_defaults(app, mod_id, t, k, mt, rp):
    """
    Save generation defaults for a specific model.
    
    Parameters:
        mod_id (str): Identifier of the model to save defaults for.
        t (float): Sampling temperature.
        k (int): Top-k sampling value.
        mt (int): Maximum tokens for generation.
        rp (float): Repetition penalty.
    
    Returns:
        str: `"No model selected."` if `mod_id` is falsy; otherwise `"Saved defaults for {mod_id}"`.
    """
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
    """
    Retrieve the global default value for a configuration key.
    
    Parameters:
        key (str): Configuration key to look up in the global defaults.
    
    Returns:
        The global default value for `key`, or an empty string if the key is not present.
    """
    return app.config_mgr.global_config.get(key, "")