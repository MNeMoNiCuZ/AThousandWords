"""Model logic for resolving configuration values."""

import logging
from typing import Dict, Any, Optional

logger = logging.getLogger("GUI.Logic")

def resolve_model_values(app, model_id: str, version_override: Optional[str] = None) -> Dict[str, Any]:
    """
    Resolve the effective parameter set for a model by combining defaults and saved user settings.
    
    Determines the active model version with priority: explicit version_override > saved user setting (validated) > model defaults; supports both nested per-version ("new style") and flat ("old style") saved configurations, and will remove invalid saved version entries from user config. The returned mapping is defaults merged with any per-version saved overrides and includes 'model_version' when determined.
    
    Parameters:
        model_id (str): Identifier of the model whose values to resolve.
        version_override (Optional[str]): Explicit version to use instead of saved or default version.
    
    Returns:
        Dict[str, Any]: Resolved model parameter dictionary (defaults merged with saved overrides). Contains 'model_version' when a version could be determined.
    """
    config = app.config_mgr.get_model_config(model_id)
    saved_model_root = app.config_mgr.user_config.get('model_settings', {}).get(model_id, {})
    
    # Check for nested version structure (New Style) vs Flat (Old Style)
    has_versions = 'versions' in saved_model_root
            
    # VALIDATION: Check if model actually supports versions
    model_versions = config.get('model_versions', {})
    model_supports_versions = bool(model_versions)
    valid_versions = list(model_versions.keys()) if model_supports_versions else []
    
    # 1. Determine Current Version with VALIDATION
    # Priority: Override (Dropdown) > Saved Active Version > Defaults
    
    # Load clean defaults for fallback logic
    clean_config = app.config_mgr.get_model_defaults(model_id)
    clean_defaults_raw = clean_config.get('defaults', {})
    
    if version_override:
        current_version = version_override
    else:
        # Try to get saved active version
        # In new style, it's at root. In old style, it's also at root.
        saved_ver = saved_model_root.get('model_version')
        
        # Validate saved version
        if saved_ver:
            if not model_supports_versions:
                # Model doesn't support versions, clear the stale value
                logger.warning(f"Model '{model_id}' does not support versions, but 'model_version: {saved_ver}' was saved. Clearing invalid value.")
                # Remove from user_config
                if 'model_settings' in app.config_mgr.user_config and model_id in app.config_mgr.user_config['model_settings']:
                    app.config_mgr.user_config['model_settings'][model_id].pop('model_version', None)
                    app.config_mgr.save_user_config()
                saved_ver = None  # Ignore it
            elif saved_ver not in valid_versions:
                # Saved version is invalid for this model
                logger.warning(f"Invalid model_version '{saved_ver}' for model '{model_id}'. Valid versions: {valid_versions}. Resetting to default.")
                # Remove from user_config
                if 'model_settings' in app.config_mgr.user_config and model_id in app.config_mgr.user_config['model_settings']:
                    app.config_mgr.user_config['model_settings'][model_id].pop('model_version', None)
                    app.config_mgr.save_user_config()
                saved_ver = None  # Ignore it
        
        if saved_ver:
            current_version = saved_ver
        else:
            # Fallback to Defaults
            if isinstance(clean_defaults_raw, dict):
                # Check if nested defaults (has string keys with dict values)
                first_key = next(iter(clean_defaults_raw), None)
                if first_key and isinstance(first_key, str) and isinstance(clean_defaults_raw.get(first_key), dict):
                    # Nested structure - use first version as default
                    current_version = first_key
                else:
                    # Flat structure - extract version from defaults
                    current_version = clean_defaults_raw.get('model_version')
            else:
                current_version = None

    # 2. Get Defaults for this version (from the clean config)
    defaults = app.config_mgr._resolve_version_specific(clean_defaults_raw, current_version)
    if isinstance(defaults, dict):
        defaults = defaults.copy()
    else:
        defaults = {}
        
    
    # 3. Get Saved Settings Diff for this version
    saved_diff = {}
    if has_versions:
        # New Style: key into specific version
        versions_dict = saved_model_root.get('versions', {})
        # Only apply if this specific version has saved data
        if current_version in versions_dict:
            saved_diff = versions_dict[current_version].copy()
            # Ensure model_version is set in diff (it was removed on save)
            saved_diff['model_version'] = current_version
    else:
        # Old Style: Flat
        # Logic: If flat settings exist, they apply. 
        saved_diff = saved_model_root.copy()
        # Remove metadata keys that aren't settings
        saved_diff.pop('versions', None)

    
    # 4. Merge: Defaults + Saved Diff
    final_values = defaults.copy()
    if saved_diff:
        final_values.update(saved_diff)
        
    # Ensure model_version is set correctly in final dict
    if current_version:
        final_values['model_version'] = current_version
        
    return final_values


def update_model_ui(app, mod_id):
    """
    Builds Gradio UI update states for controls based on the selected model's configuration and capabilities.
    
    If `mod_id` is falsy, returns placeholder updates for all controls.
    
    Parameters:
        mod_id (str): Identifier of the model whose config and defaults drive the UI state.
    
    Returns:
        list: A list of 11 Gradio `update` objects in the following order:
            0. Temperature control (visibility, value)
            1. Top-k control (visibility, value)
            2. Max tokens control (visibility, value)
            3. Repetition penalty control (visibility, value)
            4. Detail mode control (visibility, value)
            5. Include-thinking toggle (visibility, value)
            6. Strip-thinking-tags toggle (visibility, value)
            7. Prompt preset selector (visibility, choices, value)
            8. Recommended batch size (value, info tooltip)
            9. System prompt value
            10. Task prompt value
    """
    import gradio as gr
    
    if not mod_id:
        return [gr.update()] * 11
    
    cfg = app.config_mgr.get_model_config(mod_id)
    caps = cfg.get("capabilities", {})
    defs = cfg.get("defaults", {})
    
    settings = app.config_mgr.get_global_settings()
    vram = settings['gpu_vram']
    rec_batch = app.config_mgr.get_recommended_batch_size(mod_id, vram)
    
    recs = cfg.get("vram_recommendations") or cfg.get("vram_table") or {}
    recommendation_list = "\n".join([f"- {k}GB: Batch {v}" for k, v in recs.items()])
    info_tooltip = f"Number of images to process in parallel.\nVRAM recommendations for this model:\n{recommendation_list}"

    presets = cfg.get("prompt_presets", {})
    preset_keys = list(presets.keys()) if presets else []
    
    has_thinking = caps.get("include_thinking", False)
    
    default_template = defs.get("prompt_presets")
    if default_template and default_template not in preset_keys:
        logger.warning(f"Default template '{default_template}' not found in presets for {mod_id}. Using first available.")
        default_template = None
    
    final_template_value = default_template or (preset_keys[0] if presets else None)

    return [
        gr.update(visible=caps.get("temperature", True), value=defs.get("temperature", 0.5)),
        gr.update(visible=caps.get("top_k", True), value=defs.get("top_k", 40)),
        gr.update(visible=caps.get("max_tokens", True), value=defs.get("max_tokens", 300)),
        gr.update(visible=caps.get("repetition_penalty", True), value=defs.get("repetition_penalty", 1.1)),
        gr.update(visible=caps.get("detail_mode", False), value=defs.get("detail_mode", 1)),
        gr.update(visible=has_thinking, value=defs.get("include_thinking", True)),
        gr.update(visible=has_thinking, value=defs.get("strip_thinking_tags", True)),
        gr.update(visible=bool(presets), choices=preset_keys, value=final_template_value),
        gr.update(value=rec_batch, info=info_tooltip),
        gr.update(value=defs.get("system_prompt", "")),
        gr.update(value=defs.get("task_prompt", ""))
    ]


def apply_preset(app, mod_id, preset_name):
    """
    Apply the named prompt preset for the specified model to a Gradio input.
    
    Returns:
        gr.update: An update that sets the input value to the preset text when the preset exists for the model; otherwise a neutral update (no change).
    """
    import gradio as gr
    
    if not mod_id or not preset_name:
        return gr.update()
    cfg = app.config_mgr.get_model_config(mod_id)
    presets = cfg.get("prompt_presets", {})
    if preset_name in presets:
        return gr.update(value=presets[preset_name])
    return gr.update()



def get_initial_model_state(app, model_id: str) -> Dict[str, Any]:
    """
    Reset and return the model-scoped settings to a fresh state based on the model's defaults.
    
    If `model_id` is falsy, returns an empty dict.
    
    Returns:
        resolved_state (Dict[str, Any]): A dictionary containing only the current model's default settings, merged with any valid saved overrides for that model.
    """
    if not model_id:
        return {}
    
    # Use the same resolution logic as render_features
    # No version override - will use saved or default
    return resolve_model_values(app, model_id, version_override=None)