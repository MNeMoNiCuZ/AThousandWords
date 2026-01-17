"""User presets logic - CRUD operations for prompt presets."""

import gradio as gr
import logging

logger = logging.getLogger("GUI")


def get_preset_eligible_models(app):
    """Return list of models that support custom prompts (eligible for presets)."""
    eligible = []
    for model_id in app.models:
        config = app.config_mgr.get_model_config(model_id)
        features = config.get('features', [])
        has_feature = 'task_prompt' in features
        explicitly_disabled = config.get('supports_custom_prompts', True) is False
        if has_feature and not explicitly_disabled:
            eligible.append(model_id)
    return eligible


def get_user_presets_dataframe(app):
    """Get user presets formatted for the Settings Dataframe."""
    presets = app.config_mgr.user_config.get("user_prompt_presets", [])
    model_order = app.config_mgr.user_config.get('model_order', app.models)
    
    def sort_key(p):
        p_model = p.get('model', 'All Models') or 'All Models'
        p_name = p.get('name', '')
        if p_model == 'All Models':
            m_idx = -1
        else:
            try:
                m_idx = model_order.index(p_model)
            except ValueError:
                m_idx = 9999
        return (m_idx, p_name)

    sorted_presets = sorted(presets, key=sort_key)
    
    data = []
    for p in sorted_presets:
        p_model = p.get('model', 'All Models') or 'All Models'
        data.append([p_model, p.get('name', ''), p.get('text', ''), "🗑️"])
    return data


def save_user_preset(app, model_scope, name, text):
    """Save (Upsert) a user preset."""
    if not name or not text:
        gr.Warning("Preset Name and Text are required.")
        return get_user_presets_dataframe(app), gr.update(choices=["All Models"] + get_preset_eligible_models(app))
    
    if model_scope == "All Models":
        model_scope = "All Models"
    
    presets = app.config_mgr.user_config.get("user_prompt_presets", [])
    
    # Remove existing if exists (Update)
    presets = [
        p for p in presets 
        if not ((p.get('model') or 'All Models') == model_scope and p.get('name') == name)
    ]
    
    presets.append({"model": model_scope, "name": name, "text": text})
    
    app.config_mgr.user_config["user_prompt_presets"] = presets
    app.config_mgr.save_user_config()
    
    gr.Info(f"Saved preset '{name}' for {model_scope}")
    return get_user_presets_dataframe(app), gr.update(choices=["All Models"] + get_preset_eligible_models(app))


def delete_user_preset(app, model_scope, name):
    """Delete a user preset."""
    if not name:
        gr.Warning("Select a preset to delete.")
        return get_user_presets_dataframe(app)
    
    presets = app.config_mgr.user_config.get("user_prompt_presets", [])
    
    new_presets = [
        p for p in presets 
        if not ((p.get('model') or 'All Models') == model_scope and p.get('name') == name)
    ]
    
    if len(new_presets) == len(presets):
        gr.Warning(f"Preset '{name}' not found.")
        return get_user_presets_dataframe(app)
    
    app.config_mgr.user_config["user_prompt_presets"] = new_presets
    app.config_mgr.save_user_config()
    
    gr.Info(f"Deleted preset '{name}'")
    return get_user_presets_dataframe(app)
