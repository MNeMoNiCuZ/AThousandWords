"""User presets logic - CRUD operations for prompt presets."""

import gradio as gr
import logging

logger = logging.getLogger("GUI")


def get_preset_eligible_models(app):
    """
    Identify models that are eligible for user prompt presets.
    
    Returns:
        eligible_models (list[str]): Model IDs that expose the 'task_prompt' feature and have not explicitly disabled custom prompts.
    """
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
    """
    Return a table of user prompt presets formatted for display in the Settings Dataframe.
    
    Presets are sorted by model according to the user's `model_order` (with "All Models" placed before ordered models and unknown models placed last) and then by preset name. Each row contains the preset's model (defaults to "All Models"), name, text, and a delete indicator.
    
    Returns:
        list[list[str]]: 2D list where each row is [model, name, text, "🗑️"].
    """
    presets = app.config_mgr.user_config.get("user_prompt_presets", [])
    model_order = app.config_mgr.user_config.get('model_order', app.models)
    
    def sort_key(p):
        """
        Compute a tuple key for sorting a preset by model order and then by name.
        
        Parameters:
            p (dict): Preset object; expected keys are:
                - 'model' (str, optional): Model scope name; defaults to "All Models" if missing or falsy.
                - 'name' (str, optional): Preset name; defaults to empty string if missing.
        
        Returns:
            tuple: (model_index, name) where `model_index` is:
                - -1 if the model is "All Models",
                - the index of the model in the surrounding `model_order` sequence if present,
                - 9999 if the model is not found in `model_order`.
                `name` is the preset name used for secondary alphabetical ordering.
        """
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
    """
    Save or update a user prompt preset for a given model scope.
    
    If `name` or `text` is missing, emits a warning and returns the current presets dataframe and refreshed model choices without making changes.
    
    Parameters:
        model_scope (str): Target model identifier or "All Models" to make the preset global.
        name (str): Display name of the preset.
        text (str): Prompt text to save for the preset.
    
    Returns:
        tuple: A pair containing:
            - A 2D list of preset rows formatted for the settings UI dataframe.
            - A Gradio update object for the model choices (includes "All Models" plus eligible models).
    """
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
    """
    Delete a saved user prompt preset for the specified model scope.
    
    Parameters:
        model_scope (str): Target model identifier or "All Models" to indicate a global preset.
        name (str): Name of the preset to delete.
    
    Returns:
        list[list]: Updated presets table as a 2D list suitable for the settings UI.
    
    Side effects:
        Persists the updated presets to the user's configuration and emits UI warnings or info messages.
    """
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