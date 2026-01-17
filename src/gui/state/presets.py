"""
User preset management.

Handles saving, loading, and deleting user prompt presets.
"""

import logging
import gradio as gr
from typing import List, Dict, Any, Optional

logger = logging.getLogger("GUI.PresetManager")


class PresetManager:
    """Manages user prompt presets."""
    
    def __init__(self, config_mgr):
        """
        Initialize the PresetManager with a configuration manager used to access and persist model and user preset data.
        
        Parameters:
            config_mgr: An object that provides model and user configuration access and persistence methods (expected to implement at least `list_models()`, `get_model_config(model_id)`, a `user_config` dict, and `save_user_config()`).
        """
        self.config = config_mgr
    
    def get_eligible_models(self) -> List[str]:
        """
        List models that support custom prompts.
        
        Returns:
            eligible (List[str]): Model IDs that expose prompt presets.
        """
        eligible = []
        for model_id in self.config.list_models():
            try:
                config = self.config.get_model_config(model_id)
                if 'prompt_presets' in config:
                    eligible.append(model_id)
            except Exception:
                continue
        return eligible
    
    def get_presets_for_model(self, model_id: str) -> Dict[str, str]:
        """
        Retrieve merged prompt presets for a given model.
        
        Merges base presets from the model configuration, global user presets under the "all" scope, and user presets specific to the given model. When keys conflict, model-specific presets take precedence over global presets, which take precedence over base model presets.
        
        Returns:
            Dict[str, str]: Mapping of preset name to prompt text.
        """
        base_presets = {}
        
        config = self.config.get_model_config(model_id)
        if 'prompt_presets' in config:
            base_presets = dict(config['prompt_presets'])
        
        user_presets = self.config.user_config.get('user_presets', {})
        all_presets = user_presets.get('all', {})
        model_presets = user_presets.get(model_id, {})
        
        merged = {**base_presets, **all_presets, **model_presets}
        return merged
    
    def save_preset(self, model_scope: str, name: str, text: str) -> bool:
        """
        Save a user prompt preset under a model scope.
        
        Parameters:
            model_scope (str): Model ID to scope the preset, or "all" to make it global.
            name (str): Preset name; must not be empty or only whitespace.
            text (str): Prompt text; must not be empty or only whitespace.
        
        Returns:
            bool: `True` if the preset was saved successfully, `False` otherwise (e.g., when name or text is empty).
        """
        if not name or not name.strip():
            gr.Warning("Preset name cannot be empty")
            return False
        
        if not text or not text.strip():
            gr.Warning("Prompt text cannot be empty")
            return False
        
        name = name.strip()
        text = text.strip()
        scope = model_scope.strip() if model_scope else "all"
        
        if 'user_presets' not in self.config.user_config:
            self.config.user_config['user_presets'] = {}
        
        if scope not in self.config.user_config['user_presets']:
            self.config.user_config['user_presets'][scope] = {}
        
        self.config.user_config['user_presets'][scope][name] = text
        self.config.save_user_config()
        
        gr.Info(f"Preset '{name}' saved for {scope}")
        return True
    
    def delete_preset(self, model_scope: str, name: str) -> bool:
        """
        Delete a user preset for a given scope.
        
        Parameters:
            model_scope (str): Model identifier to target; when falsy, the `"all"` scope is used.
            name (str): Name of the preset to delete.
        
        Returns:
            bool: `True` if the preset existed and was removed, `False` otherwise.
        """
        if not name:
            return False
        
        scope = model_scope if model_scope else "all"
        user_presets = self.config.user_config.get('user_presets', {})
        
        if scope in user_presets and name in user_presets[scope]:
            del user_presets[scope][name]
            
            if not user_presets[scope]:
                del user_presets[scope]
            
            self.config.save_user_config()
            gr.Info(f"Preset '{name}' deleted")
            return True
        
        gr.Warning(f"Preset '{name}' not found for {scope}")
        return False
    
    def get_presets_dataframe(self) -> List[List[str]]:
        """
        Format user presets into rows suitable for a dataframe view.
        
        Returns:
            rows (List[List[str]]): List of rows where each row is
                [Scope Display, Preset Name, Prompt Text, Delete Symbol].
                Scope Display is "All Models" for the global scope ("all"), otherwise the scope value.
        """
        rows = []
        user_presets = self.config.user_config.get('user_presets', {})
        
        for scope, presets in user_presets.items():
            for name, text in presets.items():
                display_scope = "All Models" if scope == "all" else scope
                rows.append([display_scope, name, text, "🗑️"])
        
        rows.sort(key=lambda r: (r[0] != "All Models", r[0], r[1].lower()))
        return rows