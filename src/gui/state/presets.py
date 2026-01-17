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
        self.config = config_mgr
    
    def get_eligible_models(self) -> List[str]:
        """Return list of models that support custom prompts."""
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
        """Get all presets for a specific model."""
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
        """Save a user preset.
        
        Args:
            model_scope: Model ID or "all" for global
            name: Preset name
            text: Prompt text
            
        Returns True on success.
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
        """Delete a user preset.
        
        Returns True on success.
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
        """Get all user presets formatted for dataframe display.
        
        Returns list of [Model, Preset Name, Prompt Text, Delete] rows.
        """
        rows = []
        user_presets = self.config.user_config.get('user_presets', {})
        
        for scope, presets in user_presets.items():
            for name, text in presets.items():
                display_scope = "All Models" if scope == "all" else scope
                rows.append([display_scope, name, text, "🗑️"])
        
        rows.sort(key=lambda r: (r[0] != "All Models", r[0], r[1].lower()))
        return rows
