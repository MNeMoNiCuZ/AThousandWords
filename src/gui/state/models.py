"""
Model selection and ordering state.

Handles model selection, enabled models, and ordering.
"""

import logging
from typing import List, Optional, Tuple

logger = logging.getLogger("GUI.ModelManager")


class ModelManager:
    """Manages model selection and ordering."""
    
    def __init__(self, config_mgr, registry):
        self.config = config_mgr
        self.registry = registry
        
        self.all_models = config_mgr.list_models()
        self.enabled_models = config_mgr.get_enabled_models()
        
        last_model = config_mgr.user_config.get('last_model')
        
        if last_model and isinstance(last_model, str) and last_model in self.enabled_models:
            self.current_model_id = last_model
        elif self.enabled_models:
            self.current_model_id = self.enabled_models[0]
        else:
            self.current_model_id = ""
    
    def set_model(self, model_id: str):
        """Set current model and save to config."""
        if model_id and model_id in self.enabled_models:
            self.current_model_id = model_id
            self.config.user_config['last_model'] = model_id
            self.config.save_user_config()
    
    def refresh(self) -> Tuple[List[str], List[str]]:
        """Refresh model lists from config.
        
        Returns (all_models, enabled_models).
        """
        self.all_models = self.config.list_models()
        self.enabled_models = self.config.get_enabled_models()
        return self.all_models, self.enabled_models
    
    def get_models_by_media_type(self, media_type: str) -> List[str]:
        """Filter models by media type (Image or Video)."""
        filtered = []
        for model_id in self.enabled_models:
            try:
                config = self.config.get_model_config(model_id)
                model_media_type = config.get('media_type', 'Image')
                
                if isinstance(model_media_type, list):
                    if media_type in model_media_type:
                        filtered.append(model_id)
                else:
                    if model_media_type == media_type or model_media_type == "Both":
                        filtered.append(model_id)
            except Exception:
                continue
        return filtered
    
    def move_up(self, selected_model: str, current_order: List[str]) -> List[str]:
        """Move selected model up in the order list."""
        if not selected_model or selected_model not in current_order:
            return current_order
        
        order_list = list(current_order)
        idx = order_list.index(selected_model)
        
        if idx > 0:
            order_list[idx], order_list[idx - 1] = order_list[idx - 1], order_list[idx]
        
        return order_list
    
    def move_down(self, selected_model: str, current_order: List[str]) -> List[str]:
        """Move selected model down in the order list."""
        if not selected_model or selected_model not in current_order:
            return current_order
        
        order_list = list(current_order)
        idx = order_list.index(selected_model)
        
        if idx < len(order_list) - 1:
            order_list[idx], order_list[idx + 1] = order_list[idx + 1], order_list[idx]
        
        return order_list
    
    def get_model_config(self, model_id: str = None):
        """Get config for specified or current model."""
        mid = model_id or self.current_model_id
        if mid:
            return self.config.get_model_config(mid)
        return {}
    
    @property
    def has_models(self) -> bool:
        """Check if any models are available."""
        return len(self.enabled_models) > 0
