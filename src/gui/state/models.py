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
        """
        Initialize the ModelManager and determine the active model based on saved user preference or available enabled models.
        
        Parameters:
            config_mgr: Configuration manager exposing `list_models()`, `get_enabled_models()`, and a `user_config` mapping used to read the last selected model.
            registry: Model registry object; stored for later lookups and operations.
        
        Behavior:
            - Loads `all_models` via `config_mgr.list_models()` and `enabled_models` via `config_mgr.get_enabled_models()`.
            - Sets `current_model_id` to the `user_config['last_model']` value if it is a string and present in `enabled_models`.
            - If no valid saved model exists, sets `current_model_id` to the first entry of `enabled_models` if any, otherwise to an empty string.
        """
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
        """
        Select the given model as the current model and persist the selection to user config.
        
        If `model_id` is a non-empty string and is present in the manager's enabled models, this updates the manager's current model and writes `last_model` to the user configuration before saving; otherwise no changes are made.
        
        Parameters:
            model_id (str): Identifier of the model to select; must be one of `enabled_models`.
        """
        if model_id and model_id in self.enabled_models:
            self.current_model_id = model_id
            self.config.user_config['last_model'] = model_id
            self.config.save_user_config()
    
    def refresh(self) -> Tuple[List[str], List[str]]:
        """
        Reloads the available and enabled model ID lists from the configuration.
        
        Returns:
            all_models (List[str]): List of all model IDs known to the configuration.
            enabled_models (List[str]): List of model IDs currently enabled.
        """
        self.all_models = self.config.list_models()
        self.enabled_models = self.config.get_enabled_models()
        return self.all_models, self.enabled_models
    
    def get_models_by_media_type(self, media_type: str) -> List[str]:
        """
        Return enabled model IDs that support the specified media type.
        
        Models are selected if their `media_type` config equals the given `media_type`, contains it when `media_type` is a list, or is the string "Both". Models with missing or invalid configs are skipped.
        
        Parameters:
            media_type (str): The media type to filter by (e.g., "Image", "Video").
        
        Returns:
            List[str]: Enabled model IDs that support the specified media type.
        """
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
        """
        Move the selected model one position earlier in the provided order list.
        
        Parameters:
            selected_model (str): Model id to move.
            current_order (List[str]): Current ordering of model ids.
        
        Returns:
            List[str]: New ordering with `selected_model` swapped with the previous item if present and not already first; otherwise returns the original ordering.
        """
        if not selected_model or selected_model not in current_order:
            return current_order
        
        order_list = list(current_order)
        idx = order_list.index(selected_model)
        
        if idx > 0:
            order_list[idx], order_list[idx - 1] = order_list[idx - 1], order_list[idx]
        
        return order_list
    
    def move_down(self, selected_model: str, current_order: List[str]) -> List[str]:
        """
        Move the selected model one position later in the provided order.
        
        Parameters:
            selected_model (str): Model identifier to move.
            current_order (List[str]): Current ordered list of model identifiers.
        
        Returns:
            List[str]: A new list reflecting the order after moving `selected_model` down by one position;
            returns an unchanged copy if `selected_model` is empty, not present, or already last.
        """
        if not selected_model or selected_model not in current_order:
            return current_order
        
        order_list = list(current_order)
        idx = order_list.index(selected_model)
        
        if idx < len(order_list) - 1:
            order_list[idx], order_list[idx + 1] = order_list[idx + 1], order_list[idx]
        
        return order_list
    
    def get_model_config(self, model_id: str = None):
        """
        Return the configuration for the specified model or for the current model if none is provided.
        
        Parameters:
            model_id (str, optional): Model identifier to fetch; if omitted or falsy, the current model ID is used.
        
        Returns:
            dict: The model's configuration mapping, or an empty dict if no model id is available.
        """
        mid = model_id or self.current_model_id
        if mid:
            return self.config.get_model_config(mid)
        return {}
    
    @property
    def has_models(self) -> bool:
        """
        Indicates whether any enabled models exist.
        
        Returns:
            `true` if there is at least one enabled model, `false` otherwise.
        """
        return len(self.enabled_models) > 0