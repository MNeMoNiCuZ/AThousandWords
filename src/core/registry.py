import importlib
import logging
from typing import Dict, Type
from .config import ConfigManager
from src.wrappers.base import BaseCaptionModel

class ModelRegistry:
    @staticmethod
    def load_wrapper(model_id: str) -> BaseCaptionModel:
        config_mgr = ConfigManager()
        model_config = config_mgr.get_model_config(model_id)
        
        wrapper_path = model_config.get("wrapper_path")
        if not wrapper_path:
            raise ValueError(f"No wrapper_path defined for model {model_id}")
            
        try:
            module_name, class_name = wrapper_path.rsplit(".", 1)
            module = importlib.import_module(module_name)
            wrapper_class = getattr(module, class_name)
            return wrapper_class(model_config)
        except Exception as e:
            logging.error(f"Failed to load wrapper for {model_id}: {e}")
            raise
