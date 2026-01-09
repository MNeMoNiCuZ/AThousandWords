# gui/constants.py
"""
Constants, defaults, and utility functions for the GUI.
GLOBAL_DEFAULTS is loaded from configs/global.yaml - the single source of truth.
"""

import yaml
from pathlib import Path

# Load GLOBAL_DEFAULTS from config/global.yaml - THE SINGLE SOURCE OF TRUTH
_GLOBAL_CONFIG_PATH = Path(__file__).parent.parent / "config" / "global.yaml"

def _load_global_defaults():
    """Load global defaults from the config file."""
    try:
        with open(_GLOBAL_CONFIG_PATH, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}

GLOBAL_DEFAULTS = _load_global_defaults()


# Constants for auto-save mapping (UI Component Key -> config key)
UI_CONFIG_MAP = {
    "prefix": "prefix",
    "suffix": "suffix",
    "output_dir": "output_dir",
    "output_format": "output_format",
    "overwrite": "overwrite",
    "recursive": "recursive",
    "print_console": "print_console",
    "unload_model": "unload_model",
    "temperature": "last_temperature",
    "top_k": "last_top_k",
    "max_tokens": "last_max_tokens",
    "repetition_penalty": "last_repetition_penalty",
    "system_prompt": "last_system_prompt",
    "task_prompt": "last_task_prompt"
}


def filter_user_overrides(config: dict) -> dict:
    """Returns only the config values that differ from global defaults."""
    overrides = {}
    from src.gui.app import logger  # Delayed import to avoid circular dependency
    
    for key, value in config.items():
        # Always keep these special keys
        if key in ('disabled_models', 'last_model', 'multi_model', 'model_settings'):
            overrides[key] = value
            continue
        # Skip keys starting with 'last_' (model-specific session data)
        if key.startswith('last_'):
            overrides[key] = value
            continue
            
        default = GLOBAL_DEFAULTS.get(key)
        
        # Robust comparison for diffing
        is_different = False
        
        if default is None:
            # If default doesn't exist, it's an override (or new setting)
            is_different = True
        elif value != default:
            # Check for float vs int equality (e.g. 24.0 == 24)
            if isinstance(value, (int, float)) and isinstance(default, (int, float)):
                if abs(value - default) < 0.0001:
                    is_different = False
                else:
                    is_different = True
            else:
                is_different = True
        
        if is_different:
            overrides[key] = value
        
    return overrides
