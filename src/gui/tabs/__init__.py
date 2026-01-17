"""
Tab modules for the GUI.

Each module provides a factory function that creates tab components.
"""

from .tools import create_tools_tab
from .presets import create_presets_tab
from .settings import create_settings_tab
from .multi_model_tab import create_multi_model_tab, wire_multi_model_events, get_multi_model_reload_handler
from .captioning_tab import create_general_settings_accordion, create_model_settings_accordion, create_control_area, update_prompt_source_visibility, update_models_by_media_type

__all__ = [
    "create_tools_tab",
    "create_presets_tab",
    "create_settings_tab",
    "create_multi_model_tab",
    "wire_multi_model_events",
    "get_multi_model_reload_handler",
    "create_general_settings_accordion",
    "create_model_settings_accordion",
    "create_control_area",
    "update_prompt_source_visibility",
    "update_models_by_media_type",
]

