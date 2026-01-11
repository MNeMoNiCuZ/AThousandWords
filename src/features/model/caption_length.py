"""
Caption Length Feature

Controls the length of generated captions for models that support it.
"""

from ..base import BaseFeature, FeatureConfig
from typing import Any


class CaptionLengthFeature(BaseFeature):
    """
    Caption length selector for models that support variable caption lengths.
    
    The available lengths are defined in each model's YAML config under 'caption_lengths'.
    This allows different models to have different length options (e.g. Short/Normal/Long vs Brief/Detailed).
    
    Example YAML config:
        caption_lengths:
          - Short
          - Normal
          - Long
        defaults:
          caption_length: "Normal"
    """
    
    @property
    def config(self) -> FeatureConfig:
        return FeatureConfig(
            name="caption_length",
            default_value=None,
            description="Controls the length of generated captions",
            gui_type="dropdown",
            gui_label="Caption Length (Caption Mode Only)",
            gui_info="Select caption verbosity. Options depend on the model.",
            gui_choices=[]  # Populated from model configuration
        )
    
    def validate(self, value: Any) -> str:
        """
        Validate caption length.
        
        Since choices are model-specific, we accept any string here.
        The GUI will restrict choices based on the model config.
        """
        if value is None:
            return None
        return str(value)
    
    def get_gui_config(self) -> dict:
        return {
            'type': 'dropdown',
            'label': self.config.gui_label,
            'info': self.config.gui_info,
            'value': None,
            'choices': []  # Populated dynamically
        }
