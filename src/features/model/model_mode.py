"""
Model Mode Feature

Generic dropdown for selecting operating mode in models that support multiple modes.
Choices are defined per-model in the YAML config under 'model_modes'.
"""

from ..base import BaseFeature, FeatureConfig


class ModelModeFeature(BaseFeature):
    """
    Generic model operating mode selector.
    
    The available modes are defined in each model's YAML config under 'model_modes'.
    This allows different models to have different mode options.
    
    Example YAML config:
        model_modes:
          - Caption
          - Query
        defaults:
          model_mode: "Caption"
    """
    
    @property
    def config(self) -> FeatureConfig:
        return FeatureConfig(
            name="model_mode",
            default_value=None,
            description="Select the operating mode for the model",
            gui_type="dropdown",
            gui_label="Mode",
            gui_info="Select the operating mode for the model"
        )
    
    def validate(self, value) -> str:
        if value is None:
            return None
        return str(value)
    
    def get_gui_config(self) -> dict:
        return {
            'type': 'dropdown',
            'label': self.config.gui_label,
            'info': self.config.gui_info,
            'value': None,
            'choices': []
        }
