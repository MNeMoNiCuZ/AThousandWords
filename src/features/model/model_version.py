"""
Model Version Feature

Allows selection of model versions within a model family.
Model-specific: only applies to models that have multiple versions (weight files or architectures).
"""

from ..base import BaseFeature, FeatureConfig


class ModelVersionFeature(BaseFeature):
    """
    Select which model version (weight file) to use.
    
    Some models have multiple versions (e.g., SmolVLM has 256M and 500M).
    This feature allows users to choose which version to load.
    """
    
    @property
    def config(self) -> FeatureConfig:
        return FeatureConfig(
            name="model_version",
            default_value=None,  # No default - depends on model
            description="Select model version/architecture for models with multiple options.",
            gui_type="dropdown",
            gui_label="Model Version",
            gui_info="Choose which model version/architecture to use."
        )
    
    def validate(self, value) -> str:
        if value is None:
            return None  # No default - let model handle
        return str(value)
    
    def get_gui_config(self) -> dict:
        """Return GUI configuration for this feature."""
        return {
            'type': 'dropdown',
            'label': self.config.gui_label,
            'info': self.config.gui_info,
            'value': None,
            'choices': []  # Populated by model config
        }
