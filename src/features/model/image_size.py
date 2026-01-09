"""
Model Image Size Feature

Controls the internal resolution used by the model for inference.
"""

from ..base import BaseFeature, FeatureConfig


class ModelImageSizeFeature(BaseFeature):
    """
    Control image resolution for the model.
    """
    
    @property
    def config(self) -> FeatureConfig:
        return FeatureConfig(
            name="image_size",
            default_value=448,
            min_value=64,
            max_value=2048,
            description="Image resolution used for inference.",
            gui_type="number",
            gui_label="Image resolution",
            gui_info="Resizes the images internally, Recommended: 448."
        )
    
    def validate(self, value) -> int:
        """Ensure image_size is a valid positive integer."""
        if value is None:
            return self.config.default_value
            
        try:
            value = int(value)
        except (TypeError, ValueError):
            self.log_warning(f"Invalid image_size value: {value}, using default")
            return self.config.default_value
            
        if value < 64:
            self.log_warning(f"image_size must be >= 64, was {value}, using 64")
            return 64
            
        return super().validate(value)
