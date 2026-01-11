"""
Threshold Feature

Controls the confidence threshold for tag generation.
"""

from ..base import BaseFeature, FeatureConfig


class ThresholdFeature(BaseFeature):
    """
    Confidence threshold control for tagging models.
    
    Higher values result in fewer, higher-confidence tags.
    Lower values include more tags but may increase false positives.
    """
    
    @property
    def config(self) -> FeatureConfig:
        return FeatureConfig(
            name="threshold",
            default_value=0.35,
            min_value=0.0,
            max_value=1.0,
            description="Probability threshold for tag selection.",
            gui_type="slider",
            gui_label="Threshold",
            gui_info="Internal probability cutoff for candidate tokens.",
            gui_step=0.01
        )
    
    def validate(self, value) -> float:
        """Ensure threshold is a float between 0.0 and 1.0."""
        if value is None:
            return self.config.default_value
            
        try:
            value = float(value)
        except (TypeError, ValueError):
            self.log_warning(f"Invalid threshold value: {value}, using default")
            return self.config.default_value
            
        # Base class handles min/max validation
        return super().validate(value)
