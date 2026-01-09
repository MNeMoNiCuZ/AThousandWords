"""
Temperature Feature

Controls the randomness/creativity of model generation.
Higher values = more creative, lower values = more deterministic.
"""

from ..base import BaseFeature, FeatureConfig


class TemperatureFeature(BaseFeature):
    """
    Temperature control for text generation.
    
    Must be > 0 for sampling to work properly.
    Most models use values between 0.1 and 1.0.
    """
    
    @property
    def config(self) -> FeatureConfig:
        return FeatureConfig(
            name="temperature",
            default_value=0.7,
            min_value=0.01,
            max_value=2.0,
            description="Controls randomness in generation. Higher = more creative.",
            gui_type="slider",
            gui_label="Temperature",
            gui_info="Model creativitiy, higher: more creative, lower: more deterministic.",
            gui_step=0.05
        )
    
    def validate(self, value) -> float:
        """Ensure temperature is always positive."""
        if value is None:
            return self.config.default_value
            
        try:
            value = float(value)
        except (TypeError, ValueError):
            self.log_warning(f"Invalid temperature value: {value}, using default")
            return self.config.default_value
        
        if value <= 0:
            self.log_warning(f"Temperature must be > 0, was {value}, using 0.01")
            return 0.01
            
        return super().validate(value)
