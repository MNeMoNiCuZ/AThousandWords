"""
Top K Feature

Controls vocabulary filtering during generation.
Limits the model to choosing from the top K most likely tokens.
"""

from ..base import BaseFeature, FeatureConfig


class TopKFeature(BaseFeature):
    """
    Top K sampling control.
    
    Lower values = more focused, higher values = more variety.
    """
    
    @property
    def config(self) -> FeatureConfig:
        return FeatureConfig(
            name="top_k",
            default_value=50,
            min_value=1,
            max_value=100,
            description="Limit vocabulary to top K most likely tokens.",
            gui_type="slider",
            gui_label="Top K",
            gui_info="Limit vocabulary to more likely tokens. Lower: more focused.",
            gui_step=1
        )
    
    def validate(self, value) -> int:
        """Ensure top_k is a positive integer."""
        if value is None:
            return self.config.default_value
            
        try:
            value = int(value)
        except (TypeError, ValueError):
            self.log_warning(f"Invalid top_k value: {value}, using default")
            return self.config.default_value
        
        if value < 1:
            self.log_warning(f"top_k must be >= 1, was {value}, using 1")
            return 1
            
        return super().validate(value)
