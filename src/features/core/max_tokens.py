"""
Max Tokens Feature

Controls the maximum length of generated text.
"""

from ..base import BaseFeature, FeatureConfig


class MaxTokensFeature(BaseFeature):
    """
    Maximum tokens control for generation length.
    
    Higher values allow longer captions but use more VRAM and time.
    """
    
    @property
    def config(self) -> FeatureConfig:
        return FeatureConfig(
            name="max_tokens",
            default_value=1024,
            min_value=64,
            max_value=8192,
            description="Maximum number of tokens to generate.",
            gui_type="slider",
            gui_label="Max Tokens",
            gui_info="Maximum length of the generated caption. Higher = longer but slower.",
            gui_step=64
        )
    
    def validate(self, value) -> int:
        """Ensure max_tokens is a positive integer."""
        if value is None:
            return self.config.default_value
            
        try:
            value = int(value)
        except (TypeError, ValueError):
            self.log_warning(f"Invalid max_tokens value: {value}, using default")
            return self.config.default_value
        
        if value < 64:
            self.log_warning(f"max_tokens must be >= 64, was {value}, using 64")
            return 64
            
        return super().validate(value)
