"""
Repetition Penalty Feature

Controls how much the model avoids repeating words/phrases.
"""

from ..base import BaseFeature, FeatureConfig


class RepetitionPenaltyFeature(BaseFeature):
    """
    Repetition penalty control.
    
    Values > 1.0 discourage repetition.
    1.0 = no penalty, higher = more penalty.
    """
    
    @property
    def config(self) -> FeatureConfig:
        return FeatureConfig(
            name="repetition_penalty",
            default_value=1.2,
            min_value=1.0,
            max_value=2.0,
            description="Penalty for repeating words. Higher = less repetition.",
            gui_type="slider",
            gui_label="Repetition Penalty",
            gui_info="Discourage the model from repeating words. 1.0 = no penalty.",
            gui_step=0.05
        )
    
    def validate(self, value) -> float:
        """Ensure repetition_penalty is >= 1.0."""
        if value is None:
            return self.config.default_value
            
        try:
            value = float(value)
        except (TypeError, ValueError):
            self.log_warning(f"Invalid repetition_penalty value: {value}, using default")
            return self.config.default_value
        
        if value < 1.0:
            self.log_warning(f"repetition_penalty must be >= 1.0, was {value}, using 1.0")
            return 1.0
            
        return super().validate(value)
