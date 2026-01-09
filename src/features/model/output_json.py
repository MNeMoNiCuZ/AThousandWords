"""
Output JSON Feature

Controls whether to output captions as structured JSON.
"""

from ..base import BaseFeature, FeatureConfig
import json


class OutputJsonFeature(BaseFeature):
    """
    Output captions as JSON with thinking and caption keys.
    """
    
    @property
    def config(self) -> FeatureConfig:
        return FeatureConfig(
            name="output_json",
            default_value=False,
            description="Output captions as structured JSON.",
            gui_type="checkbox",
            gui_label="Output JSON",
            gui_info="Output as JSON with thinking_text and caption_text keys."
        )
    
    def validate(self, value) -> bool:
        if value is None:
            return self.config.default_value
        return bool(value)
    
    @staticmethod
    def format_output(caption: str, thinking: str = "") -> str:
        """
        Format caption and thinking as JSON.
        
        Args:
            caption: The main caption text
            thinking: Optional thinking/reasoning text
            
        Returns:
            JSON formatted string
        """
        data = {
            "thinking_text": thinking,
            "caption_text": caption
        }
        return json.dumps(data, indent=4, ensure_ascii=False)
