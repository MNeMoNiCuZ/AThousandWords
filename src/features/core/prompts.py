"""
Prompts Feature

Task prompt and system prompt for model instructions.
"""

from typing import Dict, Any
from ..base import BaseFeature, FeatureConfig


class TaskPromptFeature(BaseFeature):
    """
    The specific instruction/prompt for the captioning task.
    """
    
    @property
    def config(self) -> FeatureConfig:
        return FeatureConfig(
            name="task_prompt",
            default_value="",
            description="Specific instruction for the captioning task.",
            gui_type="textbox",
            gui_label="Task Prompt",
            gui_info="The instruction sent to the model for each image."
        )

    def get_gui_config(self) -> Dict[str, Any]:
        config = super().get_gui_config()
        config.update({
            "lines": 2,
            "max_lines": 16
        })
        return config
    
    def validate(self, value) -> str:
        if value is None:
            return self.config.default_value
        return str(value)


class SystemPromptFeature(BaseFeature):
    """
    High-level persona/context for the model.
    """
    
    @property
    def config(self) -> FeatureConfig:
        return FeatureConfig(
            name="system_prompt",
            default_value="You are a helpful image captioning assistant.",
            description="High-level model persona and context.",
            gui_type="textbox",
            gui_label="System Prompt",
            gui_info="System instruction for the model."
        )
    
    def validate(self, value) -> str:
        if value is None:
            return self.config.default_value
        return str(value)
