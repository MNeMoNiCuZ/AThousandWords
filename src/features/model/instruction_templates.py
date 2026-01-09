"""
Instruction Templates Feature

Predefined prompt templates for different captioning styles.
"""

from ..base import BaseFeature, FeatureConfig
from typing import Dict, List


class InstructionTemplatesFeature(BaseFeature):
    """
    Predefined instruction templates for common captioning styles.
    """
    
    # Template definitions - model-agnostic
    # Template definitions - model-agnostic
    # NOTE: These are now deprecated and empty.
    # Templates are defined in model YAML configs under `instruction_presets`.
    TEMPLATES: Dict[str, str] = {}
    
    @property
    def config(self) -> FeatureConfig:
        return FeatureConfig(
            name="instruction_template",
            default_value="",
            description="Preset template for instructions",
            gui_type="dropdown",
            gui_label="Instruction Template",
            gui_info="A list of preset prompts",
            gui_choices=[],  # Populated dynamically
            include_in_cli=False  # GUI preset selector only, actual prompts are included
        )
    
    def validate(self, value) -> str:
        if value is None:
            return self.config.default_value
        value = str(value)
        if value not in self.TEMPLATES and not self.TEMPLATES:
            # If templates are empty (dynamic mode), accept any string
            return value
        if value not in self.TEMPLATES:
            # Only warn if we actually have templates to check against
            self.log_warning(f"Unknown template: {value}")
            return value
        return value
    
    def get_template_text(self, template_name: str) -> str:
        """
        Get the actual prompt text for a template name.
        
        Args:
            template_name: Name of the template
            
        Returns:
            The prompt text
        """
        return self.TEMPLATES.get(template_name, "")
    
    @classmethod
    def get_available_templates(cls) -> List[str]:
        """Return list of available template names."""
        return list(cls.TEMPLATES.keys())
    
    @classmethod
    def add_custom_template(cls, name: str, prompt: str):
        """
        Add a custom template at runtime.
        
        Args:
            name: Template name
            prompt: The prompt text
        """
        cls.TEMPLATES[name] = prompt
