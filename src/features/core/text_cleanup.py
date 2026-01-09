"""
Text Cleanup Feature

Provides text normalization and cleanup utilities.
"""

from ..base import BaseFeature, FeatureConfig
import re


class CleanTextFeature(BaseFeature):
    """
    Clean text by stripping extra spaces and normalizing whitespace.
    """
    
    @property
    def config(self) -> FeatureConfig:
        return FeatureConfig(
            name="clean_text",
            default_value=True,
            description="Strip extra spaces and clean up text.",
            gui_type="checkbox",
            gui_label="Clean Text",
            gui_info="Strip extra spaces and clean up text formatting."
        )
    
    def validate(self, value) -> bool:
        if value is None:
            return self.config.default_value
        return bool(value)
    
    @staticmethod
    def apply(text: str, **kwargs) -> str:
        """Clean text by collapsing multiple spaces."""
        if not text:
            return text
        
        text = re.sub(r' +', ' ', text).strip()
        
        return text


class CollapseNewlinesFeature(BaseFeature):
    """
    Replace newlines with ". " for paragraph-style output.
    """
    
    @property
    def config(self) -> FeatureConfig:
        return FeatureConfig(
            name="collapse_newlines",
            default_value=True,
            description="Replace newlines with '. ' for paragraph style.",
            gui_type="checkbox",
            gui_label="Collapse Newlines",
            gui_info="Replace newlines with '. ' for paragraph-style captions."
        )
    
    def validate(self, value) -> bool:
        if value is None:
            return self.config.default_value
        return bool(value)
    
    @staticmethod
    def apply(text: str, **kwargs) -> str:
        """Collapse newlines into periods."""
        if not text:
            return text
        
        text = re.sub(r'\n+', '. ', text)
        text = re.sub(r'\.+\s*\.', '.', text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
