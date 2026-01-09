"""
Prefix/Suffix Feature

Text to add before/after generated captions.
"""

from ..base import BaseFeature, FeatureConfig


class PrefixFeature(BaseFeature):
    """
    Text to prepend to every caption.
    """
    
    @property
    def config(self) -> FeatureConfig:
        return FeatureConfig(
            name="prefix",
            default_value="",
            description="Text added at the beginning of every caption.",
            gui_type="textbox",
            gui_label="Prefix",
            gui_info="Text added at the beginning of every caption."
        )
    
    def validate(self, value) -> str:
        if value is None:
            return self.config.default_value
        return str(value)


class SuffixFeature(BaseFeature):
    """
    Text to append to every caption.
    """
    
    @property
    def config(self) -> FeatureConfig:
        return FeatureConfig(
            name="suffix",
            default_value="",
            description="Text added at the end of every caption.",
            gui_type="textbox",
            gui_label="Suffix",
            gui_info="Text added at the end of every caption."
        )
    
    def validate(self, value) -> str:
        if value is None:
            return self.config.default_value
        return str(value)
