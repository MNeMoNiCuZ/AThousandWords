"""
Console Logging Feature

Controls what gets printed to the console during processing.
"""

from ..base import BaseFeature, FeatureConfig


class PrintConsoleFeature(BaseFeature):
    """
    Print generated captions to console.
    """
    
    @property
    def config(self) -> FeatureConfig:
        return FeatureConfig(
            name="print_console",
            default_value=True,
            description="Print captions to console during processing",
            gui_type="checkbox",
            gui_label="Print to Console",
            gui_info="Display captioning details in the console.",
            include_in_cli=False  # GUI-only, doesn't affect actual processing
        )
    
    def validate(self, value) -> bool:
        if value is None:
            return self.config.default_value
        return bool(value)


class PrintStatusFeature(BaseFeature):
    """
    Print status messages during processing.
    """
    
    @property
    def config(self) -> FeatureConfig:
        return FeatureConfig(
            name="print_status",
            default_value=True,
            description="Print processing status messages.",
            gui_type="checkbox",
            gui_label="Print Status",
            gui_info="Print status messages when saving captions.",
            include_in_cli=False  # GUI-only, not a valid CLI argument
        )
    
    def validate(self, value) -> bool:
        if value is None:
            return self.config.default_value
        return bool(value)
