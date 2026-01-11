"""
Output Options Feature

Controls output file handling options.
"""

from ..base import BaseFeature, FeatureConfig


class OverwriteFeature(BaseFeature):
    """
    Control whether to overwrite existing caption files.
    """
    
    @property
    def config(self) -> FeatureConfig:
        return FeatureConfig(
            name="overwrite",
            default_value=True,
            description="Replace existing caption files if they exist.",
            gui_type="checkbox",
            gui_label="Overwrite",
            gui_info="Replace existing caption files if they already exist."
        )
    
    def validate(self, value) -> bool:
        if value is None:
            return self.config.default_value
        return bool(value)


class RecursiveFeature(BaseFeature):
    """
    Control whether to scan subdirectories for images.
    """
    
    @property
    def config(self) -> FeatureConfig:
        return FeatureConfig(
            name="recursive",
            default_value=True,
            description="Scan subdirectories for images.",
            gui_type="checkbox",
            gui_label="Recursive",
            gui_info="Scan subdirectories for images to process."
        )
    
    def validate(self, value) -> bool:
        if value is None:
            return self.config.default_value
        return bool(value)


class OutputFormatFeature(BaseFeature):
    """
    Control the output file format (extension).
    Accepts any valid file extension string.
    """
    
    @property
    def config(self) -> FeatureConfig:
        return FeatureConfig(
            name="output_format",
            default_value="txt",
            description="File extension for captions (e.g. txt, json, caption).",
            gui_type="textbox",
            gui_label="Output Format",
            gui_info="File extension for the generated captions (without dot)."
        )
    
    def validate(self, value) -> str:
        if value is None:
            return self.config.default_value
        # Strip leading dots and whitespace
        value = str(value).strip().lstrip(".")
        if not value:
            return "txt"
        return value
