"""
Image Resize Feature

Controls image resizing before inference to manage VRAM usage.
"""

from ..base import BaseFeature, FeatureConfig
from PIL import Image


class MaxWidthFeature(BaseFeature):
    """
    Maximum image width before inference.
    Images wider than this are resized proportionally.
    """
    
    @property
    def config(self) -> FeatureConfig:
        return FeatureConfig(
            name="max_width",
            default_value=None,
            min_value=128,
            max_value=4096,
            description="Maximum image width (None = no limit).",
            gui_type="number",
            gui_label="Max Width",
            gui_info="Maximum image width before inference. Leave empty for no limit."
        )
    
    def validate(self, value):
        if value is None or value == '' or value == 0:
            return None
        try:
            value = int(value)
            if value < 128:
                self.log_warning(f"max_width {value} too small, using 128")
                return 128
            return value
        except (TypeError, ValueError):
            return None


class MaxHeightFeature(BaseFeature):
    """
    Maximum image height before inference.
    Images taller than this are resized proportionally.
    """
    
    @property
    def config(self) -> FeatureConfig:
        return FeatureConfig(
            name="max_height",
            default_value=None,
            min_value=128,
            max_value=4096,
            description="Maximum image height (None = no limit).",
            gui_type="number",
            gui_label="Max Height",
            gui_info="Maximum image height before inference. Leave empty for no limit."
        )
    
    def validate(self, value):
        if value is None or value == '' or value == 0:
            return None
        try:
            value = int(value)
            if value < 128:
                self.log_warning(f"max_height {value} too small, using 128")
                return 128
            return value
        except (TypeError, ValueError):
            return None

def resize_image_proportionally(image: Image.Image, max_width: int = None, max_height: int = None) -> Image.Image:
    """
    Resize an image proportionally to fit within max dimensions.
    
    Args:
        image: PIL Image to resize
        max_width: Maximum width (None = no limit)
        max_height: Maximum height (None = no limit)
        
    Returns:
        Resized PIL Image (or original if no resize needed)
    """
    if max_width is None and max_height is None:
        return image
    
    orig_width, orig_height = image.size
    
    # Calculate scale factors
    width_scale = max_width / orig_width if max_width else float('inf')
    height_scale = max_height / orig_height if max_height else float('inf')
    
    # Use the smaller scale to maintain aspect ratio
    scale = min(width_scale, height_scale)
    
    if scale >= 1.0:
        return image
    
    new_width = int(orig_width * scale)
    new_height = int(orig_height * scale)
    
    resized = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    return resized
