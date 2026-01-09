"""
Batch Size Feature

Controls how many images are processed in parallel.
"""

from ..base import BaseFeature, FeatureConfig


class BatchSizeFeature(BaseFeature):
    """
    Batch size control for parallel processing.
    
    Higher values process faster but use more VRAM.
    VRAM recommendations:
    - 8GB: Batch 1
    - 12GB: Batch 1-2
    - 16GB: Batch 2-4
    - 24GB: Batch 4-8
    """
    
    @property
    def config(self) -> FeatureConfig:
        return FeatureConfig(
            name="batch_size",
            default_value=1,
            min_value=1,
            max_value=None,
            description="Number of images to process in parallel.",
            gui_type="number",
            gui_label="Batch Size",
            gui_info="Number of images to process in parallel. Higher = faster but more VRAM."
        )
    
    def validate(self, value) -> int:
        """Ensure batch_size is a positive integer."""
        if value is None:
            return self.config.default_value
            
        try:
            value = int(value)
        except (TypeError, ValueError):
            self.log_warning(f"Invalid batch_size value: {value}, using default")
            return self.config.default_value
        
        if value < 1:
            self.log_warning(f"batch_size must be >= 1, was {value}, using 1")
            return 1
            
        return super().validate(value)
    
    def get_recommended_batch_size(self, vram_gb: int) -> int:
        """
        Get recommended batch size based on available VRAM.
        
        Args:
            vram_gb: Available VRAM in gigabytes
            
        Returns:
            Recommended batch size
        """
        if vram_gb >= 24:
            return 4
        elif vram_gb >= 16:
            return 2
        elif vram_gb >= 12:
            return 1
        else:
            return 1
