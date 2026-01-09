from ..base import BaseFeature, FeatureConfig
import importlib.util

class FlashAttentionFeature(BaseFeature):
    @property
    def config(self) -> FeatureConfig:
        return FeatureConfig(
            name="flash_attention",
            default_value=False,
            gui_type="checkbox",
            gui_label="Use Flash Attention 2",
            gui_info="Enable Flash Attention 2 for faster generation (requires manual installation of flash-attn package).",
            description="Enable Flash Attention 2 acceleration."
        )

    def get_gui_config(self) -> dict:
        """Return GUI configuration, disabling if package is missing."""
        has_flash_attn = importlib.util.find_spec("flash_attn") is not None
        
        config = {
            'type': 'checkbox',
            'label': self.config.gui_label,
            'info': self.config.gui_info if has_flash_attn else "Flash Attention 2 package not found. Manually install 'flash-attn' to enable.",
            'value': False,
            'interactive': has_flash_attn
        }
        return config
