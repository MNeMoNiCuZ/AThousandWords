from ..base import BaseFeature, FeatureConfig

class MinVisualTokensFeature(BaseFeature):
    @property
    def config(self) -> FeatureConfig:
        return FeatureConfig(
            name="min_visual_tokens",
            default_value=256,
            min_value=64,
            max_value=1280,
            gui_type="slider",
            gui_label="Min Visual Tokens",
            gui_info="Minimum number of visual tokens for image processing.",
            description="Minimum number of tokens for image encoder."
        )

class MaxVisualTokensFeature(BaseFeature):
    @property
    def config(self) -> FeatureConfig:
        return FeatureConfig(
            name="max_visual_tokens",
            default_value=1280,
            min_value=256,
            max_value=4096,
            gui_type="slider",
            gui_label="Max Visual Tokens",
            gui_info="Maximum number of visual tokens for image processing.",
            description="Maximum number of tokens for image encoder."
        )

class MinVideoTokensFeature(BaseFeature):
    @property
    def config(self) -> FeatureConfig:
        return FeatureConfig(
            name="min_video_tokens",
            default_value=256,
            min_value=64,
            max_value=1280,
            gui_type="slider",
            gui_label="Min Video Tokens",
            gui_info="Minimum number of visual tokens for video processing.",
            description="Minimum number of tokens for video encoder."
        )

class MaxVideoTokensFeature(BaseFeature):
    @property
    def config(self) -> FeatureConfig:
        return FeatureConfig(
            name="max_video_tokens",
            default_value=16384,
            min_value=1024,
            max_value=32768,
            gui_type="slider",
            gui_label="Max Video Tokens",
            gui_info="Maximum number of visual tokens for video processing.",
            description="Maximum number of tokens for video encoder."
        )
