from ..base import BaseFeature, FeatureConfig

class FpsFeature(BaseFeature):
    @property
    def config(self) -> FeatureConfig:
        return FeatureConfig(
            name="fps",
            default_value=4,
            min_value=1,
            max_value=60,
            gui_type="slider",
            gui_label="Video FPS",
            gui_info="Frames per second to sample from video inputs.",
            description="Control the frame rate for video sampling."
        )
