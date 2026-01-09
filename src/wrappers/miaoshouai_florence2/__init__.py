# Local Florence2 model files for MiaoshouAI compatibility with transformers >= 4.51.0
# This package contains patched versions of the Florence2 model that properly inherit from GenerationMixin
# Based on: https://github.com/miaoshouai/ComfyUI-Miaoshouai-Tagger/pull/62

from .modeling_florence2 import Florence2ForConditionalGeneration
from .configuration_florence2 import Florence2Config, Florence2VisionConfig, Florence2LanguageConfig

__all__ = [
    "Florence2ForConditionalGeneration",
    "Florence2Config", 
    "Florence2VisionConfig",
    "Florence2LanguageConfig"
]
