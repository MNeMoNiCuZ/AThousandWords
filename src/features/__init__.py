"""
Feature Registry

Central registry for all features.
Import features from here to use them in the application.
"""

# Base
from .base import BaseFeature, FeatureConfig



# Core Configuration Features
from .core.batch_size import BatchSizeFeature
from .core.max_tokens import MaxTokensFeature
from .core.output_options import OverwriteFeature, RecursiveFeature, OutputFormatFeature
from .core.prefix_suffix import PrefixFeature, SuffixFeature
from .core.console_logging import PrintConsoleFeature, PrintStatusFeature
from .core.prompts import TaskPromptFeature, SystemPromptFeature

# Core Text Processing Features
from .core.text_cleanup import CleanTextFeature, CollapseNewlinesFeature
from .core.normalize_text import NormalizeTextFeature
from .core.remove_chinese import RemoveChineseFeature
from .core.strip_loop import StripLoopFeature

# Core Image Processing Features
from .core.image_resize import MaxWidthFeature, MaxHeightFeature, resize_image_proportionally


# Model Generation Features
from .model.temperature import TemperatureFeature
from .model.top_k import TopKFeature
from .model.repetition_penalty import RepetitionPenaltyFeature
from .model.model_version import ModelVersionFeature
from .model.fps import FpsFeature
from .model.flash_attention import FlashAttentionFeature
from .model.caption_length import CaptionLengthFeature

# Model Mode Features
from .model.model_mode import ModelModeFeature
# ReasoningFeature removed - merged into model_mode

# Thinking/Reasoning Features
from .model.thinking_tags import StripThinkingTagsFeature

# Text Processing Features (Model Specific)
from .model.strip_contents import StripContentsInsideFeature
from .model.max_word_length import MaximumWordLengthFeature
from .model.output_json import OutputJsonFeature

# Prompt Features (Model Specific)
from .model.instruction_templates import InstructionTemplatesFeature
from .model.custom_prompt_sources import (
    PromptSourceFeature,
    PromptFileExtensionFeature,
    PromptPrefixFeature,
    PromptSuffixFeature,
    get_custom_prompt_for_image,
    get_prompt_for_image,
    _extract_metadata_from_file
)

# Image Processing Features (Model Specific)
from .model.visual_tokens import MinVisualTokensFeature, MaxVisualTokensFeature, MinVideoTokensFeature, MaxVideoTokensFeature

# Tagger Features
from .model.threshold import ThresholdFeature
from .model.image_size import ModelImageSizeFeature


FEATURE_REGISTRY = {
    # Core Generation
    "temperature": TemperatureFeature(),
    "top_k": TopKFeature(),
    "max_tokens": MaxTokensFeature(),
    "repetition_penalty": RepetitionPenaltyFeature(),
    "batch_size": BatchSizeFeature(),
    "model_version": ModelVersionFeature(),
    "fps": FpsFeature(),
    "flash_attention": FlashAttentionFeature(),
    
    # Thinking
    "strip_thinking_tags": StripThinkingTagsFeature(),
    
    "clean_text": CleanTextFeature(),
    "collapse_newlines": CollapseNewlinesFeature(),
    "normalize_text": NormalizeTextFeature(),
    "remove_chinese": RemoveChineseFeature(),
    "strip_loop": StripLoopFeature(),
    
    # Output
    "overwrite": OverwriteFeature(),
    "recursive": RecursiveFeature(),
    "output_format": OutputFormatFeature(),
    "output_json": OutputJsonFeature(),
    "prefix": PrefixFeature(),
    "suffix": SuffixFeature(),
    "print_console": PrintConsoleFeature(),
    "print_status": PrintStatusFeature(),
    
    # Prompts
    "task_prompt": TaskPromptFeature(),
    "system_prompt": SystemPromptFeature(),
    "instruction_template": InstructionTemplatesFeature(),
    "prompt_source": PromptSourceFeature(),
    "prompt_file_extension": PromptFileExtensionFeature(),
    "prompt_prefix": PromptPrefixFeature(),
    "prompt_suffix": PromptSuffixFeature(),
    
    # Image Processing
    "max_width": MaxWidthFeature(),
    "max_height": MaxHeightFeature(),
    "min_visual_tokens": MinVisualTokensFeature(),
    "max_visual_tokens": MaxVisualTokensFeature(),
    "min_video_tokens": MinVideoTokensFeature(),
    "max_video_tokens": MaxVideoTokensFeature(),
    "max_video_tokens": MaxVideoTokensFeature(),
    
    # Advanced Text Cleanup (Paligemma)
    "strip_contents_inside": StripContentsInsideFeature(),
    "max_word_length": MaximumWordLengthFeature(),

    # Tagger Features
    "threshold": ThresholdFeature(),
    "image_size": ModelImageSizeFeature(),
    
    # Model Mode Features
    "model_mode": ModelModeFeature(),
    "caption_length": CaptionLengthFeature(),
}


def get_feature(name: str) -> BaseFeature:
    """
    Get a feature instance by name.
    
    Args:
        name: Feature name (e.g., "temperature")
        
    Returns:
        Feature instance or None if not found
    """
    return FEATURE_REGISTRY.get(name)


def get_all_features():
    """
    Get all registered features.
    
    Returns:
        Dictionary of all features
    """
    return FEATURE_REGISTRY.copy()


def validate_args(args: dict) -> dict:
    """
    Validate all arguments using their corresponding features.
    
    Args:
        args: Dictionary of argument name -> value
        
    Returns:
        Dictionary with validated values
    """
    validated = {}
    for name, value in args.items():
        feature = get_feature(name)
        if feature:
            validated[name] = feature.validate(value)
        else:
            validated[name] = value
    return validated


def get_defaults_for_features(feature_names: list) -> dict:
    """
    Get default values for a list of feature names.
    
    Args:
        feature_names: List of feature names
        
    Returns:
        Dictionary of feature name -> default value
    """
    defaults = {}
    for name in feature_names:
        feature = get_feature(name)
        if feature:
            defaults[name] = feature.get_default()
    return defaults
