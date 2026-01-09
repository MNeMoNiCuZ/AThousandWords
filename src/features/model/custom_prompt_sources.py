"""
Custom Prompt Sources Feature

Unified prompt source system supporting:
- Instruction Template mode (uses instruction_template or task_prompt)
- From File mode (reads from per-image files with custom extension)
- From Metadata mode (extracts from PNG metadata)
"""

from ..base import BaseFeature, FeatureConfig
from pathlib import Path
from PIL import Image
import os


class PromptSourceFeature(BaseFeature):
    """
    Unified prompt source selection dropdown.
    Replaces the old use_custom_prompts and use_metadata_prompts boolean flags.
    """
    
    @property
    def config(self) -> FeatureConfig:
        return FeatureConfig(
            name="prompt_source",
            default_value="Instruction Template",
            description="Choose where to get prompts from.",
            gui_type="dropdown",
            gui_label="Prompt Source",
            gui_info="Select prompt source for the captioning.",
            gui_choices=["Instruction Template", "From File", "From Metadata"],
            include_in_cli=False  # GUI setting for prompt building, actual prompts are included
        )
    
    def validate(self, value) -> str:
        if value is None:
            return self.config.default_value
        value = str(value)
        if value not in self.config.gui_choices:
            self.log_warning(f"Unknown prompt source: {value}, using default")
            return self.config.default_value
        return value


class PromptFileExtensionFeature(BaseFeature):
    """
    File extension for custom prompt files when using 'From File' mode.
    Renamed from CustomPromptExtensionFeature.
    """
    
    @property
    def config(self) -> FeatureConfig:
        return FeatureConfig(
            name="prompt_file_extension",
            default_value="prompt",
            description="File extension for per-image prompt files.",
            gui_type="textbox",
            gui_label="Prompt File Extension",
            gui_info="File extension for prompt files (e.g., prompt, tag, caption). Only used in 'From File' mode."
        )
    
    def validate(self, value) -> str:
        if value is None:
            value = self.config.default_value
        else:
            value = str(value)
        if not value.startswith('.'):
            value = '.' + value
        return value


class PromptPrefixFeature(BaseFeature):
    """
    Text to prepend to prompts from File or Metadata sources.
    """
    
    @property
    def config(self) -> FeatureConfig:
        return FeatureConfig(
            name="prompt_prefix",
            default_value="",
            description="Text to prepend to prompts loaded from files or metadata.",
            gui_type="textbox",
            gui_label="Prompt Prefix",
            gui_info="Text added before the prompt content. Only used in 'From File' and 'From Metadata' modes."
        )
    
    def validate(self, value) -> str:
        if value is None:
            return self.config.default_value
        return str(value)


class PromptSuffixFeature(BaseFeature):
    """
    Text to append to prompts from File or Metadata sources.
    """
    
    @property
    def config(self) -> FeatureConfig:
        return FeatureConfig(
            name="prompt_suffix",
            default_value="",
            description="Text to append to prompts loaded from files or metadata.",
            gui_type="textbox",
            gui_label="Prompt Suffix",
            gui_info="Text added after the prompt content. Only used in 'From File' and 'From Metadata' modes."
        )
    
    def validate(self, value) -> str:
        if value is None:
            return self.config.default_value
        return str(value)


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_custom_prompt_for_image(image_path: str, extension: str = ".prompt") -> str:
    """
    Get custom prompt from file for an image.
    
    Args:
        image_path: Path to the image file
        extension: Extension for prompt files
        
    Returns:
        Custom prompt text or empty string if not found
    """
    prompt_path = Path(image_path).with_suffix(extension)
    if prompt_path.exists():
        try:
            return prompt_path.read_text(encoding='utf-8').strip()
        except Exception:
            return ""
    return ""


def _extract_metadata_from_file(file_path: str) -> str:
    """
    Extract prompt from PNG metadata.
    
    Implementation from MimoVL (docs/source_captioners/mimovl/batch.py lines 134-161).
    Extracts the positive prompt from PNG 'parameters' metadata.
    
    Args:
        file_path: Path to the image file
        
    Returns:
        Extracted prompt or empty string
    """
    ext = os.path.splitext(file_path)[1].lower()
    if ext != '.png':
        return ""
    
    try:
        with Image.open(file_path) as img:
            info = dict(img.info)
        
        params_str = info.get('parameters', '')
        if not isinstance(params_str, str):
            return ""
        
        # Extract positive prompt (before "Negative prompt:" or "Steps:")
        neg_idx = params_str.find('Negative prompt:')
        steps_idx = params_str.find('Steps:')
        
        if neg_idx != -1:
            return params_str[:neg_idx].strip()
        elif steps_idx != -1:
            return params_str[:steps_idx].strip()
        else:
            return params_str.strip()
            
    except Exception:
        return ""


def get_prompt_for_image(
    image_path: str,
    mode: str,
    prefix: str = "",
    suffix: str = "",
    extension: str = ".prompt",
    instruction_template: str = "",
    task_prompt: str = "",
    instruction_presets: dict = None
) -> str:
    """
    Unified function to get prompt based on selected mode.
    
    Args:
        image_path: Path to the image file
        mode: "Instruction Template", "From File", or "From Metadata"
        prefix: Text to prepend (for File/Metadata modes)
        suffix: Text to append (for File/Metadata modes)
        extension: File extension for prompt files (for From File mode)
        instruction_template: Selected instruction template name
        task_prompt: Fallback task prompt
        instruction_presets: Dict of instruction template presets
        
    Returns:
        Final prompt string
    """
    instruction_presets = instruction_presets or {}
    
    if mode == "From File":
        source_content = get_custom_prompt_for_image(image_path, extension)
        if not source_content:
            # Fallback to task_prompt if file not found
            return task_prompt or "Describe this image."
        return f"{prefix}{source_content}{suffix}"
    
    elif mode == "From Metadata":
        source_content = _extract_metadata_from_file(image_path)
        if not source_content:
            # Fallback to task_prompt if metadata not found
            return task_prompt or "Describe this image."
        return f"{prefix}{source_content}{suffix}"
    
    else:  # "Instruction Template" (default)
        # Use instruction preset if available, otherwise use task_prompt
        if instruction_template and instruction_template in instruction_presets:
            return instruction_presets[instruction_template]
        return task_prompt or "Describe this image."
