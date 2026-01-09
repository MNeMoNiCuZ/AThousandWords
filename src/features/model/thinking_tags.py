"""
Thinking Tags Feature

Controls how thinking/reasoning content is handled in model output.
Only applicable to models that support thinking (e.g., MimoVL).
"""

from ..base import BaseFeature, FeatureConfig
import re


class StripThinkingTagsFeature(BaseFeature):
    """
    Controls whether to strip thinking tags from model output.
    
    When enabled (default): Removes <think>...</think> content from output.
    When disabled: Keeps thinking tags in the output file.
    """
    
    @property
    def config(self) -> FeatureConfig:
        return FeatureConfig(
            name="strip_thinking_tags",
            default_value=True,
            min_value=None,
            max_value=None,
            description="Remove thinking content from output.",
            gui_type="checkbox",
            gui_label="Strip Thinking Tags",
            gui_info="If checked, removes <think>...</think> content. If unchecked, keeps tags in output."
        )
    
    def validate(self, value) -> bool:
        """Ensure value is boolean."""
        if value is None:
            return self.config.default_value
        return bool(value)
    
    @staticmethod
    def apply(text: str) -> str:
        """
        Remove all thinking content from text.
        
        Handles multiple tag patterns:
        - <think>...</think>
        - <thinking>...</thinking>
        - Partial/unclosed tags
        
        Args:
            text: Text that may contain thinking tags
            
        Returns:
            Text with thinking content removed
        """
        if not text:
            return text
        
        original_len = len(text)
        
        # Remove <think>...</think> blocks
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL | re.IGNORECASE)
        
        # Remove <thinking>...</thinking> blocks
        text = re.sub(r'<thinking>.*?</thinking>', '', text, flags=re.DOTALL | re.IGNORECASE)
        
        # Remove any remaining unclosed tags
        text = re.sub(r'</?think(?:ing)?>', '', text, flags=re.IGNORECASE)
        
        # Clean up extra whitespace
        text = re.sub(r'\n+', '\n', text).strip()
        
        return text
    
    @staticmethod
    def extract_thinking_content(text: str) -> tuple:
        """
        Extract thinking content and main content separately.
        
        Args:
            text: Text that may contain thinking tags
            
        Returns:
            Tuple of (thinking_content, main_content)
        """
        if not text:
            return "", ""
            
        thinking = ""
        main = text
        
        # Extract <think>...</think>
        think_match = re.search(r'<think>(.*?)</think>', text, flags=re.DOTALL | re.IGNORECASE)
        if think_match:
            thinking = think_match.group(1).strip()
            main = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL | re.IGNORECASE)
        
        # Also check for <thinking>...</thinking>
        if not thinking:
            thinking_match = re.search(r'<thinking>(.*?)</thinking>', text, flags=re.DOTALL | re.IGNORECASE)
            if thinking_match:
                thinking = thinking_match.group(1).strip()
                main = re.sub(r'<thinking>.*?</thinking>', '', main, flags=re.DOTALL | re.IGNORECASE)
        
        return thinking, main.strip()
