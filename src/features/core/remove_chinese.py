"""
Remove Chinese Feature

Strips Chinese/CJK characters from output text.
"""

from ..base import BaseFeature, FeatureConfig
import re


class RemoveChineseFeature(BaseFeature):
    """
    Remove Chinese/CJK characters from text.
    
    Useful when models generate unwanted characters in different languages.
    """
    
    @property
    def config(self) -> FeatureConfig:
        return FeatureConfig(
            name="remove_chinese",
            default_value=False,
            description="Strip Chinese/CJK characters from output.",
            gui_type="checkbox",
            gui_label="Remove Chinese",
            gui_info="Strip Chinese/CJK characters from generated captions."
        )
    
    def validate(self, value) -> bool:
        if value is None:
            return self.config.default_value
        return bool(value)
    
    @staticmethod
    def apply(text: str, **kwargs) -> str:
        """
        Remove Chinese/CJK characters from text.
        
        Removes:
        - CJK Unified Ideographs (U+4E00-U+9FFF)
        - CJK Extension A-G
        
        Args:
            text: Input text
            
        Returns:
            Text with Chinese characters removed
        """
        if not text:
            return text
        
        # Remove CJK Unified Ideographs (most common Chinese characters)
        text = re.sub(r'[\u4e00-\u9fff]', '', text)
        
        # Remove CJK Extension A
        text = re.sub(r'[\u3400-\u4dbf]', '', text)
        
        # Remove CJK Compatibility Ideographs
        text = re.sub(r'[\uf900-\ufaff]', '', text)
        
        # Clean up any double spaces that may result
        text = re.sub(r' +', ' ', text).strip()
        
        return text
