"""
Strip Contents Inside Feature

Removes content inside brackets/parentheses and cleans up resulting whitespace.
Extracted from Paligemma2Wrapper.
"""

from ..base import BaseFeature, FeatureConfig
import re

class StripContentsInsideFeature(BaseFeature):
    """
    Strips content inside specified brackets.
    
    Bracket types are defined per-model in YAML under 'strip_bracket_types'.
    Default: ["(", "[", "{"] if not specified.
    """
    
    @property
    def config(self) -> FeatureConfig:
        return FeatureConfig(
            name="strip_contents_inside",
            default_value=True,
            description="Strip content inside configured brackets (e.g. (), [], {})",
            gui_type="checkbox",
            gui_label="Strip Brackets Content",
            gui_info="Removes text inside brackets."
        )
    
    @classmethod
    def apply(cls, text: str, **kwargs) -> str:
        """Apply stripping logic."""
        if not text:
            return text
            
        # Get bracket types from config, default to common brackets
        # Use `or` to also handle explicit None values
        bracket_types = kwargs.get('strip_bracket_types') or ['(', '[', '{']
        
        for char in bracket_types:
            if char == "(":
                text = re.sub(r'\([^)]*\)', ' ', text)
            elif char == "[":
                text = re.sub(r'\[[^\]]*\]', ' ', text)
            elif char == "{":
                text = re.sub(r'\{[^}]*\}', ' ', text)
            elif char == "<":
                text = re.sub(r'<[^>]*>', ' ', text)
        
        # Cleanup resulting whitespace and punctuation
        text = re.sub(r'\s{2,}', ' ', text)
        text = re.sub(r'\s([,.!?;])', r'\1', text)
        text = re.sub(r'([,.!?;])\s', r'\1 ', text)
        
        return text.strip()
