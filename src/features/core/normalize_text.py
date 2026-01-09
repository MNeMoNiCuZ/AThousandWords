"""
Normalize Text Feature

Normalizes Unicode punctuation and removes Markdown formatting.
"""

from ..base import BaseFeature, FeatureConfig
import re


class NormalizeTextFeature(BaseFeature):
    """
    Normalize Unicode punctuation and remove Markdown formatting.
    """
    
    @property
    def config(self) -> FeatureConfig:
        return FeatureConfig(
            name="normalize_text",
            default_value=True,
            description="Normalize Unicode punctuation and remove Markdown.",
            gui_type="checkbox",
            gui_label="Normalize Text",
            gui_info="Normalize Unicode punctuation and remove Markdown formatting."
        )
    
    def validate(self, value) -> bool:
        if value is None:
            return self.config.default_value
        return bool(value)
    
    # Unicode replacement map
    UNICODE_REPLACEMENTS = {
        '\u2014': ', ',   # Em dash (—) to comma and space
        '\u2013': '-',    # En dash (–) to hyphen
        '\u2018': "'",    # Left single quote (') to apostrophe
        '\u2019': "'",    # Right single quote (') to apostrophe
        '\u201c': '"',    # Left double quote (") to straight quote
        '\u201d': '"',    # Right double quote (") to straight quote
        '\u2026': '...',  # Ellipsis (…) to three dots
        '\u3002': '.',    # Ideographic full stop (。) to period
        '\uff1a': ':',    # Full-width colon (：) to colon
        '\u3001': ',',    # Ideographic comma (、) to comma
        '\uff1b': ';',    # Full-width semicolon (；) to semicolon
    }
    
    @classmethod
    def apply(cls, text: str, **kwargs) -> str:
        """Normalize text by replacing Unicode characters and removing Markdown."""
        if not text:
            return text
        
        # Count and apply Unicode replacements
        for old, new in cls.UNICODE_REPLACEMENTS.items():
            text = text.replace(old, new)
        
        # Count and remove Markdown formatting
        markdown_patterns = [
            (r'\*\*(.*?)\*\*', r'\1', 'bold'),      # Bold **text**
            (r'\*(.*?)\*', r'\1', 'italic'),         # Italic *text*
            (r'_(.*?)_', r'\1', 'underscore'),       # Italic _text_
            (r'`(.*?)`', r'\1', 'code'),             # Inline code `code`
        ]
        
        for pattern, repl, name in markdown_patterns:
            text = re.sub(pattern, repl, text)
        
        # Headers, links, blockquotes, lists
        text = re.sub(r'^#{1,6}\s*(.*?)$', r'\1', text, flags=re.MULTILINE)
        text = re.sub(r'\[(.*?)\]\(.*?\)', r'\1', text)
        text = re.sub(r'^>\s*(.*?)$', r'\1', text, flags=re.MULTILINE)
        text = re.sub(r'^[-*]\s*(.*?)$', r'\1', text, flags=re.MULTILINE)
        text = re.sub(r'^\d+\.\s*(.*?)$', r'\1', text, flags=re.MULTILINE)
        
        # Normalize dashes
        text = re.sub(r'[–—]', '-', text)
        
        # Collapse multiple spaces/tabs within lines, but PRESERVE newlines
        # (CollapseNewlinesFeature handles the vertical structure)
        text = re.sub(r'[ \t]+', ' ', text)
        
        return text.strip()
