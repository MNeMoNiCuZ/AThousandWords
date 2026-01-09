"""
Maximum Word Length Feature

Removes words exceeding a certain character length and prunes text intelligently.
"""

from ..base import BaseFeature, FeatureConfig

class MaximumWordLengthFeature(BaseFeature):
    """
    Checks for words longer than 'max_word_length'. 
    If found, prunes the text at the nearest preceding sentence break to avoid partial garbage.
    """
    
    @property
    def config(self) -> FeatureConfig:
        return FeatureConfig(
            name="max_word_length",
            default_value=30,
            description="Max characters per word before pruning.",
            gui_type="number",
            gui_label="Max Word Length",
            gui_info="Prunes any text after a word exceeds this length (0 to disable)."
        )
    
    @classmethod
    def apply(cls, text: str, **kwargs) -> str:
        """Apply length check logic."""
        # Get limit from kwargs (feature system passes args) or default
        limit = kwargs.get('max_word_length', 30)
        
        try:
            limit = int(limit)
        except (ValueError, TypeError):
            limit = 30
            
        if limit <= 0:
            return text
            
        words = text.split()
        for word in words:
            if len(word) > limit:
                # Logic: finding closest prune point
                # Find the word position
                word_pos = text.find(word)
                if word_pos == -1: continue # Should not happen
                
                # Look for prune points before this word
                sub_text = text[:word_pos]
                last_period = sub_text.rfind('.')
                last_comma = sub_text.rfind(',')
                
                prune_index = max(last_period, last_comma)
                
                if prune_index != -1:
                    return text[:prune_index].strip()
                else:
                    # No punctuation found, just cut before the word
                    return sub_text.strip()
                    
        return text
