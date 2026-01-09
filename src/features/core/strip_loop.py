"""
Strip Loop Feature

Detects and removes repeating patterns at the end of text output.
Common LLM hallucination where the model loops on a word/phrase.
"""

from ..base import BaseFeature, FeatureConfig
import re


class StripLoopFeature(BaseFeature):
    """
    Remove repeating patterns at the end of text.
    
    Detects patterns like:
    - "tree, tree, tree, tree, tree..."
    - "word word word word..."
    - "phrase. phrase. phrase..."
    """
    
    @property
    def config(self) -> FeatureConfig:
        return FeatureConfig(
            name="strip_loop",
            default_value=True,
            description="Remove repeating word/phrase loops at the end of text.",
            gui_type="checkbox",
            gui_label="Strip Loops",
            gui_info="Remove repeating word/phrase patterns at the end of output"
        )
    
    def validate(self, value) -> bool:
        if value is None:
            return self.config.default_value
        return bool(value)
    
    @staticmethod
    def apply(text: str, min_repeats: int = 3, **kwargs) -> str:
        """
        Detect and remove repeating patterns at the end of text.
        
        Args:
            text: Input text that may have looping patterns
            min_repeats: Minimum number of repeats to consider it a loop (default 3)
            
        Returns:
            Text with trailing loops removed
        """
        if not text or len(text) < 10:
            return text
            
        # Helper: Tokenize but keep delimiters to reconstruct text
        # Tokens will be words, or delimiter strings
        tokens = [t for t in re.split(r'(\b\w+\b)', text) if t]
        
        # Analyze from the end
        n = len(tokens)
        
        # We look for a period 'p' (length of the repeating pattern in tokens)
        # We try periods from 1 up to n/2
        # Example: "A, B, A, B" -> Tokens ["A", ", ", "B", ", ", "A", ", ", "B"] (Length 7)
        # Pattern "A", ", ", "B", ", " -> Length 4.
        
        max_period = min(50, n // 2) # Limit period check size for performance
        best_cut_index = -1
        
        for p in range(1, max_period + 1):
            # Check pattern of length p ending at n
            # But the loop might be cut off: "A B A B A" (Length 5, Pattern A B (2))
            # We need to find if the suffix is a repetition of the pattern preceding it.
            
            # Implementation strategy:
            # 1. Take the last 'p' tokens as candidate pattern
            # 2. Check if the block before it matches
            # 3. If so, count backwards
            
            # Refined Strategy:
            # Check candidate pattern starting at n-p
            pattern = tokens[n-p:n]
            
            # Look backwards for this pattern
            cursor = n - p
            repeats = 1
            
            while cursor >= p:
                prev_chunk = tokens[cursor-p : cursor]
                if prev_chunk == pattern:
                    repeats += 1
                    cursor -= p
                else:
                    break
            
            # Check for partial matches at the very end? 
            # (Not effectively handled by above simple logic, but usually loops run for a while)
            # If we found a solid block of repeats
            if repeats >= min_repeats:
                # We found a loop!
                # Cut at the start of the first repeat in this sequence
                # (Keep one instance? Or keep zero if it's strictly additive garbage? 
                #  Usually we want to keep the first instance of the phrase)
                
                # Logic: The text ends with 'repeats' copies of 'pattern'.
                # We want to keep only 1 copy.
                best_cut_index = n - (repeats - 1) * p
                
                # For very short patterns (single punctuation or word), require more repeats to be safe
                # e.g. " . . . "
                raw_pattern_text = "".join(pattern).strip()
                if len(raw_pattern_text) < 3 and repeats < 5:
                    continue
                    
                # We prioritize the longest period found (usually covers "A B A B" better than "A" "B")
                # BUT, if we find a loop of 10 repeats of length 2, vs 2 repeats of length 10...
                # We essentially want to cut as much as possible?
                # Actually, usually getting the largest 'repeats' count is good, or largest 'total length covering'.
                
                # If we construct the string up to best_cut_index
                new_text = "".join(tokens[:best_cut_index])
                
                # Clean trailing comma/whitespace caused by cut
                # e.g. "Word, Word, " -> "Word" (Cut leaves "Word, ")
                new_text = new_text.rstrip(' ,.;')
                
                return new_text
                
        # Fallback: Check for character-level loops (for cases not splitting cleanly on word boundaries)
        # e.g. "......" or "abababab"
        
        return text
