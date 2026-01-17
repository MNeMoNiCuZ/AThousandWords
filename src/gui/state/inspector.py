"""
Inspector state management.

Handles selection state for the media inspector panel.
"""

from pathlib import Path
from typing import Optional, Any


class InspectorState:
    """Manages inspector panel selection state."""
    
    def __init__(self):
        self.selected_index: Optional[int] = None
        self.selected_path: Optional[Path] = None
        self.is_open: bool = False
    
    def open(self, index: int, path: Path):
        """Open inspector for selected item."""
        self.selected_index = index
        self.selected_path = path
        self.is_open = True
    
    def close(self):
        """Close inspector panel."""
        self.selected_index = None
        self.selected_path = None
        self.is_open = False
    
    def get_caption_path(self) -> Optional[Path]:
        """Get path to caption file for current selection."""
        if self.selected_path:
            return self.selected_path.with_suffix('.txt')
        return None
    
    def read_caption(self) -> str:
        """Read caption for current selection."""
        caption_path = self.get_caption_path()
        if caption_path and caption_path.exists():
            try:
                return caption_path.read_text(encoding='utf-8')
            except Exception:
                return ""
        return ""
    
    def save_caption(self, text: str) -> bool:
        """Save caption for current selection.
        
        Returns True on success.
        """
        caption_path = self.get_caption_path()
        if not caption_path:
            return False
        
        try:
            caption_path.write_text(text, encoding='utf-8')
            return True
        except Exception:
            return False
    
    @property
    def has_selection(self) -> bool:
        """Check if an item is selected."""
        return self.selected_index is not None
