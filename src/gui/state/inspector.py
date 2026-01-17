"""
Inspector state management.

Handles selection state for the media inspector panel.
"""

from pathlib import Path
from typing import Optional, Any


class InspectorState:
    """Manages inspector panel selection state."""
    
    def __init__(self):
        """
        Initialize inspector state managing the inspector panel selection.
        
        Attributes:
            selected_index (Optional[int]): Index of the selected item, or None if no selection.
            selected_path (Optional[Path]): Path of the selected item, or None if no selection.
            is_open (bool): True if the inspector panel is open, False otherwise.
        """
        self.selected_index: Optional[int] = None
        self.selected_path: Optional[Path] = None
        self.is_open: bool = False
    
    def open(self, index: int, path: Path):
        """
        Open the inspector for the given item and mark it as open.
        
        Parameters:
            index (int): Index of the selected item.
            path (Path): Filesystem path of the selected item.
        """
        self.selected_index = index
        self.selected_path = path
        self.is_open = True
    
    def close(self):
        """
        Close the inspector and clear its current selection.
        
        Resets `selected_index` and `selected_path` to None and sets `is_open` to False.
        """
        self.selected_index = None
        self.selected_path = None
        self.is_open = False
    
    def get_caption_path(self) -> Optional[Path]:
        """
        Get the .txt caption file path for the current selection.
        
        Returns:
            Optional[Path]: Path to the caption file with suffix `.txt` corresponding to `selected_path`, or `None` if no selection.
        """
        if self.selected_path:
            return self.selected_path.with_suffix('.txt')
        return None
    
    def read_caption(self) -> str:
        """
        Retrieve caption text associated with the current selection.
        
        Returns:
            caption (str): Caption text if the corresponding `.txt` caption file exists and is readable; an empty string if there is no selection, the caption file does not exist, or an error occurs while reading.
        """
        caption_path = self.get_caption_path()
        if caption_path and caption_path.exists():
            try:
                return caption_path.read_text(encoding='utf-8')
            except Exception:
                return ""
        return ""
    
    def save_caption(self, text: str) -> bool:
        """
        Save the given caption text to the caption file for the current selection.
        
        Returns:
            True if the caption was written successfully, False if there is no caption path or the write failed.
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
        """
        Indicates whether there is a current selection.
        
        Returns:
            True if a selected index is set, False otherwise.
        """
        return self.selected_index is not None