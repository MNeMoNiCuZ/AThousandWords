"""
Gallery display state management.

Handles gallery layout settings and data preparation.
"""

from pathlib import Path
from typing import List, Tuple, Any


class GalleryState:
    """Manages gallery display settings."""
    
    REF_COLUMNS = 4
    BASE_ROW_HEIGHT = 210
    
    def __init__(self, config_mgr):
        self.config = config_mgr
        
        settings = config_mgr.get_global_settings()
        self.columns = settings.get('gallery_columns', 4)
        self.rows = settings.get('gallery_rows', 3)
        
        if 'gallery_rows' not in settings and 'gallery_height' in settings:
            self.rows = max(1, int(settings['gallery_height']) // self.BASE_ROW_HEIGHT)
    
    def calc_height(self) -> int:
        """Calculate dynamic gallery height based on rows and columns."""
        return int(self.rows * self.BASE_ROW_HEIGHT * (self.REF_COLUMNS / max(1, self.columns)))
    
    def prepare_gallery_data(self, images: List, start: int, end: int) -> List[Tuple[Any, str]]:
        """Prepare gallery data with video indicators.
        
        Args:
            images: List of image objects
            start: Start index
            end: End index
            
        Returns:
            List of (thumbnail, filename) tuples
        """
        VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv', '.wmv', '.m4v'}
        
        page_items = images[start:end]
        gallery_data = []
        
        for img_obj in page_items:
            path = img_obj.path
            ext = path.suffix.lower()
            
            if ext in VIDEO_EXTENSIONS:
                thumbnail = img_obj.thumbnail if hasattr(img_obj, 'thumbnail') and img_obj.thumbnail else str(path)
                gallery_data.append((thumbnail, path.name))
            else:
                gallery_data.append((str(path), path.name))
        
        return gallery_data
    
    def update_settings(self, columns: int = None, rows: int = None):
        """Update gallery settings."""
        if columns is not None:
            self.columns = int(columns)
        if rows is not None:
            self.rows = int(rows)
