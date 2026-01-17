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
        """
        Initialize the GalleryState with a configuration manager.
        
        Stores the provided config manager on self.config, reads global gallery settings, and sets self.columns and self.rows using 'gallery_columns' and 'gallery_rows' with defaults 4 and 3. If 'gallery_rows' is not present but 'gallery_height' is provided, derives rows as max(1, int(gallery_height) // BASE_ROW_HEIGHT).
        """
        self.config = config_mgr
        
        settings = config_mgr.get_global_settings()
        self.columns = settings.get('gallery_columns', 4)
        self.rows = settings.get('gallery_rows', 3)
        
        if 'gallery_rows' not in settings and 'gallery_height' in settings:
            self.rows = max(1, int(settings['gallery_height']) // self.BASE_ROW_HEIGHT)
    
    def calc_height(self) -> int:
        """
        Compute the gallery's display height taking the current rows and columns into account.
        
        Returns:
            int: Gallery height in pixels computed from rows, base row height, and reference columns.
        """
        return int(self.rows * self.BASE_ROW_HEIGHT * (self.REF_COLUMNS / max(1, self.columns)))
    
    def prepare_gallery_data(self, images: List, start: int, end: int) -> List[Tuple[Any, str]]:
        """
        Prepare gallery items for display by pairing a thumbnail or path with each file's name.
        
        Parameters:
            images (List): Sequence of objects with a `path` attribute (a pathlib.Path) and an optional `thumbnail` attribute for video items.
            start (int): Start index (inclusive) of the slice to process.
            end (int): End index (exclusive) of the slice to process.
        
        Returns:
            List[Tuple[Any, str]]: List of (thumbnail_or_path, filename) tuples. For files with a video extension the object's `thumbnail` is used when present and truthy, otherwise the file path string is used; for non-video files the file path string is used.
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
        """
        Update the gallery's column and row counts.
        
        Parameters:
            columns (int | None): New number of columns; if provided, stored as an integer.
            rows (int | None): New number of rows; if provided, stored as an integer.
        """
        if columns is not None:
            self.columns = int(columns)
        if rows is not None:
            self.rows = int(rows)