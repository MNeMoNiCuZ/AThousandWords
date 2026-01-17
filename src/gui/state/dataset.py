"""
Dataset loading and management.

Handles file/folder loading, path tracking, and dataset state.
"""

import json
import logging
import shutil
import uuid
import re
from pathlib import Path
from typing import List, Tuple, Optional

from src.core.loader import DataLoader
from src.core.dataset import Dataset

logger = logging.getLogger("GUI.DatasetManager")


class DatasetManager:
    """Manages dataset loading and file operations."""
    
    def __init__(self, config_mgr):
        self.config = config_mgr
        self.dataset = Dataset()
        
        self.default_input_dir = Path("input")
        self.default_input_dir.mkdir(exist_ok=True)
        
        self.input_list_path = Path("user/input.json")
        self.current_input_path = str(self.input_list_path)
        self.is_drag_and_drop = False
    
    def clear(self):
        """Clear the dataset."""
        self.dataset = Dataset()
        self.is_drag_and_drop = False
    
    def load_files(self, file_objs) -> int:
        """Load files from drag-and-drop.
        
        Returns number of files added.
        """
        if not file_objs:
            return 0
        
        uploads_dir = Path("user/uploads")
        uploads_dir.mkdir(parents=True, exist_ok=True)
        
        persistent_files = []
        uuid_pattern = re.compile(r'^[0-9a-f]{32}_(.+)$')
        
        for file_obj in file_objs:
            temp_path_str = file_obj.name if hasattr(file_obj, 'name') else file_obj
            temp_path = Path(temp_path_str)
            
            if not temp_path.exists():
                logger.warning(f"Temp file missing: {temp_path}")
                continue
            
            filename = temp_path.name
            match = uuid_pattern.match(filename)
            original_filename = match.group(1) if match else filename
            
            safe_name = f"{uuid.uuid4().hex}_{original_filename}"
            dest_path = uploads_dir / safe_name
            
            try:
                shutil.copy2(temp_path, dest_path)
                persistent_files.append(str(dest_path.absolute()))
            except Exception as e:
                logger.error(f"Failed to persist file {temp_path}: {e}")
        
        if not persistent_files:
            return 0
        
        new_dataset = DataLoader.scan_directory(persistent_files)
        existing_paths = {str(img.path) for img in self.dataset.images}
        
        added = 0
        for img in new_dataset.images:
            if str(img.path) not in existing_paths:
                self.dataset.images.append(img)
                added += 1
        
        self.current_input_path = str(uploads_dir.absolute())
        self.is_drag_and_drop = True
        self._save_dataset_list()
        
        return added
    
    def load_from_path(self, folder_path: str, recursive: bool = False, 
                        limit_count: int = 0) -> Tuple[bool, str]:
        """Load images from path.
        
        Returns (success, message).
        """
        path_str = folder_path.strip() if folder_path else ""
        
        if not path_str:
            source_path = self.default_input_dir
        else:
            source_path = Path(path_str)
        
        if not source_path.exists():
            return False, f"Folder not found: {source_path}"
        
        self.dataset = DataLoader.scan_directory(str(source_path.absolute()), recursive=recursive)
        
        try:
            limit = int(limit_count)
            if limit > 0 and len(self.dataset.images) > limit:
                logger.info(f"Limiting to first {limit} files (from {len(self.dataset.images)} detected)")
                self.dataset.images = self.dataset.images[:limit]
        except (ValueError, TypeError):
            pass
        
        self._save_dataset_list()
        self.current_input_path = str(source_path.absolute())
        self.is_drag_and_drop = False
        
        return True, f"Loaded {len(self.dataset.images)} files"
    
    def _save_dataset_list(self):
        """Save current dataset file paths to JSON for CLI usage."""
        try:
            self.input_list_path.parent.mkdir(parents=True, exist_ok=True)
            file_list = [str(img.path.absolute()) for img in self.dataset.images]
            with open(self.input_list_path, 'w', encoding='utf-8') as f:
                json.dump(file_list, f, indent=2)
            self.current_input_path = str(self.input_list_path)
        except Exception as e:
            logger.error(f"Failed to save input list: {e}")
    
    def analyze_paths(self) -> Tuple[Optional[Path], bool, List]:
        """Analyze dataset paths for common root and collisions.
        
        Returns (common_root, mixed_sources, collisions).
        """
        if not self.dataset.images:
            return None, False, []
        
        all_paths = [img.path for img in self.dataset.images]
        
        try:
            parents = [set(p.resolve().parents) for p in all_paths]
            common_ancestors = parents[0].intersection(*parents[1:]) if len(parents) > 1 else parents[0]
            
            if common_ancestors:
                common_root = max(common_ancestors, key=lambda p: len(str(p)))
            else:
                common_root = None
        except Exception:
            common_root = None
        
        mixed_sources = len(set(p.parent for p in all_paths)) > 1
        
        filenames = [p.name for p in all_paths]
        seen = {}
        collisions = []
        for name in filenames:
            if name in seen:
                collisions.append(name)
            seen[name] = True
        
        return common_root, mixed_sources, list(set(collisions))
    
    @property
    def count(self) -> int:
        """Number of images in dataset."""
        return len(self.dataset.images)
    
    @property
    def images(self):
        """Dataset images list."""
        return self.dataset.images
