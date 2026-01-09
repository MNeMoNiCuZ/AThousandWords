import os
from pathlib import Path
from typing import List, Generator
from PIL import Image
from .dataset import Dataset, MediaObject

VALID_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tiff', '.mp4', '.avi', '.mov', '.mkv', '.webm'}

class DataLoader:
    @staticmethod
    def scan_directory(input_path: str, recursive: bool = False) -> Dataset:
        if isinstance(input_path, list):
            images = []
            for file_obj in input_path:
                # Handle Gradio objects or direct paths
                p = file_obj.name if hasattr(file_obj, 'name') else file_obj
                fpath = Path(p)
                
                if fpath.suffix.lower() in VALID_EXTENSIONS:
                    # Check for sibling text file explicitly
                    # This works if fpath is a real path on disk (or temp with siblings)
                    images.append(DataLoader._create_image_object(fpath))
            return Dataset(images)

        # Standard Path String logic
        root_path = Path(input_path)
        if not root_path.exists():
            return Dataset()

        images = []
        
        if root_path.is_file():
            # Support reading from a JSON list of file paths
            if root_path.suffix.lower() == '.json':
                import json
                try:
                    with open(root_path, 'r', encoding='utf-8') as f:
                        file_list = json.load(f)
                    
                    if isinstance(file_list, list):
                        images = []
                        for p in file_list:
                             # Handle paths relative to the json file's directory if they are not absolute
                             fp = Path(p)
                             if not fp.is_absolute():
                                 fp = root_path.parent / fp
                                 
                             if fp.exists() and fp.suffix.lower() in VALID_EXTENSIONS:
                                 images.append(DataLoader._create_image_object(fp))
                        return Dataset(images)
                except Exception as e:
                    print(f"Error reading input list json: {e}")
                    return Dataset()
            
            elif root_path.suffix.lower() in VALID_EXTENSIONS:
                images.append(DataLoader._create_image_object(root_path))
        else:
            pattern = "**/*" if recursive else "*"
            for file_path in root_path.glob(pattern):
                if file_path.suffix.lower() in VALID_EXTENSIONS:
                    images.append(DataLoader._create_image_object(file_path))
                    
        return Dataset(images)

    @staticmethod
    def _create_image_object(path: Path) -> MediaObject:
        # Optimistic loading: We don't read the image bytes yet, just path
        # If there's a matching .txt, load it as existing caption
        caption = ""
        txt_path = path.with_suffix('.txt')
        if txt_path.exists():
            try:
                caption = txt_path.read_text(encoding='utf-8').strip()
            except Exception:
                pass
                
        # Basic metadata shell (media_type is auto-detected in __post_init__)
        return MediaObject(
            path=path,
            caption=caption,
            metadata={} 
        )
