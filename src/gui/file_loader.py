"""File loading utilities for the GUI."""

import shutil
import uuid
import re
import logging
from pathlib import Path

from src.core.loader import DataLoader

logger = logging.getLogger("GUI.FileLoader")


def persist_uploaded_files(file_objs, uploads_dir: Path) -> list:
    """
    Persist uploaded files from temporary paths into a stable uploads directory.
    
    Strips an optional Gradio UUID prefix from each uploaded filename, writes each file to uploads_dir with a new UUID-prefixed filename to avoid collisions, and returns the absolute paths of successfully persisted files.
    
    Parameters:
        file_objs: Iterable of uploaded file objects or path-like strings as provided by Gradio; each item should reference a readable temporary file.
        uploads_dir (Path): Directory where files will be persisted; created if it does not exist.
    
    Returns:
        list[str]: Absolute paths to the files that were successfully persisted.
    """
    uploads_dir.mkdir(parents=True, exist_ok=True)
    
    persistent_files = []
    uuid_pattern = re.compile(r'^[0-9a-f]{32}_(.+)$')
    
    for file_obj in file_objs:
        temp_path_str = file_obj.name if hasattr(file_obj, 'name') else file_obj
        temp_path = Path(temp_path_str)
        
        if not temp_path.exists():
            logger.warning(f"Temp file missing: {temp_path}")
            continue
        
        # Extract original filename by stripping Gradio's UUID prefix
        filename = temp_path.name
        match = uuid_pattern.match(filename)
        original_filename = match.group(1) if match else filename
        
        # Create persistent path with UUID to avoid collisions
        safe_name = f"{uuid.uuid4().hex}_{original_filename}"
        dest_path = uploads_dir / safe_name
        
        try:
            shutil.copy2(temp_path, dest_path)
            persistent_files.append(str(dest_path.absolute()))
        except Exception as e:
            logger.error(f"Failed to persist file {temp_path}: {e}")
    
    return persistent_files


def load_new_files_to_dataset(dataset, persistent_files):
    """
    Load new files into an existing dataset, avoiding duplicates.
    
    Args:
        dataset: Existing dataset object
        persistent_files: List of paths to load
        
    Returns:
        Number of new files added
    """
    new_dataset = DataLoader.scan_directory(persistent_files)
    
    existing_paths = {str(img.path) for img in dataset.images}
    added_count = 0
    
    for img in new_dataset.images:
        if str(img.path) not in existing_paths:
            dataset.images.append(img)
            added_count += 1
    
    return added_count


def analyze_input_paths(dataset):
    """
    Determine the common directory for all image paths in a dataset, whether the images originate from mixed sources, and any filename collisions.
    
    Parameters:
        dataset: Dataset object whose `images` attribute is an iterable of objects with a `path` attribute (a Path-like object).
    
    Returns:
        tuple: (common_root, mixed_sources, collisions)
            common_root (Path or None): the common directory path for all images, or `None` if no common path can be computed.
            mixed_sources (bool): `True` if images come from mixed sources (no single common root), `False` otherwise.
            collisions (list): list of filenames that appear under multiple different full paths.
    """
    import os
    
    if not dataset or not dataset.images:
        return None, False, []
        
    paths = [img.path.absolute() for img in dataset.images]
    
    filename_map = {}
    collisions = []
    for p in paths:
        name = p.name
        if name in filename_map:
            if filename_map[name] != p:
                collisions.append(name)
        else:
            filename_map[name] = p
    
    collisions = list(set(collisions))
    
    try:
        common_root = Path(os.path.commonpath(paths))
        mixed_sources = False
    except ValueError:
        common_root = None
        mixed_sources = True
        
    return common_root, mixed_sources, collisions


def create_zip(file_paths: list) -> str:
    """
    Create a ZIP archive containing the given files, preserving their relative paths when a common root can be determined.
    
    Parameters:
        file_paths (list): Iterable of file system paths (strings or Path-like) to include in the archive.
    
    Returns:
        str or None: Absolute path to the created ZIP file, or `None` if no files were provided or the archive could not be created.
    """
    import zipfile
    import tempfile
    import datetime
    import os
    
    if not file_paths:
        return None
        
    try:
        paths = [Path(p).absolute() for p in file_paths]
        common_root = Path(os.path.commonpath(paths))
    except ValueError:
        common_root = None
        
    try:
        export_dir = Path(tempfile.gettempdir())
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        zip_name = f"captions_{timestamp}.zip"
        zip_path = export_dir / zip_name
        
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
            for file_path in file_paths:
                p = Path(file_path)
                if p.exists():
                    arcname = p.relative_to(common_root) if common_root else p.name
                    zf.write(p, arcname=arcname)
        
        return str(zip_path.absolute())
        
    except Exception as e:
        logger.error(f"Failed to create zip: {e}")
        return None
