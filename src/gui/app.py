"""
CaptioningApp class - Core application logic for the GUI. Keep this script clean. Write actual functionalities elsewhere  and import them here only when necessary
"""

import math
import gradio as gr
import logging
import os
import zipfile
import tempfile
from pathlib import Path

# Colorama for colored console output
try:
    from colorama import Fore, Style, init
    init(autoreset=True)
    HAS_COLORAMA = True
except ImportError:
    HAS_COLORAMA = False
    class Fore:
        CYAN = YELLOW = GREEN = MAGENTA = RED = WHITE = RESET = ''
    class Style:
        BRIGHT = DIM = RESET_ALL = ''

from src.core.config import ConfigManager
from src.core.loader import DataLoader
from src.core.dataset import Dataset
from src.core.registry import ModelRegistry

from .constants import GLOBAL_DEFAULTS, UI_CONFIG_MAP, filter_user_overrides
from .state import (
    DatasetManager, PaginationState, GalleryState,
    ModelManager, PresetManager, InspectorState
)
from .cli_generator import generate_cli_command as _generate_cli_command
from . import multi_model as _multi_model
from . import file_loader as _file_loader
from . import presets_logic as _presets
from . import inspector_logic as _inspector
from . import run_inference_logic as _run_inference
from . import settings_logic as _settings
from .logic import model_logic as _model_logic
import src.features as feature_registry

logger = logging.getLogger("GUI")


class CaptioningApp:
    """Main application class for the A Thousand Words GUI."""
    
    def __init__(self):
        self.config_mgr = ConfigManager()
        
        # Force re-read user_config from disk on app creation
        self.config_mgr.user_config = self.config_mgr._load_yaml(self.config_mgr.user_config_path)
        
        self.registry = ModelRegistry()
        
        # Initialize state modules
        self._dataset_mgr = DatasetManager(self.config_mgr)
        self._model_mgr = ModelManager(self.config_mgr, self.registry)
        self._presets = PresetManager(self.config_mgr)
        self._inspector = InspectorState()
        
        # Gallery and Pagination
        settings = self.config_mgr.get_global_settings()
        items_per_page = settings.get('gallery_items_per_page', 50)
        self._pagination = PaginationState(items_per_page)
        self._gallery = GalleryState(self.config_mgr)
    
    # Property shortcuts delegating to state modules
    
    @property
    def dataset(self):
        """Access dataset object (delegate to manager)."""
        return self._dataset_mgr.dataset
    
    @dataset.setter
    def dataset(self, value):
        self._dataset_mgr.dataset = value
    
    @property
    def models(self):
        """All available model IDs."""
        return self._model_mgr.all_models
    
    @property
    def enabled_models(self):
        """Enabled model IDs only."""
        return self._model_mgr.enabled_models
    
    @property
    def current_model_id(self):
        """Currently selected model ID."""
        return self._model_mgr.current_model_id
    
    @current_model_id.setter
    def current_model_id(self, value):
        self._model_mgr.current_model_id = value
    
    @property
    def selected_path(self):
        """Currently selected path in inspector."""
        return self._inspector.selected_path
    
    @selected_path.setter
    def selected_path(self, value):
        self._inspector.selected_path = value
    
    @property
    def selected_index(self):
        """Currently selected index in gallery."""
        return self._inspector.selected_index
    
    @selected_index.setter
    def selected_index(self, value):
        self._inspector.selected_index = value
    
    @property
    def default_input_dir(self):
        """Default input directory path."""
        return self._dataset_mgr.default_input_dir
    
    @property
    def input_list_path(self):
        """Path to input file list JSON."""
        return self._dataset_mgr.input_list_path
    
    @property
    def current_input_path(self):
        """Current input path string."""
        return self._dataset_mgr.current_input_path
    
    @current_input_path.setter
    def current_input_path(self, value):
        self._dataset_mgr.current_input_path = value
    
    @property
    def is_drag_and_drop(self):
        """True if current source is drag-and-drop."""
        return self._dataset_mgr.is_drag_and_drop
    
    @is_drag_and_drop.setter
    def is_drag_and_drop(self, value):
        self._dataset_mgr.is_drag_and_drop = value
    
    @property
    def gallery_columns(self):
        """Gallery columns count."""
        return self._gallery.columns
    
    @gallery_columns.setter
    def gallery_columns(self, value):
        self._gallery.columns = value
    
    @property
    def gallery_rows(self):
        """Gallery rows count."""
        return self._gallery.rows
    
    @gallery_rows.setter
    def gallery_rows(self, value):
        self._gallery.rows = value
    
    @property
    def gallery_items_per_page(self):
        """Items per page limit."""
        return self._pagination.items_per_page
    
    @gallery_items_per_page.setter
    def gallery_items_per_page(self, value):
        self._pagination.items_per_page = value
    
    @property
    def current_page(self):
        """Current page number (1-based)."""
        return self._pagination.current_page
    
    @current_page.setter
    def current_page(self, value):
        self._pagination.current_page = value


    def calc_gallery_height(self):
        """Calculate dynamic gallery height based on rows/cols to maintain aspect ratio."""
        return _settings.calc_gallery_height(self)

    def save_last_model(self, mod_id):
        """Saves the last used model to user config."""
        return _settings.save_last_model(self, mod_id)
    
    def move_model_up(self, selected_model, current_order_state):
        """Move selected model up in the order list."""
        return _settings.move_model_up(self, selected_model, current_order_state)
    
    def move_model_down(self, selected_model, current_order_state):
        """Move selected model down in the order list."""
        return _settings.move_model_down(self, selected_model, current_order_state)

    def get_models_by_media_type(self, media_type: str) -> list:
        """Filter models by media type (Image or Video).
        
        Models with media_type as list will appear in all matching categories.
        """
        filtered = []
        for model_id in self.enabled_models:
            config = self.config_mgr.get_model_config(model_id)
            model_media_types = config.get('media_type', 'Image')  # Default to Image if not specified
            
            # Handle both string and list formats
            if isinstance(model_media_types, str):
                model_media_types = [model_media_types]
            
            # Check if requested media type is in model's supported types
            if media_type in model_media_types:
                filtered.append(model_id)
        
        return filtered
    
    def analyze_input_paths(self):
        """Analyze the current dataset paths to determine structure."""
        return _file_loader.analyze_input_paths(self.dataset)

    def create_zip(self, file_paths: list) -> str:
        """Create zip file, preserving relative structure if possible."""
        return _file_loader.create_zip(file_paths)

    def refresh_models(self):
        """Refresh the list of enabled models."""
        self._model_mgr.enabled_models = self.config_mgr.get_enabled_models()
        new_val = self.current_model_id if self.current_model_id in self.enabled_models else (self.enabled_models[0] if self.enabled_models else None)
        return gr.update(choices=self.enabled_models, value=new_val), gr.update(choices=self.models, value=self.enabled_models)

    def save_settings(self, vram, models_checked, gal_cols, gal_rows, limit_cnt, o_dir, o_fmt, over, rec, con, unload, pre, suf, clean, collapse, normalize, remove_cn, strip_loop, max_w, max_h, current_mod_id, model_ver, batch_sz, max_tok, settings_state, items_per_page):
        """Save settings from the main UI directly to user_config.yaml."""
        return _settings.save_settings(self, vram, models_checked, gal_cols, gal_rows, limit_cnt, o_dir, o_fmt, over, rec, con, unload, pre, suf, clean, collapse, normalize, remove_cn, strip_loop, max_w, max_h, current_mod_id, model_ver, batch_sz, max_tok, settings_state, items_per_page)
    
    def save_settings_simple(self, vram, system_ram, models_checked, gal_cols, gal_rows, unload, model_order_text, items_per_page, theme_mode):
        """Save settings from the Settings tab (simplified version)."""
        return _settings.save_settings_simple(self, vram, system_ram, models_checked, gal_cols, gal_rows, unload, model_order_text, items_per_page, theme_mode)
    
    def reset_to_defaults(self):
        """Delete user_config.yaml to reset all settings to defaults."""
        return _settings.reset_to_defaults(self)

    # User Presets Library (delegated to presets_logic module)

    def get_preset_eligible_models(self):
        """Return list of models that support custom prompts."""
        return _presets.get_preset_eligible_models(self)

    def get_user_presets_dataframe(self):
        """Get user presets formatted for the Settings Dataframe."""
        return _presets.get_user_presets_dataframe(self)

    def save_user_preset(self, model_scope, name, text):
        """Save (Upsert) a user preset."""
        return _presets.save_user_preset(self, model_scope, name, text)

    def delete_user_preset(self, model_scope, name):
        """Delete a user preset."""
        return _presets.delete_user_preset(self, model_scope, name)


    def load_settings(self):
        """Force re-read from disk (reload) and return UI values."""
        return _settings.load_settings(self)

    def load_files(self, file_objs):
        """Load files from drag-and-drop - ADDS to existing dataset."""
        if not file_objs:
            return self._get_gallery_data(), None, 1, self.get_total_label(), self._get_pagination_vis()
        
        uploads_dir = Path("user/uploads")
        
        # Persist uploaded files to stable location
        persistent_files = _file_loader.persist_uploaded_files(file_objs, uploads_dir)
        
        if not persistent_files:
            return self._get_gallery_data(), None, 1, self.get_total_label(), self._get_pagination_vis()

        # Load new files into dataset
        _file_loader.load_new_files_to_dataset(self.dataset, persistent_files)
        
        self.current_input_path = str(uploads_dir.absolute())
        self.is_drag_and_drop = True
        
        self._save_dataset_list()
        self.current_page = 1
        return self._get_gallery_data(), None, 1, self.get_total_label(), self._get_pagination_vis()


    def load_input_source(self, folder_path, recursive=False, limit_count=0):
        """Load images from path (manual) or default input folder if empty."""
        path_str = folder_path.strip()
        
        # Determine source path
        if not path_str:
            # Fallback to default ./input
            source_path = self.default_input_dir
            if not source_path.exists():
                # Allow default input to not exist silently? Or warn?
                # Usually better to warn if explicit action taken
                pass 
        else:
            source_path = Path(path_str)
            
        # Validate existance
        if not source_path.exists():
            gr.Warning(f"Folder not found: {source_path}")
            return [], gr.update(visible=False)
            
        self.dataset = DataLoader.scan_directory(str(source_path.absolute()), recursive=recursive)
        
        # Apply count limit if specified
        try:
            limit = int(limit_count)
            if limit > 0 and len(self.dataset.images) > limit:
                print(f"Limiting to first {limit} files (from {len(self.dataset.images)} detected).")
                self.dataset.images = self.dataset.images[:limit]
        except (ValueError, TypeError):
            # Ignore invalid input (empty string, non-int)
            pass
            
        self._save_dataset_list()
        
        self._save_dataset_list()
        
        # Update current input path logic
        self.current_input_path = str(source_path.absolute())
        self.is_drag_and_drop = False  # Manual folder load, not drag/drop
        
        self.current_page = 1
        return self._get_gallery_data(), gr.update(visible=True), 1, self.get_total_label(), self._get_pagination_vis()

    def _save_dataset_list(self):
        """Save current dataset file paths to a JSON list for CLI usage."""
        import json
        try:
            file_list = [str(img.path.absolute()) for img in self.dataset.images]
            with open(self.input_list_path, 'w', encoding='utf-8') as f:
                json.dump(file_list, f, indent=2)
            self.current_input_path = str(self.input_list_path)
        except Exception as e:
            logger.error(f"Failed to save temp input list: {e}")

    def get_total_pages(self):
        """Calculate total number of pages."""
        total_items = len(self.dataset.images)
        if total_items == 0:
            return 1
        return math.ceil(total_items / max(1, self.gallery_items_per_page))

    def get_page_info(self):
        """Get pagination info string."""
        total_pages = self.get_total_pages()
        return f"Page {self.current_page} of {total_pages}"

    def get_total_label(self):
        """Get total pages label string."""
        return f"/ {self.get_total_pages()}"

    def _get_pagination_vis(self):
        """Get visibility update for pagination row (hide if 1 page or less)."""
        return gr.update(visible=self.get_total_pages() > 1)

    def next_page(self):
        """Go to next page."""
        if self.current_page < self.get_total_pages():
            self.current_page += 1
        return self._get_gallery_data(), self.current_page, self.get_total_label(), self._get_pagination_vis()

    def prev_page(self):
        """Go to previous page."""
        if self.current_page > 1:
            self.current_page -= 1
        return self._get_gallery_data(), self.current_page, self.get_total_label(), self._get_pagination_vis()
    
    def jump_to_page(self, page_num):
        """Jump to specific page."""
        try:
            val = int(page_num)
            if 1 <= val <= self.get_total_pages():
                self.current_page = val
        except (ValueError, TypeError):
            pass
        return self._get_gallery_data(), self.current_page, self.get_total_label(), self._get_pagination_vis()

    def update_items_per_page(self, count):
        """Update items per page and reset to page 1."""
        try:
            val = int(count)
            if val > 0:
                self.gallery_items_per_page = val
                self.current_page = 1
        except ValueError:
            pass
        return self._get_gallery_data(), 1, self.get_total_label(), self._get_pagination_vis()
    
    def _get_gallery_data(self):
        """Get gallery data for display with video indicators, respecting pagination."""
        gallery_data = []
        
        # Calculate slice
        start_idx = (self.current_page - 1) * self.gallery_items_per_page
        end_idx = start_idx + self.gallery_items_per_page
        
        # Ensure within bounds
        visible_images = self.dataset.images[start_idx:end_idx]
        
        for media in visible_images:
            # Use video thumbnail if available, otherwise use image path
            image_path = str(media.path)
            caption = media.caption or media.path.name
            
            # Add video emoji indicator to caption for videos
            if media.is_video():
                thumb_path = media.get_thumbnail_path()
                if thumb_path:
                    image_path = thumb_path
                caption = f"🎬 {caption}"
            
            gallery_data.append((image_path, caption))
        
        return gallery_data

    def open_inspector(self, evt: gr.SelectData):
        """Open the inspector panel for a selected media item."""
        return _inspector.open_inspector(self, evt)
    
    def remove_from_gallery(self):
        """Remove currently selected image from dataset (not from disk)."""
        return _inspector.remove_from_gallery(self)

    def save_and_close(self, caption):
        """Save caption and close inspector."""
        return _inspector.save_and_close(self, caption)

    def close_inspector(self):
        """Close the inspector panel."""
        return _inspector.close_inspector()
    
    def clear_gallery(self):
        """Clear the dataset, gallery, and close inspector."""
        return _inspector.clear_gallery(self)


    def update_model_ui(self, mod_id):
        """Update UI components based on selected model."""
        return _model_logic.update_model_ui(self, mod_id)

    def apply_preset(self, mod_id, preset_name):
        """Apply a prompt preset."""
        return _model_logic.apply_preset(self, mod_id, preset_name)

    def auto_save_setting(self, key, value):
        """Saves a UI setting to user_config.yaml automatically."""
        return _settings.auto_save_setting(self, key, value)

    def save_model_defaults(self, mod_id, t, k, mt, rp):
        """Saves current generation settings as user defaults for the model."""
        return _settings.save_model_defaults(self, mod_id, t, k, mt, rp)

    def reset_to_global(self, key):
        """Returns the global default for a specific key."""
        return _settings.reset_to_global(self, key)

    def generate_cli_command(self, mod, args, skip_defaults=True):
        """Generate CLI command without running inference."""
        return _generate_cli_command(self.config_mgr, mod, args, self.current_input_path, skip_defaults)

    
    def run_inference(self, mod, args):
        """Run inference on the dataset."""
        return _run_inference.run_inference(self, mod, args)
    
    # Multi-Model Captioning Methods
    
    def _sanitize_model_name(self, model_id: str) -> str:
        """Convert model ID to sanitized output format extension."""
        return _multi_model.sanitize_model_name(model_id)
    
    def load_multi_model_settings(self):
        """Load multi-model configuration from user_config.yaml"""
        return _multi_model.load_multi_model_settings(self.config_mgr, self.models)
    
    def save_multi_model_settings(self, *inputs):
        """Save multi-model configuration to user_config."""
        return _multi_model.save_multi_model_settings(self.config_mgr, self.models, *inputs)
    
    def generate_multi_model_commands(self, *inputs):
        """Generate CLI commands for all enabled models."""
        return _multi_model.generate_multi_model_commands(self, *inputs)
    
    def generate_multi_model_commands_with_settings(self, current_settings, checkboxes, formats):
        """Generate CLI commands using current UI settings."""
        return _multi_model.generate_multi_model_commands_with_settings(self, current_settings, checkboxes, formats)
    
    def run_multi_model_inference(self, *inputs):
        """Run multiple models sequentially on the dataset."""
        return _multi_model.run_multi_model_inference(self, *inputs)
