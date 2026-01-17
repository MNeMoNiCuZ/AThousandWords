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
        """
        Initialize core application managers and UI state.
        
        Creates a ConfigManager and reloads the user configuration from disk, constructs the ModelRegistry, and initializes state modules: DatasetManager, ModelManager, PresetManager, and InspectorState. Initializes pagination and gallery state using global settings, defaulting gallery_items_per_page to 50 if the setting is absent.
        """
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
        """
        Return the application's current dataset.
        
        Returns:
            Dataset: The Dataset instance containing loaded media paths and associated metadata used by the app.
        """
        return self._dataset_mgr.dataset
    
    @dataset.setter
    def dataset(self, value):
        """
        Set the application's current dataset.
        
        Parameters:
            value: The dataset object to assign as the current dataset for the application.
        """
        self._dataset_mgr.dataset = value
    
    @property
    def models(self):
        """All available model IDs."""
        return self._model_mgr.all_models
    
    @property
    def enabled_models(self):
        """
        List model IDs that are currently enabled.
        
        Returns:
            list: Enabled model IDs.
        """
        return self._model_mgr.enabled_models
    
    @property
    def current_model_id(self):
        """
        The currently selected model identifier.
        
        Returns:
            current_model_id (str or None): The model ID of the currently selected model, or `None` if no model is selected.
        """
        return self._model_mgr.current_model_id
    
    @current_model_id.setter
    def current_model_id(self, value):
        """
        Set the currently active model identifier.
        
        Parameters:
            value (str): The model ID to make the current active model.
        """
        self._model_mgr.current_model_id = value
    
    @property
    def selected_path(self):
        """
        Get the currently selected path in the inspector.
        
        Returns:
            The currently selected path as a string, or `None` if no item is selected.
        """
        return self._inspector.selected_path
    
    @selected_path.setter
    def selected_path(self, value):
        """
        Set the inspector's currently selected item's path.
        
        Parameters:
            value (str | None): File system path or identifier of the selected gallery item; pass None to clear the selection.
        """
        self._inspector.selected_path = value
    
    @property
    def selected_index(self):
        """
        Gets the zero-based index of the currently selected item in the gallery.
        
        Returns:
            selected_index (int | None): Zero-based index of the selected gallery item, or `None` if no item is selected.
        """
        return self._inspector.selected_index
    
    @selected_index.setter
    def selected_index(self, value):
        """
        Set the index of the currently selected gallery item in the inspector.
        
        Parameters:
            value (int | None): Zero-based index of the selected item in the dataset, or None to clear the selection.
        """
        self._inspector.selected_index = value
    
    @property
    def default_input_dir(self):
        """
        Default input directory used when no explicit input source is provided.
        
        Returns:
            str: Filesystem path of the default input directory.
        """
        return self._dataset_mgr.default_input_dir
    
    @property
    def input_list_path(self):
        """
        Filesystem path to the JSON file that stores the current dataset's input file list.
        
        Returns:
            pathlib.Path or str: Path to the input list JSON file.
        """
        return self._dataset_mgr.input_list_path
    
    @property
    def current_input_path(self):
        """
        Get the current input directory or file path used by the dataset.
        
        Returns:
            current_input_path (str): The path to the currently selected input source (directory or file), or an empty string if none is set.
        """
        return self._dataset_mgr.current_input_path
    
    @current_input_path.setter
    def current_input_path(self, value):
        """
        Set the application's current input path used to locate and load media files.
        
        Parameters:
        	value (str | Path): Filesystem path to use as the current input directory.
        """
        self._dataset_mgr.current_input_path = value
    
    @property
    def is_drag_and_drop(self):
        """
        Indicates whether the current input source originated from a drag-and-drop upload.
        
        Returns:
            bool: `True` if the current source originated from a drag-and-drop upload, `False` otherwise.
        """
        return self._dataset_mgr.is_drag_and_drop
    
    @is_drag_and_drop.setter
    def is_drag_and_drop(self, value):
        """
        Set whether the current dataset was loaded via drag-and-drop.
        
        Parameters:
            value (bool): True if the current input source came from a drag-and-drop upload, False otherwise.
        """
        self._dataset_mgr.is_drag_and_drop = value
    
    @property
    def gallery_columns(self):
        """
        Get the current number of gallery columns.
        
        Returns:
            columns (int): Number of columns in the gallery.
        """
        return self._gallery.columns
    
    @gallery_columns.setter
    def gallery_columns(self, value):
        """
        Set the number of columns displayed in the gallery.
        
        Parameters:
            value (int): Desired number of columns in the gallery layout.
        """
        self._gallery.columns = value
    
    @property
    def gallery_rows(self):
        """
        Get the number of rows in the gallery display.
        
        Returns:
            int: Number of rows displayed in the gallery.
        """
        return self._gallery.rows
    
    @gallery_rows.setter
    def gallery_rows(self, value):
        """
        Set the number of rows displayed in the gallery.
        
        Parameters:
            value (int): Number of gallery rows to display.
        """
        self._gallery.rows = value
    
    @property
    def gallery_items_per_page(self):
        """
        Get the current number of gallery items displayed per page.
        
        Returns:
            int: Number of items shown on each gallery page.
        """
        return self._pagination.items_per_page
    
    @gallery_items_per_page.setter
    def gallery_items_per_page(self, value):
        """
        Set the number of gallery items displayed per page.
        
        Parameters:
            value (int): Number of items to display on each gallery page.
        """
        self._pagination.items_per_page = value
    
    @property
    def current_page(self):
        """Current page number (1-based)."""
        return self._pagination.current_page
    
    @current_page.setter
    def current_page(self, value):
        """
        Set the current gallery page in the pagination state.
        
        Parameters:
            value (int): The target page number (1-based).
        """
        self._pagination.current_page = value


    def calc_gallery_height(self):
        """
        Calculate the gallery height that preserves media aspect ratio for the current rows and columns.
        
        Returns:
            int: Computed gallery height in pixels.
        """
        return _settings.calc_gallery_height(self)

    def save_last_model(self, mod_id):
        """Saves the last used model to user config."""
        return _settings.save_last_model(self, mod_id)
    
    def move_model_up(self, selected_model, current_order_state):
        """
        Move a model one position earlier in the model ordering.
        
        Parameters:
            selected_model (str): The model ID to move up.
            current_order_state (list[str]): The current ordered list of model IDs.
        
        Returns:
            list[str]: The updated ordered list of model IDs after moving `selected_model` up; if `selected_model` is already first, returns the original order.
        """
        return _settings.move_model_up(self, selected_model, current_order_state)
    
    def move_model_down(self, selected_model, current_order_state):
        """Move selected model down in the order list."""
        return _settings.move_model_down(self, selected_model, current_order_state)

    def get_models_by_media_type(self, media_type: str) -> list:
        """
        Return model IDs that support the given media type.
        
        Parameters:
        	media_type (str): Media type to filter by (e.g., "Image" or "Video"); comparison is exact.
        
        Returns:
        	list: A list of model IDs whose configured `media_type` includes `media_type`. Models with no `media_type` configured are treated as supporting "Image". Both string and list configurations for `media_type` are supported.
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
        """
        Determine structural information about the current dataset's input paths.
        
        Returns:
            dict: Analysis of the dataset paths, including detected media counts, media types, and directory structure.
        """
        return _file_loader.analyze_input_paths(self.dataset)

    def create_zip(self, file_paths: list) -> str:
        """
        Create a zip archive containing the given files, preserving their relative directory structure when possible.
        
        Parameters:
            file_paths (list[str]): List of filesystem paths to include in the archive.
        
        Returns:
            str: Filesystem path to the created zip archive.
        """
        return _file_loader.create_zip(file_paths)

    def refresh_models(self):
        """
        Refresh the enabled-models selection from configuration and produce UI update objects for the model selectors.
        
        Updates the internal enabled models list from the configuration and chooses the current model (keeps the existing current model if still enabled, otherwise picks the first enabled model or None). Returns two Gradio update objects: one for the enabled-model selector (choices set to enabled models, value set to the chosen current model) and one for the all-models selector (choices set to all models, value set to the enabled models list).
        
        Returns:
            tuple: (enabled_models_update, all_models_update) where
                enabled_models_update: Gradio update for the enabled-model selector (`choices` = enabled models, `value` = selected current model or None),
                all_models_update: Gradio update for the all-models selector (`choices` = all models, `value` = list of enabled model IDs).
        """
        self._model_mgr.enabled_models = self.config_mgr.get_enabled_models()
        new_val = self.current_model_id if self.current_model_id in self.enabled_models else (self.enabled_models[0] if self.enabled_models else None)
        return gr.update(choices=self.enabled_models, value=new_val), gr.update(choices=self.models, value=self.enabled_models)

    def save_settings(self, vram, models_checked, gal_cols, gal_rows, limit_cnt, o_dir, o_fmt, over, rec, con, unload, pre, suf, clean, collapse, normalize, remove_cn, strip_loop, max_w, max_h, current_mod_id, model_ver, batch_sz, max_tok, settings_state, items_per_page):
        """Save settings from the main UI directly to user_config.yaml."""
        return _settings.save_settings(self, vram, models_checked, gal_cols, gal_rows, limit_cnt, o_dir, o_fmt, over, rec, con, unload, pre, suf, clean, collapse, normalize, remove_cn, strip_loop, max_w, max_h, current_mod_id, model_ver, batch_sz, max_tok, settings_state, items_per_page)
    
    def save_settings_simple(self, vram, system_ram, models_checked, gal_cols, gal_rows, unload, model_order_text, items_per_page):
        """
        Persist a simplified set of settings submitted from the Settings tab.
        
        Parameters:
            vram: Current VRAM selection or limit.
            system_ram: Current system RAM selection or limit.
            models_checked: Iterable of model IDs that are enabled/checked.
            gal_cols (int): Number of gallery columns to save.
            gal_rows (int): Number of gallery rows to save.
            unload: Whether to unload unused models when saving (boolean-like).
            model_order_text (str): Text representation of model ordering.
            items_per_page (int): Number of items to show per gallery page.
        
        Returns:
            dict: Mapping of setting names to their persisted/current UI values.
        """
        return _settings.save_settings_simple(self, vram, system_ram, models_checked, gal_cols, gal_rows, unload, model_order_text, items_per_page)
    
    def reset_to_defaults(self):
        """
        Reset all user-configurable settings to their default values by clearing persisted user configuration.
        """
        return _settings.reset_to_defaults(self)

    # User Presets Library (delegated to presets_logic module)

    def get_preset_eligible_models(self):
        """
        List models that support custom prompts.
        
        Returns:
            list: A list of model identifiers (strings) that accept custom prompt presets.
        """
        return _presets.get_preset_eligible_models(self)

    def get_user_presets_dataframe(self):
        """
        Provide user presets formatted for the settings dataframe.
        
        Returns:
            pandas.DataFrame: DataFrame where each row represents a user preset and columns match the schema expected by the settings UI.
        """
        return _presets.get_user_presets_dataframe(self)

    def save_user_preset(self, model_scope, name, text):
        """
        Save or update a user preset for the given model scope.
        
        Parameters:
            model_scope (str): Identifier or scope (model ID or 'global') the preset applies to.
            name (str): Name of the preset.
            text (str): Preset content (prompt or configuration text).
        """
        return _presets.save_user_preset(self, model_scope, name, text)

    def delete_user_preset(self, model_scope, name):
        """Delete a user preset."""
        return _presets.delete_user_preset(self, model_scope, name)


    def load_settings(self):
        """Force re-read from disk (reload) and return UI values."""
        return _settings.load_settings(self)

    def load_files(self, file_objs):
        """
        Handle drag-and-drop file uploads by persisting new files, adding them to the current dataset, and updating gallery state.
        
        Parameters:
            file_objs: Iterable of uploaded file-like objects from the UI.
        
        Returns:
            tuple: (gallery_data, None, current_page, total_label, pagination_visible)
                - gallery_data: List of visible gallery items for the current page.
                - None: Placeholder for a value intentionally unused by the UI.
                - current_page: The page index after loading (always 1 on success or when no files provided).
                - total_label: A textual label summarizing total items/pages.
                - pagination_visible: Boolean indicating whether pagination controls should be shown.
        """
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
        """
        Load images from a specified folder (or the configured default input directory when folder_path is empty), populate the app dataset, and reset pagination for the new dataset.
        
        Parameters:
            folder_path (str): Path to the folder to load. If empty or whitespace, the app's default input directory is used.
            recursive (bool): If True, include images from subdirectories when scanning.
            limit_count (int | str): Maximum number of images to keep from the scanned results; values <= 0 or non-integer input mean no limit.
        
        Returns:
            tuple:
                gallery_data (list): Visible gallery items for the current page (each item is a tuple of image path and caption).
                gallery_visibility_update (gradio.Update): Gradio update object controlling gallery visibility.
                current_page (int): Current page number after loading (always reset to 1 on successful load).
                total_label (str): Formatted label summarizing total items/pages for display in the UI.
                pagination_visible (bool): Whether pagination controls should be shown.
        Side effects:
            - Replaces the app's dataset with the scanned images.
            - Persists the dataset list to disk.
            - Updates current_input_path to the absolute path of the source folder.
            - Sets is_drag_and_drop to False and resets current_page to 1.
        """
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
        """
        Open the inspector for the selected gallery item.
        
        Parameters:
            evt (gr.SelectData): Selection event data from the gallery (contains selected index and value).
        
        Returns:
            dict: Updated UI state for the inspector and gallery (visibility, selected item/index, and related inspector fields).
        """
        return _inspector.open_inspector(self, evt)
    
    def remove_from_gallery(self):
        """Remove currently selected image from dataset (not from disk)."""
        return _inspector.remove_from_gallery(self)

    def save_and_close(self, caption):
        """
        Save the provided caption for the currently inspected item and close the inspector.
        
        Parameters:
            caption (str): Text to save as the caption for the selected item.
        """
        return _inspector.save_and_close(self, caption)

    def close_inspector(self):
        """Close the inspector panel."""
        return _inspector.close_inspector()
    
    def clear_gallery(self):
        """Clear the dataset, gallery, and close inspector."""
        return _inspector.clear_gallery(self)


    def update_model_ui(self, mod_id):
        """
        Compute UI state for the specified model.
        
        Parameters:
            mod_id (str): Identifier of the model to use for deriving UI state.
        
        Returns:
            dict: Mapping of UI component keys to their updated values for the interface.
        """
        return _model_logic.update_model_ui(self, mod_id)

    def apply_preset(self, mod_id, preset_name):
        """Apply a prompt preset."""
        return _model_logic.apply_preset(self, mod_id, preset_name)

    def auto_save_setting(self, key, value):
        """Saves a UI setting to user_config.yaml automatically."""
        return _settings.auto_save_setting(self, key, value)

    def save_model_defaults(self, mod_id, t, k, mt, rp):
        """
        Save the provided generation parameters as the user defaults for the given model.
        
        Parameters:
            mod_id (str): Identifier of the model to update.
            t (float): Temperature value to save as the model's default.
            k (int): Top-k sampling value to save as the model's default.
            mt (int): Maximum token (or output length) value to save as the model's default.
            rp (float): Repetition penalty value to save as the model's default.
        
        Returns:
            None
        """
        return _settings.save_model_defaults(self, mod_id, t, k, mt, rp)

    def reset_to_global(self, key):
        """
        Restore and return the global default value for the given settings key.
        
        Parameters:
            key (str): The settings key to reset.
        
        Returns:
            The global default value for the specified key.
        """
        return _settings.reset_to_global(self, key)

    def generate_cli_command(self, mod, args, skip_defaults=True):
        """
        Generate a CLI command for the given model and arguments.
        
        Parameters:
            mod: Model identifier or model configuration used to build the command.
            args: Mapping of CLI option names to values that will be applied to the command.
            skip_defaults (bool): If True, omit options that are set to their default values.
        
        Returns:
            cli_command (str): The generated command-line string.
        """
        return _generate_cli_command(self.config_mgr, mod, args, self.current_input_path, skip_defaults)

    
    def run_inference(self, mod, args):
        """
        Execute inference for the specified model over the current dataset.
        
        Parameters:
            mod: Identifier or model object specifying which model to run inference with.
            args: Argument object or mapping containing inference options and parameters.
        
        Returns:
            The value returned by the underlying inference runner (typically inference outputs or a result structure).
        """
        return _run_inference.run_inference(self, mod, args)
    
    # Multi-Model Captioning Methods
    
    def _sanitize_model_name(self, model_id: str) -> str:
        """
        Produce a filesystem-safe sanitized model name for use as an output filename extension.
        
        Parameters:
            model_id (str): Original model identifier (e.g., registry key or full model spec).
        
        Returns:
            str: Sanitized model name suitable for use as a filename/extension.
        """
        return _multi_model.sanitize_model_name(model_id)
    
    def load_multi_model_settings(self):
        """Load multi-model configuration from user_config.yaml"""
        return _multi_model.load_multi_model_settings(self.config_mgr, self.models)
    
    def save_multi_model_settings(self, *inputs):
        """
        Persist multi-model configuration into the user configuration file.
        
        Parameters:
            *inputs: Variable positional arguments that specify the multi-model configuration to save; these arguments are forwarded to the underlying multi-model settings handler and may vary by caller.
        
        Returns:
            The result of the save operation (implementation-defined), such as a status object or boolean indicating success.
        """
        return _multi_model.save_multi_model_settings(self.config_mgr, self.models, *inputs)
    
    def generate_multi_model_commands(self, *inputs):
        """
        Generate CLI commands for all enabled models using the provided inputs.
        
        Returns:
            commands (list[str]): List of CLI command strings, one entry per generated model command.
        """
        return _multi_model.generate_multi_model_commands(self, *inputs)
    
    def generate_multi_model_commands_with_settings(self, current_settings, checkboxes, formats):
        """
        Generate CLI commands for running multiple models using the provided UI settings.
        
        Parameters:
            current_settings (dict): Current UI settings that should be applied to each generated command.
            checkboxes (dict): Mapping of checkbox identifiers to booleans indicating which models/options are selected.
            formats (dict): Output format options (per-model or global) to include in the generated commands.
        
        Returns:
            List of CLI command strings configured for multi-model runs using the given settings and selections.
        """
        return _multi_model.generate_multi_model_commands_with_settings(self, current_settings, checkboxes, formats)
    
    def run_multi_model_inference(self, *inputs):
        """Run multiple models sequentially on the dataset."""
        return _multi_model.run_multi_model_inference(self, *inputs)