"""
CaptioningApp class - Core application logic for the GUI.
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
from src.tools.wrapper_metadata import MetadataTool
from src.tools.wrapper_resize import ResizeTool
from .constants import GLOBAL_DEFAULTS, UI_CONFIG_MAP, filter_user_overrides
import src.features as feature_registry

logger = logging.getLogger("GUI")


class CaptioningApp:
    """Main application class for the A Thousand Words GUI."""
    
    def __init__(self):
        self.config_mgr = ConfigManager()
        
        # CRITICAL: Force re-read user_config from disk on every app creation
        # This is needed because ConfigManager is a singleton and retains startup values
        # When user refreshes the page, create_ui() creates a new CaptioningApp,
        # so we force the singleton to re-read the latest config from disk
        self.config_mgr.user_config = self.config_mgr._load_yaml(self.config_mgr.user_config_path)
        
        self.registry = ModelRegistry()
        self.dataset = Dataset()
        self.models = self.config_mgr.list_models()
        self.enabled_models = self.config_mgr.get_enabled_models()
        
        # Load last used model from user config, ensure it's valid
        last_model = self.config_mgr.user_config.get('last_model')
        
        # Validate and set current_model_id
        if last_model and isinstance(last_model, str) and last_model in self.enabled_models:
            self.current_model_id = last_model
        elif self.enabled_models:
            self.current_model_id = self.enabled_models[0]
        else:
            self.current_model_id = ""
        
        self.selected_path = None
        
        # Ensure default input dir exists
        self.default_input_dir = Path("input")
        self.default_input_dir.mkdir(exist_ok=True)
        
        # Use a temporary JSON file list as the default input source for CLI commands
        self.input_list_path = Path("user/input.json")
        self.current_input_path = str(self.input_list_path)
        
        # Gallery settings - use merged config (global + user)
        settings = self.config_mgr.get_global_settings()
        self.gallery_columns = settings['gallery_columns']
        self.gallery_rows = settings.get('gallery_rows', 3)
        # Migration: if gallery_rows missing but height exists, approximate rows
        if 'gallery_rows' not in settings and 'gallery_height' in settings:
             self.gallery_rows = max(1, int(settings['gallery_height']) // 210)
        
        # Pagination Settings
        self.gallery_items_per_page = getattr(settings, 'gallery_items_per_page', 50)
        # Check if key exists in settings (config dict)
        if 'gallery_items_per_page' in settings:
            self.gallery_items_per_page = int(settings['gallery_items_per_page'])
        else:
            self.gallery_items_per_page = 50 # Default
            
        self.current_page = 1
        
        # Track selected index for remove from gallery feature
        self.selected_index = None
        
        # Track if current input source is from drag & drop (for default output dir logic)
        self.is_drag_and_drop = False

    def calc_gallery_height(self):
        """
        Calculate dynamic gallery height based on rows and columns.
        Height = Rows * BaseRowHeight * (RefCols / CurrentCols)
        """
        if self.gallery_rows == 0:
            return None
            
        # --- TWEAKABLE VALUES ---
        BASE_ROW_HEIGHT = 245  # Height of one row when at 6 columns
        REF_COLS = 6           # Reference column count for the base height
        # ------------------------
        
        # Scaling factor: Fewer cols = taller images = more height needed per row
        scale = REF_COLS / max(1, self.gallery_columns)
        
        return int(self.gallery_rows * BASE_ROW_HEIGHT * scale)

    def save_last_model(self, mod_id):
        """Saves the last used model to user config."""
        if mod_id:
            self.current_model_id = mod_id
            self.config_mgr.user_config['last_model'] = mod_id
            self.config_mgr._save_yaml(self.config_mgr.user_config_path, self.config_mgr.user_config)
    
    def move_model_up(self, selected_model, current_order_state):
        """Move selected model up in the order list."""
        if not selected_model:
            gr.Warning("Please select a model first")
            return gr.update(), gr.update(), current_order_state
        
        # Use the state, not config (this allows multiple clicks without saving)
        current_order = list(current_order_state)
        
        if selected_model not in current_order:
            gr.Warning(f"Model {selected_model} not found in order list")
            return gr.update(), gr.update(), current_order_state
        
        idx = current_order.index(selected_model)
        if idx == 0:
            gr.Info("Model is already at the top")
            return gr.update(), gr.update(), current_order_state
        
        # Swap with previous
        current_order[idx], current_order[idx - 1] = current_order[idx - 1], current_order[idx]
        
        # Update radio, hidden textbox, and state
        return (
            gr.update(choices=current_order, value=selected_model), 
            gr.update(value="\n".join(current_order)),
            current_order  # Update state
        )
    
    def move_model_down(self, selected_model, current_order_state):
        """Move selected model down in the order list."""
        if not selected_model:
            gr.Warning("Please select a model first")
            return gr.update(), gr.update(), current_order_state
        
        # Use the state, not config (this allows multiple clicks without saving)
        current_order = list(current_order_state)
        
        if selected_model not in current_order:
            gr.Warning(f"Model {selected_model} not found in order list")
            return gr.update(), gr.update(), current_order_state
        
        idx = current_order.index(selected_model)
        if idx >= len(current_order) - 1:
            gr.Info("Model is already at the bottom")
            return gr.update(), gr.update(), current_order_state
        
        # Swap with next
        current_order[idx], current_order[idx + 1] = current_order[idx + 1], current_order[idx]
        
        # Update radio, hidden textbox, and state
        return (
            gr.update(choices=current_order, value=selected_model), 
            gr.update(value="\n".join(current_order)),
            current_order  # Update state
        )

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
    
        return filtered
    
    def analyze_input_paths(self):
        """
        Analyze the current dataset paths to determine structure.
        
        Returns:
            tuple: (common_root (Path|None), mixed_sources (bool), collisions (list))
        """
        if not self.dataset or not self.dataset.images:
            return None, False, []
            
        paths = [img.path.absolute() for img in self.dataset.images]
        
        # 1. Check for duplicates filenames (Collision detection)
        filename_map = {}
        collisions = []
        for p in paths:
            name = p.name
            if name in filename_map:
                # If we have seen this filename, check if it's the exact same file path (ok) or different (collision)
                if filename_map[name] != p:
                    collisions.append(name)
            else:
                filename_map[name] = p
        
        # Deduplicate collision list
        collisions = list(set(collisions))
        
        # 2. Find common root
        try:
            # os.path.commonpath raises ValueError if paths are on different drives (Windows)
            common_root = Path(os.path.commonpath(paths))
            
            # If common_root is the root drive itself (e.g. C:\), treat as mixed/no-shared-folder 
            # unless all files are literally in the root.
            # Actually, standard behavior is fine. If all in C:\Images, root is C:\Images.
            # If valid common root exists, check if it's meaningful (i.e. parent of all files)
            
            # If paths are on same drive but scattered (e.g. C:\A\1.png and C:\B\2.png) -> root C:\
            # This allows relative pathing: A\1.png and B\2.png. This is acceptable.
            
            mixed_sources = False
            
        except ValueError:
            # Different drives
            common_root = None
            mixed_sources = True
            
        return common_root, mixed_sources, collisions

    def create_zip(self, file_paths: list) -> str:
        """
        Create a zip file containing the specified files.
        Files are stored with relative paths if a common root exists among them,
        or flattened if not.
        
        Returns:
            str: Path to the created zip file
        """
        if not file_paths:
            return None
            
        # Analyze output structure to decide zip internal structure
        # (This logic mirrors the input analysis but for the OUTPUT files)
        try:
            paths = [Path(p).absolute() for p in file_paths]
            common_root = Path(os.path.commonpath(paths))
        except ValueError:
            common_root = None
            
        # Create temp zip
        try:
            # Create a named temp file that isn't deleted on close
            # We will return this path to Gradio, which handles cleanup or it stays in temp
            # Use system temp directory for zips to ensure cleanup
            import tempfile
            export_dir = Path(tempfile.gettempdir())
            
            import datetime
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            zip_name = f"captions_{timestamp}.zip"
            zip_path = export_dir / zip_name
            
            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
                for file_path in file_paths:
                    p = Path(file_path)
                    if p.exists():
                        if common_root:
                            # Verify arcname is relative to common_root
                            try:
                                arcname = p.relative_to(common_root)
                            except ValueError:
                                arcname = p.name
                        else:
                            arcname = p.name
                        
                        zf.write(p, arcname=arcname)
            
            return str(zip_path.absolute())
            
        except Exception as e:
            logger.error(f"Failed to create zip: {e}")
            return None

    def refresh_models(self):
        """Refresh the list of enabled models."""
        self.enabled_models = self.config_mgr.get_enabled_models()
        new_val = self.current_model_id if self.current_model_id in self.enabled_models else (self.enabled_models[0] if self.enabled_models else None)
        return gr.update(choices=self.enabled_models, value=new_val), gr.update(choices=self.models, value=self.enabled_models)

    def save_settings(self, vram, models_checked, gal_cols, gal_rows, limit_cnt, o_dir, o_fmt, over, rec, con, unload, pre, suf, clean, collapse, normalize, remove_cn, strip_loop, max_w, max_h, current_mod_id, model_ver, batch_sz, max_tok, settings_state, items_per_page):
        """Save settings from the main UI directly to user_config.yaml."""
        
        # 1. Update Global Config
        self.config_mgr.user_config.update({
            'gpu_vram': int(vram) if vram is not None else 24,
            'gallery_columns': int(gal_cols),
            'gallery_rows': int(gal_rows),
            'gallery_items_per_page': int(items_per_page) if items_per_page else 50,
            'limit_count': int(limit_cnt) if limit_cnt and str(limit_cnt).strip() else 0,
            'disabled_models': list(set(self.models) - set(models_checked)),
            'output_dir': o_dir,
            'output_format': o_fmt,
            'overwrite': over,
            'recursive': rec,
            'print_console': con,
            'prefix': pre,
            'suffix': suf,
            'clean_text': clean,
            'collapse_newlines': collapse,
            'normalize_text': normalize,
            'remove_chinese': remove_cn,
            'strip_loop': strip_loop,
            'max_width': int(max_w) if max_w else None,
            'max_height': int(max_h) if max_h else None,
            'unload_model': unload
        })
        
        # 2. Save model-specific features using DIFFING (only save non-default values)
        if current_mod_id and isinstance(current_mod_id, str):
            # Get effective defaults for this model (Global Defaults + Base Model Config + Version Overrides)
            # CRITICAL FIX: Use get_model_defaults() to get CLEAN defaults (without user saved overrides)
            # This ensures we diff against the YAML defaults, not the user's previously saved values!
            clean_config = self.config_mgr.get_model_defaults(current_mod_id)
            clean_defaults_raw = clean_config.get('defaults', {})
            
            # Resolve version-specific defaults from the clean config
            # We use the internal helper to ensure consistent logic
            model_defaults = self.config_mgr._resolve_version_specific(clean_defaults_raw, model_ver)
            
            # Get model config to check which features are supported
            model_config = self.config_mgr.get_model_config(current_mod_id)
            supported_features = set(model_config.get('features', []))
            
            # VALIDATION: Check if model supports versions
            model_versions = model_config.get('model_versions', {})
            model_supports_versions = bool(model_versions)
            valid_versions = list(model_versions.keys()) if model_supports_versions else []
            
            # Validate model_ver BEFORE using it
            if model_ver:
                if not model_supports_versions:
                    # Model doesn't support versions - clear the value and warn
                    logging.warning(f"Cannot save model_version '{model_ver}' for '{current_mod_id}' - model does not support versions")
                    gr.Warning(f"Model '{current_mod_id}' does not support versions. Version setting ignored.")
                    model_ver = None  # Clear it
                elif model_ver not in valid_versions:
                    # Invalid version for this model
                    logging.warning(f"Cannot save invalid model_version '{model_ver}' for '{current_mod_id}'. Valid: {valid_versions}")
                    gr.Warning(f"Invalid version '{model_ver}' for model '{current_mod_id}'. Valid versions: {', '.join(valid_versions)}")
                    model_ver = None  # Clear it
            
            # Prepare full settings dict (Static + Dynamic)
            # Start with static inputs passed explicitly
            all_feature_values = {
                'batch_size': batch_sz,
                'max_tokens': max_tok
            }
            
            # Only add model_version if it's valid
            if model_ver:
                all_feature_values['model_version'] = model_ver
            
            # Update with dynamic features from state
            if settings_state:
                all_feature_values.update(settings_state)
            
            # Special Handling: Prompt Presets "Custom" Logic
            # If prompt_presets is present but NOT "Custom", do NOT save task_prompt.
            # This allows the system to resolve the task_prompt from the preset on load.
            if 'prompt_presets' in all_feature_values:
                if all_feature_values['prompt_presets'] != "Custom":
                    if 'task_prompt' in all_feature_values:
                        del all_feature_values['task_prompt']
            
            # Universal features are always candidates for saving
            UNIVERSAL_FEATURES = {'batch_size', 'max_tokens', 'custom_task_prompt'}
            
            # Update user_config with the diff
            if 'model_settings' not in self.config_mgr.user_config:
                self.config_mgr.user_config['model_settings'] = {}
            
            # Get existing root to preserve other versions and existing data if they exist
            existing_root = self.config_mgr.user_config['model_settings'].get(current_mod_id, {})
            
            if model_ver:
                # Versioned Model: Enforce strict clean structure (model_version + versions ONLY)
                existing_versions = existing_root.get('versions', {})
                
                # Create CLEAN root structure
                new_root = {
                    'model_version': model_ver,
                    'versions': existing_versions
                }
                
                # Get OR Create the version dictionary to MERGE into
                if model_ver not in new_root['versions']:
                     new_root['versions'][model_ver] = {}
                
                target_dict = new_root['versions'][model_ver]
                
                logger.info(f"Saving Versioned Model: {current_mod_id} [{model_ver}]")
                logger.info(f"Active Features: {list(all_feature_values.keys())}")
                
                # MERGE STRATEGY: Update target_dict with active values, remove defaults
                # We iterate through all_feature_values (ACTIVE keys)
                # Keys NOT in all_feature_values are left untouched in target_dict (Preserved)
                for feature_name, value in all_feature_values.items():
                    if feature_name in supported_features or feature_name in UNIVERSAL_FEATURES or feature_name == 'model_version':
                        # Get default
                        default_val = model_defaults.get(feature_name)
                        if default_val is None:
                            feature = feature_registry.get_feature(feature_name)
                            if feature: default_val = feature.get_default()
                        
                        # Update or Remove
                        if value != default_val:
                            # Setting is non-default -> Update/Add
                            # Only log if it's a change or new
                            if target_dict.get(feature_name) != value:
                                logger.info(f"  Setting {feature_name}: {value} (Default: {default_val})")
                            target_dict[feature_name] = value
                        else:
                            # Setting matches default -> Remove
                            if feature_name in target_dict:
                                logger.info(f"  Resetting {feature_name} to default (Removing from saved)")
                                target_dict.pop(feature_name)
                                
                # Apply new root
                self.config_mgr.user_config['model_settings'][current_mod_id] = new_root
                
            else:
                # Flat Model: Save directly merge
                # Ensure root dict exists and uses existing one
                if current_mod_id not in self.config_mgr.user_config['model_settings']:
                    self.config_mgr.user_config['model_settings'][current_mod_id] = {}
                    
                target_dict = self.config_mgr.user_config['model_settings'][current_mod_id]
                
                logger.info(f"Saving Flat Model: {current_mod_id}")
                
                 # MERGE STRATEGY: Update target_dict
                for feature_name, value in all_feature_values.items():
                    if feature_name in supported_features or feature_name in UNIVERSAL_FEATURES or feature_name == 'model_version':
                        # Get default
                        default_val = model_defaults.get(feature_name)
                        if default_val is None:
                            feature = feature_registry.get_feature(feature_name)
                            if feature: default_val = feature.get_default()
                            
                        # Update or Remove
                        if value != default_val:
                            target_dict[feature_name] = value
                        else:
                            if feature_name in target_dict:
                                target_dict.pop(feature_name)
        
        # Filter to only save user overrides (values different from defaults)
        filtered_config = filter_user_overrides(self.config_mgr.user_config)
        
        # CRITICAL: Clean the in-memory config to verify it matches the filtered state
        self.config_mgr.user_config = filtered_config
        
        self.config_mgr._save_yaml(self.config_mgr.user_config_path, filtered_config)
        
        # Save last selected model
        if current_mod_id and isinstance(current_mod_id, str) and current_mod_id in self.enabled_models:
            self.save_last_model(current_mod_id)

        self.refresh_models()
        self.gallery_columns = int(gal_cols)
        self.gallery_rows = int(gal_rows)
        self.gallery_items_per_page = int(items_per_page) if items_per_page else 50
        

        gr.Info("Settings saved successfully!")
        return []
    
    def save_settings_simple(self, vram, system_ram, models_checked, gal_cols, gal_rows, unload, model_order_text, items_per_page):
        """Save settings from the Settings tab (simplified version)."""
        
        # Parse model_order from textbox
        model_order_lines = [line.strip() for line in model_order_text.split('\n') if line.strip()]
        
        # Validate: warn about invalid model IDs but don't block save
        all_models_set = set(self.models)
        invalid_models = [m for m in model_order_lines if m not in all_models_set]
        if invalid_models:
            gr.Warning(f"Unknown models in order list (ignored): {', '.join(invalid_models)}")
        
        # Filter to valid models only
        valid_model_order = [m for m in model_order_lines if m in all_models_set]
        
        self.config_mgr.user_config.update({
            'gpu_vram': vram,
            'system_ram': system_ram,  # Add System RAM
            'gallery_columns': gal_cols,
            'gallery_rows': gal_rows,
            'gallery_items_per_page': int(items_per_page) if items_per_page else 50,
            'disabled_models': list(set(self.models) - set(models_checked)),
            'unload_model': unload,
            'model_order': valid_model_order  # NEW: Save model order
        })
        
        # Update instance variables
        self.gallery_columns = int(gal_cols)
        self.gallery_rows = int(gal_rows)
        self.gallery_items_per_page = int(items_per_page) if items_per_page else 50
        
        # Filter and save
        filtered_config = filter_user_overrides(self.config_mgr.user_config)
        self.config_mgr._save_yaml(self.config_mgr.user_config_path, filtered_config)
        
        # CRITICAL: Reload the model list with the new order IMMEDIATELY
        # This forces ConfigManager to re-sort models based on the saved order
        self.models = self.config_mgr.list_models()
        self.enabled_models = self.config_mgr.get_enabled_models()
        
        # Prepare updates
        # 1. Main model dropdown and enabled models checkbox
        model_dd_update, model_chk_update = self.refresh_models()
        
        # 2. Multi-model components (checkboxes and formats)
        # We need to return updates for ALL models in self.models (order matters)
        # IMPORTANT: Now using the NEW order from self.models
        multi_updates = []
        for model_id in self.models:
            is_visible = model_id in self.enabled_models
            # Update checkbox visibility
            multi_updates.append(gr.update(visible=is_visible))
            
        # 3. Multi-model formats (same visibility as checkboxes)
        for model_id in self.models:
            is_visible = model_id in self.enabled_models
            multi_updates.append(gr.update(visible=is_visible))
            

        
        if gal_rows == 0:
            gr.Info("💡 Gallery hidden - refresh the page (F5) to apply")
        else:
            gr.Info("Settings saved successfully!")
            
        # Return merged list: [model_sel, models_chk, model_order_radio, *checkboxes, *formats]
        # Radio needs to be updated with the new order
        radio_update = gr.update(choices=self.models, value=None)
        return [model_dd_update, model_chk_update, radio_update] + multi_updates
    
    def reset_to_defaults(self):
        """Delete user_config.yaml to reset all settings to defaults."""
        import os
        
        user_config_path = self.config_mgr.user_config_path
        
        if user_config_path.exists():
            try:
                os.remove(user_config_path)

                return True, "Settings reset to defaults. Please refresh the page manually."
            except Exception as e:
                logger.error(f"Failed to delete user config: {e}")
                return False, f"Error: Failed to reset settings - {e}"
        else:
            return True, "No custom settings found. Already using defaults."

            return True, "No custom settings found. Already using defaults."

    # =========================================================================
    # USER PRESETS LIBRARY
    # =========================================================================

    def get_preset_eligible_models(self):
        """
        Return list of models that support custom prompts (eligible for presets).
        Logic mirrors model_info.py 'complex_prompt' check.
        """
        eligible = []
        for model_id in self.models:
            config = self.config_mgr.get_model_config(model_id)
            features = config.get('features', [])
            
            # Must have task_prompt feature AND (supports_custom_prompts != False)
            has_feature = 'task_prompt' in features
            explicitly_disabled = config.get('supports_custom_prompts', True) is False
            
            if has_feature and not explicitly_disabled:
                eligible.append(model_id)
        return eligible

    def get_user_presets_dataframe(self):
        """
        Get user presets formatted for the Settings Dataframe.
        Columns: [Model, Preset Name, Prompt Text, Delete]
        """
        presets = self.config_mgr.user_config.get("user_prompt_presets", [])
        
        # Sort Logic:
        # 1. Group by Model Order (User customized order)
        # 2. "All Models" at top
        # 3. Sort by Name within model
        
        # Helper for sort key
        model_order = self.config_mgr.user_config.get('model_order', self.models)
        
        def sort_key(p):
            p_model = p.get('model', 'All Models') or 'All Models'
            p_name = p.get('name', '')
            
            if p_model == 'All Models':
                m_idx = -1
            else:
                try:
                    m_idx = model_order.index(p_model)
                except ValueError:
                    m_idx = 9999 # Unknown models at end
            
            return (m_idx, p_name)

        sorted_presets = sorted(presets, key=sort_key)
        
        # Convert to List of Lists for Dataframe
        data = []
        for p in sorted_presets:
            p_model = p.get('model', 'All Models') or 'All Models'
            # Add "Delete" column with trash icon
            data.append([p_model, p.get('name', ''), p.get('text', ''), "🗑️"])
            
        return data

    def save_user_preset(self, model_scope, name, text):
        """
        Save (Upsert) a user preset.
        """
        if not name or not text:
             gr.Warning("Preset Name and Text are required.")
             return self.get_user_presets_dataframe(), gr.update(choices=["All Models"] + self.get_preset_eligible_models())
             
        # Normalize model scope
        if model_scope == "All Models":
            model_scope = "All Models"
            
        # Get current list
        presets = self.config_mgr.user_config.get("user_prompt_presets", [])
        
        # Remove existing if exists (Update)
        # Identify by (model, name)
        # Safe normalize p.get('model') to 'All Models' if None/Empty for comparison
        presets = [
            p for p in presets 
            if not ( (p.get('model') or 'All Models') == model_scope and p.get('name') == name )
        ]
        
        # Add new
        presets.append({
            "model": model_scope,
            "name": name,
            "text": text
        })
        
        # Save
        self.config_mgr.user_config["user_prompt_presets"] = presets
        self.config_mgr.save_user_config()
        
        gr.Info(f"Saved preset '{name}' for {model_scope}")
        
        # Return updated dataframe and ensuring choices are fresh
        return self.get_user_presets_dataframe(), gr.update(choices=["All Models"] + self.get_preset_eligible_models())

    def delete_user_preset(self, model_scope, name):
        """
        Delete a user preset.
        """
        if not name:
            gr.Warning("Select a preset to delete.")
            return self.get_user_presets_dataframe()
            
        presets = self.config_mgr.user_config.get("user_prompt_presets", [])
        
        # Remove
        new_presets = [
            p for p in presets 
            if not ( (p.get('model') or 'All Models') == model_scope and p.get('name') == name )
        ]
        
        if len(new_presets) == len(presets):
             gr.Warning(f"Preset '{name}' not found.")
             return self.get_user_presets_dataframe()
             
        self.config_mgr.user_config["user_prompt_presets"] = new_presets
        self.config_mgr.save_user_config()
        
        gr.Info(f"Deleted preset '{name}'")
        return self.get_user_presets_dataframe()


    def load_settings(self):
        """Force re-read user config from disk and return values for UI components."""
        # 1. CRITICAL: Force reload from disk EVERY TIME (page refresh doesn't restart server)
        # This ensures we always have the latest saved settings
        self.config_mgr.user_config = self.config_mgr._load_yaml(self.config_mgr.user_config_path)
        
        # 2. Get merged settings
        cfg = self.config_mgr.get_global_settings()
        
        # 3. Update app state where necessary
        self.gallery_columns = cfg.get('gallery_columns', 6)

        self.gallery_rows = cfg.get('gallery_rows', 3)
        # Migration logic same as init
        if 'gallery_rows' not in cfg and 'gallery_height' in cfg:
             self.gallery_rows = max(1, int(cfg['gallery_height']) // 210)
        
        # 4. Ensure current_model_id is valid
        if self.current_model_id not in self.enabled_models and self.enabled_models:
            self.current_model_id = self.enabled_models[0]
        
        # 5. Get model order (user config overrides global)
        current_order = self.config_mgr.user_config.get('model_order', self.config_mgr.global_config.get('model_order', self.models))
        
        # 6. Return values for all bound components
        # Order must match demo.load outputs in main.py
        
        # Safe Preset Loading
        try:
             presets_data = self.get_user_presets_dataframe()
             presets_choices = ["All Models"] + self.get_preset_eligible_models()
        except Exception as e:
             logger.error(f"Error loading presets: {e}")
             presets_data = []
             presets_choices = ["All Models"]

        return [
            # System
            cfg['gpu_vram'], 
            self.config_mgr.get_enabled_models(),
            self.gallery_columns,
            self.gallery_rows,
            "" if not cfg.get('limit_count') else cfg['limit_count'],
            
            # Output Group
            cfg['output_dir'], cfg['output_format'], cfg['overwrite'],
            
            # Options Group
            cfg['recursive'], cfg['print_console'], cfg.get('unload_model', True), cfg['clean_text'], cfg['collapse_newlines'],
            
            # Text Processing Group
            cfg['normalize_text'], cfg['remove_chinese'], cfg['strip_loop'],
            
            # Image Group
            cfg['max_width'], cfg['max_height'],
            
            # Pre/Suffix
            cfg['prefix'], cfg['suffix'],
            
            # Model Selection (must be valid model ID string)
            self.current_model_id if self.current_model_id else (self.enabled_models[0] if self.enabled_models else ""),
            
            # Model Order (hidden textbox)
            "\n".join(current_order),
            
            # Model Order Radio (NEW)
            gr.update(choices=current_order, value=None),
            
            # Gallery Items Per Page
            self.gallery_items_per_page,
            
            # Pagination Visibility
            self._get_pagination_vis()
        ]

    def load_files(self, file_objs):
        """Load files from drag-and-drop - ADDS to existing dataset.
        
        Persists uploaded files from temp directory to user_data/uploads
        to ensure stable paths for processing.
        """
        if not file_objs:
            return self._get_gallery_data(), None, 1, self.get_total_label(), self._get_pagination_vis()
        
        import shutil
        import uuid
        import os
        
        # Ensure uploads directory exists
        uploads_dir = Path("user/uploads")
        uploads_dir.mkdir(parents=True, exist_ok=True)
        
        persistent_files = []
        
        for file_obj in file_objs:
            # Handle Gradio objects or direct paths (strings)
            temp_path_str = file_obj.name if hasattr(file_obj, 'name') else file_obj
            temp_path = Path(temp_path_str)
            
            if not temp_path.exists():
                logger.warning(f"Temp file missing: {temp_path}")
                continue
            
            # Extract original filename by stripping Gradio's UUID prefix
            # Gradio naming pattern: {uuid}_{original_filename}
            # We need to detect if the filename has this pattern and extract the original name
            filename = temp_path.name
            
            # Check if filename starts with a UUID pattern (32 hex chars followed by underscore)
            import re
            uuid_pattern = re.compile(r'^[0-9a-f]{32}_(.+)$')
            match = uuid_pattern.match(filename)
            
            if match:
                # Extract original filename without Gradio's UUID prefix
                original_filename = match.group(1)
            else:
                # No UUID prefix found, use filename as-is
                original_filename = filename
            
            # Create persistent path using our own UUID to avoid collisions
            # Pattern: user_data/uploads/{uuid}_{original_clean_filename}
            safe_name = f"{uuid.uuid4().hex}_{original_filename}"
            dest_path = uploads_dir / safe_name
            
            try:
                # Copy file to persistent location
                shutil.copy2(temp_path, dest_path)
                persistent_files.append(str(dest_path.absolute()))
                
                # Debug log removed to reduce console noise
                # print(f"DEBUG: Persisted {original_filename} -> {dest_path}")
            except Exception as e:
                logger.error(f"Failed to persist file {temp_path}: {e}")
        
        if not persistent_files:
            return self._get_gallery_data(), None, 1, self.get_total_label(), self._get_pagination_vis()

        # Load new persistent files (not the temp ones)
        new_dataset = DataLoader.scan_directory(persistent_files)
        
        # Track existing paths to avoid duplicates
        existing_paths = {str(img.path) for img in self.dataset.images}
        
        # Add new images that aren't already in dataset
        for img in new_dataset.images:
            if str(img.path) not in existing_paths:
                self.dataset.images.append(img)
        
        # For drag and drop, we use the input.json list which points to these persistent files
        # We can set the current input path string to the uploads dir for display/logic if needed,
        # but the CLI generation uses the JSON list primarily.
        self.current_input_path = str(uploads_dir.absolute())
        self.is_drag_and_drop = True  # Flag this as a drag/drop source
        
        # Return gallery data and clear the file component so it can receive more files
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
        if not evt:
            return (
                gr.update(visible=False),  # inspector_group
                gr.update(selected="img_tab"),  # insp_tabs (default to image tab)
                None,  # insp_img
                None,  # insp_video
                ""  # insp_cap
            )
        
        # Adjust index for pagination
        page_offset = (self.current_page - 1) * self.gallery_items_per_page
        index = evt.index + page_offset
        
        self.selected_index = index  # Track selected index for remove feature
        
        if index < len(self.dataset.images):
            media_obj = self.dataset.images[index]
            self.selected_path = media_obj.path
            
            # Determine if this is a video or image and switch to appropriate tab
            if media_obj.is_video():
                # Switch to Video tab
                return (
                    gr.update(visible=True),  # inspector_group
                    gr.update(selected="vid_tab"),  # insp_tabs (select video tab by ID)
                    gr.update(value=None, visible=False),  # insp_img (Use update to force hide)
                    gr.update(value=str(media_obj.path), visible=True),  # insp_video
                    media_obj.caption  # insp_cap
                )
            else:
                # Switch to Image tab
                return (
                    gr.update(visible=True),  # inspector_group
                    gr.update(selected="img_tab"),  # insp_tabs (select image tab by ID)
                    gr.update(value=str(media_obj.path), visible=True),  # insp_img
                    gr.update(value=None, visible=False),  # insp_video
                    media_obj.caption  # insp_cap
                )
        
        return (
            gr.update(visible=False),  # inspector_group
            gr.update(selected="img_tab"),  # insp_tabs
            None,  # insp_img
            None,  # insp_video
            ""  # insp_cap
        )
    
    def remove_from_gallery(self):
        """Remove currently selected image from dataset (not from disk)."""
        if self.selected_index is not None and 0 <= self.selected_index < len(self.dataset.images):
            removed = self.dataset.images.pop(self.selected_index)

            self.selected_index = None
            self.selected_path = None
            return self._get_gallery_data(), gr.update(visible=False), self.current_page, self.get_total_label(), self._get_pagination_vis()
        else:
            gr.Warning("No image selected")
            return self._get_gallery_data(), gr.update(visible=True), self.current_page, self.get_total_label(), self._get_pagination_vis()

    def save_and_close(self, caption):
        """Save caption and close inspector."""
        if self.selected_path:
            for img in self.dataset.images:
                if img.path == self.selected_path:
                    img.update_caption(caption)
                    img.save_caption() # Keep original save_caption
                    break
        self._save_dataset_list() # Added call to save dataset list
        return self._get_gallery_data(), gr.update(visible=False)

    def close_inspector(self):
        """Close the inspector panel."""
        return gr.update(visible=False)
    
    def clear_gallery(self):
        """Clear the dataset, gallery, and close inspector."""
        self.dataset = Dataset()
        self.dataset = Dataset()
        self.dataset = Dataset()
        self.selected_path = None
        self.current_page = 1

        return [], gr.update(visible=False), 1, self.get_total_label(), self._get_pagination_vis()

    # --- Tools ---
    def run_metadata(self, src, upd, pre, suf, clean, collapse, norm, out_dir, ext):
        """Run metadata extraction tool."""
        print(f"--- Starting Metadata Extraction ---")
        print(f"Source: {src}, Overwrite: {upd}")
        print(f"Output: {out_dir if out_dir else 'In-place'}, Extension: {ext}")
        
        result = MetadataTool().apply_to_dataset(self.dataset, src, upd, 
                                      prefix=pre, suffix=suf,
                                      clean=clean, collapse=collapse, normalize=norm,
                                      output_dir=out_dir, extension=ext)
        print(f"Result: {result}")
        print("--- Metadata Extraction Complete ---")
        gr.Info(result)
        return self._get_gallery_data()

    def run_resize(self, dim, out_dir, pre, suf, ext, overwrite):
        """Run resize tool."""
        print(f"--- Starting Image Resize ---")
        print(f"Max Dimension: {dim}px")
        print(f"Output: {out_dir if out_dir else 'In-place'}, Overwrite: {overwrite}")
        print(f"Name format: {pre}[name]{suf}{ext}")
        
        result = ResizeTool.apply_to_dataset(self.dataset, int(dim), 
                                           output_dir=out_dir, prefix=pre, suffix=suf, 
                                           extension=ext, overwrite=overwrite)
        print(f"Result: {result}")
        print("--- Resize Complete ---")
        gr.Info(result)
        return self._get_gallery_data()

    def update_model_ui(self, mod_id):
        """Update UI components based on selected model."""
        if not mod_id:
            return [gr.update()] * 11
        
        cfg = self.config_mgr.get_model_config(mod_id)
        caps = cfg.get("capabilities", {})
        defs = cfg.get("defaults", {})
        
        # Get merged settings for VRAM
        settings = self.config_mgr.get_global_settings()
        vram = settings['gpu_vram']
        rec_batch = self.config_mgr.get_recommended_batch_size(mod_id, vram)
        
        # Get VRAM table/info for tooltip
        recs = cfg.get("vram_recommendations") or cfg.get("vram_table") or {}
        recommendation_list = "\n".join([f"- {k}GB: Batch {v}" for k, v in recs.items()])
        info_tooltip = f"Number of images to process in parallel.\nVRAM recommendations for this model:\n{recommendation_list}"

        # Helper for prompt presets
        presets = cfg.get("prompt_presets", {})
        preset_keys = list(presets.keys()) if presets else []
        
        # Handle strip_thinking_tags visibility
        has_thinking = caps.get("include_thinking", False)
        
        
        # Robustly resolve default template
        default_template = defs.get("prompt_presets")
        if default_template and default_template not in preset_keys:
            print(f"Warning: Default template '{default_template}' not found in presets for {mod_id}. Using first available.")
            default_template = None
        
        final_template_value = default_template or (preset_keys[0] if presets else None)

        return [
            gr.update(visible=caps.get("temperature", True), value=defs.get("temperature", 0.5)),
            gr.update(visible=caps.get("top_k", True), value=defs.get("top_k", 40)),
            gr.update(visible=caps.get("max_tokens", True), value=defs.get("max_tokens", 300)),
            gr.update(visible=caps.get("repetition_penalty", True), value=defs.get("repetition_penalty", 1.1)),
            gr.update(visible=caps.get("detail_mode", False), value=defs.get("detail_mode", 1)),
            gr.update(visible=has_thinking, value=defs.get("include_thinking", True)),
            gr.update(visible=has_thinking, value=defs.get("strip_thinking_tags", True)),
            gr.update(visible=bool(presets), choices=preset_keys, value=final_template_value),
            gr.update(value=rec_batch, info=info_tooltip),
            gr.update(value=defs.get("system_prompt", "")),
            gr.update(value=defs.get("task_prompt", ""))
        ]

    def apply_preset(self, mod_id, preset_name):
        """Apply a prompt preset."""
        if not mod_id or not preset_name:
            return gr.update()
        cfg = self.config_mgr.get_model_config(mod_id)
        presets = cfg.get("prompt_presets", {})
        if preset_name in presets:
            return gr.update(value=presets[preset_name])
        return gr.update()

    def auto_save_setting(self, key, value):
        """Saves a UI setting to user_config.yaml automatically."""
        if key in UI_CONFIG_MAP:
            config_key = UI_CONFIG_MAP[key]
            self.config_mgr.user_config[config_key] = value
            self.config_mgr._save_yaml(self.config_mgr.user_config_path, self.config_mgr.user_config)


    def save_model_defaults(self, mod_id, t, k, mt, rp):
        """Saves current generation settings as user defaults for the model."""
        if not mod_id:
            return "No model selected."
        data = {
            "defaults": {
                "temperature": t,
                "top_k": k,
                "max_tokens": mt,
                "repetition_penalty": rp
            }
        }
        self.config_mgr.save_user_model_config(mod_id, data)
        return f"Saved defaults for {mod_id}"

    def reset_to_global(self, key):
        """Returns the global default for a specific key."""
        return self.config_mgr.global_config.get(key, "")

    def generate_cli_command(self, mod, args, skip_defaults=True):
        """Generate CLI command without running inference.
        
        Args:
            mod: Model ID
            args: Dictionary of arguments  
            skip_defaults: If True, skip parameters that match defaults. If False, show all parameters.
        
        Returns the CLI command that would be executed.
        """
        # Build CLI command for copy/paste
        cli_parts = ["py captioner.py"]
        cli_parts.append(f"--model {mod}")
        cli_parts.append(f"--input \"input\"")
        
        # Output directory (only if set)
        out_dir = args.get("output_dir", "")
        if out_dir:
            cli_parts.append(f"--output \"{out_dir}\"")
        
        # Dynamic feature arguments
        from src.features import get_all_features, FEATURE_REGISTRY
        
        model_config = self.config_mgr.get_model_config(mod)
        # Extract feature names safely (features can be strings or dicts)
        features_list = model_config.get("features", [])
        model_feature_names = set()
        for f in features_list:
            if isinstance(f, dict):
                model_feature_names.add(f['name'])
            else:
                model_feature_names.add(f)
        
        # Get all features and filter to those relevant for this model
        all_feature_instances = get_all_features()
        
        # Define GLOBAL features: universal features that apply to ALL or MOST models
        # These are always considered, even if not in the model's YAML features list
        global_feature_names = {
            # Core Universal (every model)
            "batch_size", "max_tokens",
            
            # Output Control (universal)
            "overwrite", "recursive", "output_format", "output_json",
            "prefix", "suffix",
            
            # Text Processing (universal)
            "clean_text", "collapse_newlines", "normalize_text", 
            "remove_chinese", "strip_loop",
            
            # Image Processing (universal)
            "max_width", "max_height",
            
            # Console/Logging (universal)
            "print_console", "print_status",
            
            # System Control (universal)
            "unload_model",
            
            # Prompt Features (very common, used by most VLM models)
            # These are critical and should be included for compatibility
            "task_prompt", "system_prompt", "prompt_presets",
            "prompt_source", "prompt_file_extension", 
            "prompt_prefix", "prompt_suffix"
        }
        
        # Collect features: ONLY those in global list OR model's feature list
        relevant_features = {}
        for name, feature in all_feature_instances.items():
            if name in global_feature_names or name in model_feature_names:
                relevant_features[name] = feature
        
        # Build CLI arguments
        cli_args = []
        cli_args.append(f"--model {mod}")
        
        # Add input directory (Always use the temp json list which reflects GUI state)
        cli_args.append(f"--input \"{self.current_input_path}\"")
        
        # Add all feature arguments (skip GUI-only features)
        for name, feature in relevant_features.items():
            # Skip features that shouldn't appear in CLI
            if not feature.config.include_in_cli:
                continue
            
            # Get prompt_source to determine which prompt-related features to include
            prompt_source = args.get('prompt_source', 'Instruction Presets')
            
            # Skip prompt_file_extension, prompt_prefix, prompt_suffix if not using "From File" or "From Metadata"
            if name in ['prompt_file_extension', 'prompt_prefix', 'prompt_suffix']:
                if prompt_source == 'Instruction Presets':
                    continue  # These are only relevant for "From File" and "From Metadata" modes
            
            # task_prompt and system_prompt are always included as fallbacks/instructions
            # No filtering needed.
                
            value = args.get(name, feature.get_default())
            
            # Skip if value matches default AND skip_defaults is True (don't clutter CLI)
            # EXCEPTION: For boolean flags, if the value is True, we MUST include the flag
            # because the CLI uses 'store_true' (defaulting to False if omitted).
            # So if default is True and we skip it, the CLI sees it as False.
            is_bool = isinstance(value, bool)
            if skip_defaults and value == feature.get_default() and not is_bool:
                continue
            
            # Format based on type
            if is_bool:
                if value:  # Only add flag if True (regardless of default)
                    cli_args.append(f"--{name.replace('_', '-')}")
            elif isinstance(value, (int, float)):
                cli_args.append(f"--{name.replace('_', '-')} {value}")
            elif isinstance(value, str):
                # When skip_defaults=False, include all strings (even empty) for completeness
                # When skip_defaults=True, only include non-empty strings
                if not skip_defaults or value:
                    cli_args.append(f"--{name.replace('_', '-')} \"{value}\"")
        
        return f"python captioner.py {' '.join(cli_args)}"
    
        return "python captioner.py {' '.join(cli_args)}"
    
    def run_inference(self, mod, args):
        """Run inference on the dataset."""
        # Get global VRAM setting if not present - use merged config
        if "gpu_vram" not in args:
            settings = self.config_mgr.get_global_settings()
            args["gpu_vram"] = settings['gpu_vram']

        # --- DRAG & DROP SAFETY CHECK ---
        # If input comes from our internal uploads folder (Drag/Drop) AND no output dir is specified,
        # we MUST default to a safe "output" folder to prevent saving into the temp/hidden uploads dir.
        # Users expect Drag/Drop -> /output (or explicit path), never internal user_data.
        
        # Use explicit flag instead of path checking (which gets overwritten by input.json path)
        if self.is_drag_and_drop and not args.get("output_dir"):
            print("Detected Drag & Drop input with no output directory. Defaulting to 'output'.")
            args["output_dir"] = "output"

        # Generate CLI command using shared method - NEVER skip defaults for execution
        # User wants to see exactly what is running, including all global settings
        cli_command = self.generate_cli_command(mod, args, skip_defaults=False)

        # Get batch size for batch count calculation
        bs = args.get("batch_size", 1)
        
        # Calculate batch count
        total_images = len(self.dataset.images) if self.dataset and self.dataset.images else 0
        batch_count = (total_images + int(bs) - 1) // int(bs) if total_images > 0 else 0
        
        # --- PATH ANALYSIS ---
        common_root, mixed_sources, collisions = self.analyze_input_paths()
        
        # 1. Check for Collisions
        if collisions:
            # If we have filename collisions, we MUST have a common structure to disambiguate
            # unless the user is outputting to different folders? output_dir is single.
            # If collisions exist and we flatten (no common root or mixed drives), files will overwrite.
            # Even with common root, if we don't use it (flatten), they collide.
            # But we plan to mirror structure if common_root exists.
            
            # Critical Error Condition: Collisions exist AND (sources are mixed drives OR user forced flat output?)
            # Actually, if we have collisions, we rely on the relative path from common root to separate them.
            # If common_root is None (mixed drives), we CANNOT mirror structure easily in a single output root.
            # (We could map D:\A.png -> Output\D\A.png, but that's ugly/complex).
            
            if not common_root:
                msg = f"❌ Filename collisions detected across different drives/roots: {', '.join(collisions[:3])}... Cannot save safely to single output. Please process these separately."
                gr.Warning(msg)
                print(msg)
                gr.Warning(msg)
                print(msg)
                return self._get_gallery_data(), gr.update(visible=False), {}
        
        # 2. Mixed Source Warning
        if mixed_sources:
             msg = "⚠️ Warning: Inputs are from different drives/locations. Output files will be flattened into the output folder."
             gr.Warning(msg)
             print(msg)

        # Pass common_root to args so wrapper can use it for relative pathing
        if common_root:
            args['input_root'] = common_root

        # Print to console
        print(f"--- Starting Inference Run for {mod} ---")
        print("")
        print(cli_command)
        print("")
        
        if not self.dataset or not self.dataset.images:
            msg = "⚠️ Warning: No images found. Please load a folder or add images to the 'Input Source' before running."
            print(msg)
            gr.Warning("No images found. Please load a folder or add images to the 'Input Source' before running.")
            return self._get_gallery_data(), gr.update(visible=False), {}

        try:
            # Temporary dataset slicing for runtime limit
            limit_count = args.get('limit_count', 0)
            run_dataset = self.dataset
            
            try:
                limit = int(limit_count)
                if limit > 0 and len(self.dataset.images) > limit:
                    print(f"Limiting to first {limit} files (from {len(self.dataset.images)} total).")
                    # Create a shallow copy with sliced images list
                    import copy
                    run_dataset = copy.copy(self.dataset)
                    run_dataset.images = self.dataset.images[:limit]
            except (ValueError, TypeError):
                pass
        
            # Wrapper run now returns (list of generated file paths, stats)
            generated_files, stats = ModelRegistry.load_wrapper(mod).run(run_dataset, args)
            
            if isinstance(generated_files, list) and generated_files:
                # Successfully generated files - create ZIP
                zip_path = self.create_zip(generated_files)
                if zip_path:
                    # Update download button
                    return self._get_gallery_data(), gr.update(visible=True, value=zip_path), stats
            
            return self._get_gallery_data(), gr.update(visible=False), stats
            
        except RuntimeError as e:
            # Catch OOM RuntimeError from wrapper without re-raising
            if "out of memory" in str(e).lower() or "vram" in str(e).lower():
                # OOM error - show warning popup (no traceback)
                gr.Warning("🔴 CUDA OUT OF MEMORY - Reduce batch size, resize images in general settings, use another model, or close other GPU intensive processes to free up memory")
                return self._get_gallery_data(), gr.update(visible=False), {}
            else:
                # Other RuntimeError - re-raise with traceback
                import traceback
                traceback.print_exc()
                raise gr.Error(f"Processing failed: {str(e)}")
                
        except Exception as e:
            # Handle other exceptions
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
            raise gr.Error(f"Processing failed: {str(e)}")
    
    # Multi-Model Captioning Methods
    
    def _sanitize_model_name(self, model_id: str) -> str:
        """Convert model ID to sanitized output format extension."""
        import re
        # Remove special characters, keep alphanumeric and underscores
        sanitized = re.sub(r'[^a-zA-Z0-9_]', '', model_id)
        return sanitized  # Just the name, no . prefix or .txt suffix
    
    def load_multi_model_settings(self):
        """Load multi-model configuration from user_config.yaml"""
        # Force reload to get latest settings
        self.config_mgr.user_config = self.config_mgr._load_yaml(self.config_mgr.user_config_path)
        
        multi_config = self.config_mgr.user_config.get('multi_model', {})
        enabled_models = multi_config.get('enabled_models', [])
        output_formats = multi_config.get('output_formats', {})
        
        # Return settings for all models
        settings = []
        for model_id in self.models:
            enabled = model_id in enabled_models
            format_ext = output_formats.get(model_id, self._sanitize_model_name(model_id))
            settings.append((enabled, format_ext))
        
        return settings
    
    def save_multi_model_settings(self, *inputs):
        """Save multi-model configuration to user_config."""
        # Split inputs: first half are checkboxes, second half are format inputs
        num_models = len(self.models)
        checkboxes = inputs[:num_models]
        formats = inputs[num_models:]
        
        # Build enabled list and format dict
        enabled = []
        format_dict = {}
        
        for i, model_id in enumerate(self.models):
            if checkboxes[i]:
                enabled.append(model_id)
            
            # Diffing Logic:
            # 1. Get default format (sanitized model ID)
            default_format = self._sanitize_model_name(model_id)
            user_format = formats[i]
            
            # 2. Only add to dict if DIFFERENT from default
            # We treat empty string as "use default", so we generally expect a value.
            # If the user clears the box, the UI sends empty string? 
            # If it's empty, we probably shouldn't save it either, or treat it as default.
            # The UI placeholder is "extension".
            
            if user_format and user_format != default_format:
                format_dict[model_id] = user_format
        
        # Save to config
        self.config_mgr.user_config['multi_model'] = {
            'enabled_models': enabled,
            'output_formats': format_dict
        }
        
        # Filter and save
        from .constants import filter_user_overrides
        filtered_config = filter_user_overrides(self.config_mgr.user_config)
        self.config_mgr._save_yaml(self.config_mgr.user_config_path, filtered_config)
        

        gr.Info("Multi-model settings saved!")
        return []
    
    def generate_multi_model_commands(self, *inputs):
        """Generate CLI commands for all enabled models."""
        # Parse inputs
        num_models = len(self.models)
        checkboxes = inputs[:num_models]
        formats = inputs[num_models:]
        
        enabled_models = [self.models[i] for i in range(num_models) if checkboxes[i]]
        
        if not enabled_models:
            return "No models selected!"
        
        # Get current global settings
        global_settings = self.config_mgr.get_global_settings()
        
        commands = []
        for model_id in enabled_models:
            # Get model config and its saved defaults
            model_config = self.config_mgr.get_model_config(model_id)
            model_defaults = model_config.get('defaults', {})
            
            # Merge: global settings + model defaults + custom output format
            args = {
                **global_settings,  # Global settings (overwrite, recursive, etc.)
                **model_defaults,  # Model-specific defaults from config
                'output_format': formats[self.models.index(model_id)]  # Custom format
            }
            
            cmd = self.generate_cli_command(model_id, args)
            commands.append(f"# Model: {model_id}")
            commands.append(cmd)
            commands.append("")
        
        return "\n".join(commands)
    
    def generate_multi_model_commands_with_settings(self, current_settings, checkboxes, formats):
        """Generate CLI commands using current UI settings."""
        enabled_models = [self.models[i] for i in range(len(self.models)) if checkboxes[i]]
        
        if not enabled_models:
            return "No models selected!"
        
        # Get global settings and merge with current UI settings
        global_settings = self.config_mgr.get_global_settings()
        global_settings.update(current_settings)  # Override with current UI values
        
        commands = []
        for model_id in enabled_models:
            # Get model config and defaults
            model_config = self.config_mgr.get_model_config(model_id)
            model_defaults = model_config.get('defaults', {})
            
            # Merge: global settings (with UI overrides) + model defaults
            # NOTE: Do NOT override output_format - we want 100% reusable commands
            args = {
                **global_settings,  # Global + current UI settings
                **model_defaults,  # Model-specific defaults
            }
            
            # Pass skip_defaults=False to show FULL command with all parameters
            cmd = self.generate_cli_command(model_id, args, skip_defaults=False)
            commands.append(f"# Model: {model_id}")
            commands.append(cmd)
            commands.append("")
        
        return "\n".join(commands)
    
    def run_multi_model_inference(self, *inputs):
        """Run multiple models sequentially on the dataset."""
        import time
        
        # Parse inputs
        # Last argument is limit_count
        limit_count = inputs[-1]
        
        num_models = len(self.models)
        checkboxes = inputs[:num_models]
        formats = inputs[num_models:-1] # Everything else except last item
        
        enabled_models = [self.models[i] for i in range(num_models) if checkboxes[i]]
        
        if not enabled_models:
            gr.Warning("No models selected!")
            gallery_data = self._get_gallery_data()
            return gallery_data
        
        if not self.dataset or not self.dataset.images:
            gr.Warning("No images loaded!")
            gallery_data = self._get_gallery_data()
            return gallery_data
        # Get current global settings
        settings = self.config_mgr.get_global_settings()
        
        # Statistics tracking
        start_time = time.time()
        models_completed = []
        total_captions = 0
        total_chars = 0
        total_words = 0
        
        # Run each model sequentially
        for model_idx, model_id in enumerate(enabled_models):
            # Build args with custom output format
            args = {
                **settings,
                'output_format': formats[self.models.index(model_id)],
                'overwrite': settings['overwrite'],
                'print_console': settings['print_console'],
                'limit_count': limit_count # Pass runtime limit to inference
            }
            
            print(f"\n{Fore.MAGENTA}{'='*60}{Style.RESET_ALL}")
            print(f"{Fore.MAGENTA}  MULTI-MODEL PROGRESS: Model {model_idx+1}/{len(enabled_models)}{Style.RESET_ALL}")
            print(f"{Fore.MAGENTA}{'='*60}{Style.RESET_ALL}")
            print(f"  {Fore.YELLOW}Model:{Style.RESET_ALL}  {model_id}")
            print(f"  {Fore.YELLOW}Output:{Style.RESET_ALL} {args['output_format']}")
            print("")
            
            try:
                # Ignore stats for multi-model individual runs (we aggregate differently or rely on console)
                _, _, stats = self.run_inference(model_id, args)
                models_completed.append(model_id)
                print(f"  {Fore.GREEN} {model_id} completed successfully{Style.RESET_ALL}")
                
                # Count captions generated for statistics
                total_captions += len(self.dataset.images)
                
            except Exception as e:
                gr.Warning(f"Model {model_id} failed: {str(e)}")
                print(f"  {Fore.RED}✗ {model_id} failed: {e}{Style.RESET_ALL}")
        
        # Calculate final statistics
        elapsed_time = time.time() - start_time
        
        # Print enhanced finish report with colors (Magenta to match multi-model theme)
        print(f"\n{Fore.MAGENTA}{'='*60}{Style.RESET_ALL}")
        print(f"{Fore.MAGENTA}         FINISHED MULTI-MODEL PROCESSING{Style.RESET_ALL}")
        print(f"{Fore.MAGENTA}{'='*60}{Style.RESET_ALL}")
        print(f"  {Fore.YELLOW}Models Run:{Style.RESET_ALL}  {', '.join(models_completed)}")
        print(f"  {Fore.YELLOW}Images:{Style.RESET_ALL}      {len(self.dataset.images)}")
        print(f"  {Fore.YELLOW}Total Runs:{Style.RESET_ALL}  {total_captions}")
        print(f"  {Fore.YELLOW}Time:{Style.RESET_ALL}        {elapsed_time:.1f}s")
        print(f"{Fore.MAGENTA}{'='*60}{Style.RESET_ALL}\n")
        
        msg = (
            f"Models: {len(models_completed)}/{len(enabled_models)} completed<br>"
            f"Images: {len(self.dataset.images)}<br>"
            f"Total Captions: {total_captions}<br>"
            f"Time: {elapsed_time:.1f}s"
        )
        gr.Info(msg, title="Multi-Model Complete")

        gallery_data = self._get_gallery_data()
        return gallery_data
