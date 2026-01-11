import os
import yaml
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List

class ConfigManager:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ConfigManager, cls).__new__(cls)
            cls._instance.init()
        return cls._instance

    def init(self):
        # root_dir is project root (parent of src/)
        self.root_dir = Path(__file__).parent.parent.parent
        self.configs_dir = self.root_dir / "src" / "config"
        self.user_data_dir = self.root_dir / "user"
        self.global_config_path = self.configs_dir / "global.yaml"
        self.user_config_path = self.user_data_dir / "user_config.yaml"
        self.models_dir = self.configs_dir / "models"

        # Create user data dir if it doesn't exist
        self.user_data_dir.mkdir(exist_ok=True)
        if not self.user_config_path.exists():
            # Create empty user config
            with open(self.user_config_path, 'w') as f:
                f.write("# User Overrides for Global Settings\n")

        self.global_config = self._load_yaml(self.global_config_path)
        self.user_config = self._load_yaml(self.user_config_path)
        
    
        # Load feature layout presets
        self.feature_layouts_path = self.configs_dir / "feature_layouts.yaml"
        self._feature_layouts = self._load_yaml(self.feature_layouts_path)
        
        # Validate model configurations
        self.validate_model_configs()

    def validate_model_configs(self):
        """
        Check all model configs for misconfigurations.
        Specifically checks if core preprocessing features are listed in the 'features' list.
        """
        # Features that should NOT be listed in model 'features' list because they are core/preprocessing
        # and handled automatically by the base wrapper.
        CORE_FEATURES_DENYLIST = [
            "max_width", "max_height", 
            "batch_size", "max_tokens", 
            "overwrite", "recursive", "output_format", 
            "prefix", "suffix", 
            "print_console", 
            "image_size" # If user considers this core/preprocessing
        ]
        
        # Only flag max_width/height as explicitly requested, and others that are definitely core
        # We'll use a safer list based on user's specific complaint + the obvious ones
        MISCONFIGURED_FEATURES = ["max_width", "max_height", "batch_size"]

        files = list(self.models_dir.glob("*.yaml"))
        for f in files:
            try:
                config = self._load_yaml(f)
                features_list = config.get('features', [])
                if not features_list:
                    continue
                    
                misconfigured = [feat for feat in features_list if feat in MISCONFIGURED_FEATURES]
                
                if misconfigured:
                    model_name = config.get('name', f.stem)
                    # Use print to ensure visibility in console as requested (logging might be filtered)
                    print(f"⚠️  Configuration Warning: Model '{model_name}' ({f.name}) lists core features in 'features': {', '.join(misconfigured)}. These should be removed.")
            except Exception as e:
                logging.error(f"Failed to validate config {f}: {e}")

    def get_feature_layout_presets(self) -> Dict[str, List[str]]:
        """Return the presets from feature_layouts.yaml."""
        # RELOAD from disk to support hot-updates
        self._feature_layouts = self._load_yaml(self.feature_layouts_path)
        return self._feature_layouts.get('presets', {})
    
    def resolve_feature_rows(self, model_id: str) -> List[List[str]]:
        """
        Resolve feature rows for a model.
        
        If model has feature_rows in its config, use those (resolving preset references).
        Otherwise, return None to indicate default layout should be used.
        
        Returns:
            List of feature lists (each representing a row), or None for default.
        """
        model_config = self.get_model_config(model_id)
        feature_rows = model_config.get('feature_rows')
        
        if feature_rows is None:
            return None  # Use default layout if key is missing
        
        if not feature_rows:
            return []  # Explicitly empty layout
        
        presets = self.get_feature_layout_presets()
        resolved = []
        
        for row in feature_rows:
            if isinstance(row, dict) and 'preset' in row:
                # Reference to a preset
                preset_name = row['preset']
                if preset_name in presets:
                    resolved.append(presets[preset_name])
                else:
                    logging.warning(f"Unknown feature layout preset: {preset_name}")
            elif isinstance(row, list):
                # Inline feature list
                resolved.append(row)
            else:
                logging.warning(f"Invalid feature_row format: {row}")
        
        return resolved

    def _load_yaml(self, path: Path) -> Dict[str, Any]:
        if not path.exists():
            return {}
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f) or {}
        except Exception as e:
            logging.error(f"Failed to load config {path}: {e}")
            return {}

    def _save_yaml(self, path: Path, data: Dict[str, Any]):
        try:
            with open(path, 'w', encoding='utf-8') as f:
                yaml.dump(data, f, default_flow_style=False)
        except Exception as e:
            logging.error(f"Failed to save config {path}: {e}")

    def get_global_settings(self) -> Dict[str, Any]:
        """Returns global settings with user overrides applied."""
        merged = self.global_config.copy()
        merged.update(self.user_config)
        return merged

    def get_model_config(self, model_id: str) -> Dict[str, Any]:
        """
        Returns the full configuration for a specific model.
        Priority: User Model Config > Default Model Config > User Global Config > Global Default Config
        """
        # Layer 1: Global Defaults
        final_config = self.global_config.copy()
        
        # Layer 2: User Global Overrides
        self._deep_update(final_config, self.user_config)

        # Layer 3: Base Model Config
        base_model_path = self.models_dir / f"{model_id}.yaml"
        base_config = self._load_yaml(base_model_path)
        self._deep_update(final_config, base_config)

        return final_config

    def get_model_defaults(self, model_id: str) -> Dict[str, Any]:
        """
        Returns the effective DEFAULT configuration for a model, WITHOUT user model-specific overrides.
        This represents the "baseline" state (Global Defaults -> User Global Overrides -> Base Model Config).
        Used for diffing to determine what to save.
        """
        # Layer 1: Global Defaults
        final_config = self.global_config.copy()
        
        # Layer 2: User Global Overrides
        self._deep_update(final_config, self.user_config)

        # Layer 3: Base Model Config
        base_model_path = self.models_dir / f"{model_id}.yaml"
        base_config = self._load_yaml(base_model_path)
        self._deep_update(final_config, base_config)
        
        return final_config

    def get_model_config(self, model_id: str) -> Dict[str, Any]:
        """
        Returns the full configuration for a specific model.
        Priority: User Model Settings (Saved Diffs) > Base Model Config > User Global Config > Global Default Config
        """
        # Start with the effective defaults
        final_config = self.get_model_defaults(model_id)
        
        # Layer 4: User Model-Specific Overrides (New Diff-Based System)
        # These are stored in user_config['model_settings'][model_id]
        user_model_settings = self.user_config.get('model_settings', {}).get(model_id, {})
        if user_model_settings:
            # We treat these as overrides on top of the defaults
            # However, for flat feature values, we want to ensure we don't just overwrite the 'defaults' dict structure
            # effectively, features are often stored in 'defaults' key in the yaml, but 'model_settings' might be flat key-values
            # or structured. Let's look at how they are saved.
            # In save_settings, we save them as flat key-values in model_settings[model_id].
            # But the model config expects them in 'defaults' dict for the actual values used by wrappers?
            # Actually, wrappers usually look at the top level or 'defaults' depending on implementation.
            # But `get_model_config` usually returns a structure where 'defaults' contains the values.
            # Let's check how `save_settings` saves them.
            # It saves them as flat keys: {'temperature': 0.9, ...}
            # So we should update the 'defaults' section of the final config with these values.
            
            if 'defaults' not in final_config:
                final_config['defaults'] = {}
            
            self._deep_update(final_config['defaults'], user_model_settings)
        
        return final_config

    def save_user_config(self):
        """Saves values currently in self.user_config to disk."""
        self._save_yaml(self.user_config_path, self.user_config)

    def save_user_model_config(self, model_id: str, data: Dict[str, Any]):
        """Deprecated: Model settings are now saved in user_config.yaml under model_settings key."""
        logger.warning(f"save_user_model_config is deprecated. Use model_settings in user_config.yaml instead.")

    def _deep_update(self, target: Dict[str, Any], source: Dict[str, Any]):
        """Robust deep merge for config dictionaries."""
        for key, value in source.items():
            if isinstance(value, dict) and key in target and isinstance(target[key], dict):
                self._deep_update(target[key], value)
            else:
                target[key] = value
    
    def _resolve_version_specific(self, data: Dict[str, Any], variant: Optional[str] = None) -> Dict[str, Any]:
        """
        Resolve version-specific configuration data.
        
        If data is nested by version (string keys), return the variant's data.
        If data is flat (non-string keys or values), return as-is.
        
        Args:
            data: Configuration data (could be flat or nested by version)
            variant: Model version string (e.g., "Alpha-Two", "Beta-One")
            
        Returns:
            Resolved flat dictionary
        """
        if not data or not isinstance(data, dict):
            return data or {}
        
        # Check if data is nested by version (has string keys with dict values)
        first_key = next(iter(data), None)
        if first_key is None:
            return {}
        
        # Robust check: To be version-nested, keys must be strings and values must be dicts.
        # If we see mixed types (some dicts, some not), assume it's a FLAT structure (e.g., polluted with other keys).
        is_nested = False
        
        if isinstance(first_key, str):
            first_val = data.get(first_key)
            if isinstance(first_val, dict):
                # Potential nested structure. Verify it's not just a coincidence (like 'defaults': {...})
                # Check if variant exists or if ALL top-level items look like versions?
                # Actually, simpler heuristic: If variant is requested and exists, great.
                # If variant NOT in data, we usually fallback.
                # BUT, if 'defaults' is a key in data, this is likely NOT a version map.
                # Let's rely on the assumption that version names won't be standard config keys like 'features' etc.
                is_nested = True
                
        if is_nested:
            # Version-nested structure
            if variant and variant in data:
                return data[variant]
            elif variant:
                 # Variant requested but not found.
                 # CRITICAL CHECK: Before falling back, ensure this really looks like version data.
                 # If we have keys like "max_tokens" (int) alongside "SomeVersion" (dict), it's mixed/broken.
                 # Treating it as flat is safer than picking a random dict.
                 
                 # Check if ANY value is NOT a dict
                 if any(not isinstance(v, dict) for v in data.values()):
                     # Mixed content! Treat as flat.
                     logging.warning(f"Config data appears mixed (versions + flat values). Treating as flat to conform with request for '{variant}'.")
                     return data
                 
                 # Fallback to first version
                 logging.debug(f"Variant '{variant}' not found in version-specific data. Using first available: '{first_key}'")
                 return data[first_key]
            else:
                 # No variant requested. If it's truly nested, we probably want the first one or default?
                 # If this function is called without variant on nested data, implies we want a default version.
                 return data[first_key]
        else:
            # Flat structure - return as-is
            return data
    
    def get_version_defaults(self, model_id: str, variant: Optional[str] = None) -> Dict[str, Any]:
        """
        Get version-specific defaults for a model.
        
        Args:
            model_id: Model identifier
            variant: Model version (e.g., "Alpha-Two", "Beta-One")
            
        Returns:
            Flat dictionary of default values for the specified version
        """
        config = self.get_model_config(model_id)
        defaults = config.get('defaults', {})
        
        # CRITICAL: Only apply version resolution if model actually supports versions
        # Check if model has model_versions defined
        has_versions = 'model_versions' in config and config['model_versions']
        
        if not has_versions:
            # Non-versioned model - return flat defaults as-is, ignore variant parameter
            return defaults if isinstance(defaults, dict) else {}
        
        # If no variant specified, try to get it from defaults itself
        if not variant and isinstance(defaults, dict):
            # For nested defaults, extract from structure
            # For flat defaults, check if defaults has model_version key
            if 'model_version' in defaults:
                variant = defaults['model_version']
            else:
                # If defaults is nested, get first version
                first_key = next(iter(defaults), None)
                if first_key and isinstance(first_key, str) and isinstance(defaults.get(first_key), dict):
                    variant = first_key
        
        return self._resolve_version_specific(defaults, variant)
    
    def get_version_prompt_presets(self, model_id: str, variant: Optional[str] = None) -> Dict[str, str]:
        """
        Get version-specific prompt presets for a model.
        
        Args:
            model_id: Model identifier
            variant: Model version (e.g., "Alpha-Two", "Beta-One")
            
        Returns:
            Dictionary mapping preset names to prompt strings for the specified version
        """
        config = self.get_model_config(model_id)
        presets = config.get('prompt_presets', {})
        
        # CRITICAL: Only apply version resolution if model actually supports versions
        has_versions = 'model_versions' in config and config['model_versions']
        
        resolved_presets = {}
        
        if not has_versions:
             # Non-versioned model - use flat presets as base
             resolved_presets = presets.copy() if isinstance(presets, dict) else {}
        else:
            # If no variant specified, use default from config
            if not variant:
                defaults = self.get_version_defaults(model_id)
                variant = defaults.get('model_version')
            
            resolved_presets = self._resolve_version_specific(presets, variant)
            # Ensure it's a dict (in case _resolve returns None)
            if not isinstance(resolved_presets, dict):
                resolved_presets = {}


        # Merge User Presets (Global + Model Specific)
        # Structure of user_prompt_presets in user_config:
        # List[Dict]: [{"model": "All Models"|"model_id", "name": "Name", "text": "Prompt"}, ...]
        
        user_presets_list = self.user_config.get("user_prompt_presets", [])
        if user_presets_list:
            # 1. Apply Global Presets (model == "All Models" or empty)
            for p in user_presets_list:
                p_model = p.get("model", "")
                p_name = p.get("name", "")
                p_text = p.get("text", "")
                
                if not p_name or not p_text:
                    continue
                    
                if p_model == "All Models" or not p_model:
                     resolved_presets[p_name] = p_text
            
            # 2. Apply Model Specific Presets (model == model_id)
            # These overwrite globals and built-ins
            for p in user_presets_list:
                p_model = p.get("model", "")
                p_name = p.get("name", "")
                p_text = p.get("text", "")
                
                if not p_name or not p_text:
                    continue
                
                if p_model == model_id:
                     resolved_presets[p_name] = p_text

        return resolved_presets

    def get_recommended_batch_size(self, model_id: str, vram_gb: int, variant: Optional[str] = None) -> int:
        config = self.get_model_config(model_id)
        vram_table = config.get("vram_table", {})
        
        # Determine which table to use
        target_table = {}
        
        if not vram_table:
            return 1

        # Check if table is flat (int keys) or nested (str keys)
        # We sample one key to check type
        first_key = next(iter(vram_table))
        
        if isinstance(first_key, int):
            # Flat table (Legacy behavior) - Applies to all variants
            target_table = vram_table
        elif isinstance(first_key, str):
            # Nested table - Pick specific variant
            # If no variant passed, try to get default from config
            if not variant:
                defaults = config.get('defaults', {})
                variant = defaults.get('model_version')
            
            if variant and variant in vram_table:
                target_table = vram_table[variant]
            else:
                # Robust fallback for VRAM table
                # Check if it looks like a flat table disguised with string keys (unlikely for VRAM, but possible)
                if all(isinstance(v, int) for v in vram_table.values()):
                     # It's actually a flat table with string keys (maybe "8GB" style keys but parsed weirdly?)
                     # But wait, logic above expects int keys for flat table.
                     pass

                # Fallback to first variant in table
                try:
                    first_variant = next(iter(vram_table))
                    target_table = vram_table[first_variant]
                    # Only log if we have a variant request that failed
                    if variant:
                        # Don't log warning if it's just a lookup miss on a clean table, 
                        # but here we are falling back to a likely completely different model size.
                        # logging.warning(f"Model '{model_id}': Variant '{variant}' VRAM table not found. Using '{first_variant}'.")
                        pass
                except Exception:
                     return 1
                     
        else:
            # Unknown format
            # logging.warning(f"Model '{model_id}': Unknown VRAM table format.")
            return 1
            
        # Find the highest VRAM key that is <= user's VRAM
        # Keys are VRAM reqs (int), Values are batch sizes.
        best_batch = 1
        sorted_keys = sorted([k for k in target_table.keys() if isinstance(k, int)])
        
        for k in sorted_keys:
            if vram_gb >= k:
                best_batch = target_table[k]
            else:
                break
        
        return best_batch

    def get_enabled_models(self) -> List[str]:
        all_models = self.list_models()
        disabled = self.user_config.get("disabled_models", [])
        return [m for m in all_models if m not in disabled]

    def set_model_state(self, model_id: str, enabled: bool):
        disabled = self.user_config.get("disabled_models", [])
        if not enabled:
            if model_id not in disabled:
                disabled.append(model_id)
        else:
            if model_id in disabled:
                disabled.remove(model_id)
        
        self.user_config["disabled_models"] = disabled
        self._save_yaml(self.user_config_path, self.user_config)

    def list_models(self) -> List[str]:
        """Returns a list of available model IDs based on config files, sorted by model_order."""
        files = list(self.models_dir.glob("*.yaml"))
        model_ids = []
        
        for f in files:
            model_id = f.stem
            # Check if wrapper is a placeholder
            if not self._is_placeholder_wrapper(model_id):
                model_ids.append(model_id)
        
        # Sort by model_order if it exists (check user config first, then global)
        user_model_order = self.user_config.get('model_order', [])
        global_model_order = self.global_config.get('model_order', [])
        model_order = user_model_order if user_model_order else global_model_order
        
        if model_order:
            # Create ordered list
            ordered = []
            unlisted = []
            
            # Add models in the specified order
            for model_id in model_order:
                if model_id in model_ids:
                    ordered.append(model_id)
            
            # Add models not in model_order (alphabetically sorted)
            for model_id in sorted(model_ids):
                if model_id not in ordered:
                    unlisted.append(model_id)
            
            return ordered + unlisted
        else:
            # Fallback to alphabetical if no model_order defined
            return sorted(model_ids)
    
    def _is_placeholder_wrapper(self, model_id: str) -> bool:
        """Check if a model wrapper is marked as a placeholder."""
        try:
            config = self.get_model_config(model_id)
            wrapper_path = config.get('wrapper_path', '')
            
            if not wrapper_path:
                return True  # No wrapper path means placeholder
            
            # Import the wrapper class
            module_path, class_name = wrapper_path.rsplit('.', 1)
            module = __import__(module_path, fromlist=[class_name])
            wrapper_class = getattr(module, class_name, None)
            
            if wrapper_class is None:
                return True
            
            # Check for PLACEHOLDER attribute
            return getattr(wrapper_class, 'PLACEHOLDER', False)
            
        except Exception as e:
            logging.warning(f"Could not check placeholder status for {model_id}: {e}")
            return True  # If we can't load it, treat as placeholder

