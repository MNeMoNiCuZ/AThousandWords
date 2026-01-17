"""CLI command generation for the GUI."""

import logging
from src.features import get_all_features

logger = logging.getLogger("GUI.CLI")


def generate_cli_command(config_mgr, mod, args, current_input_path, skip_defaults=True):
    """
    Generate CLI command without running inference.
    
    Args:
        config_mgr: ConfigManager instance
        mod: Model ID
        args: Dictionary of arguments
        current_input_path: Current input path for dataset
        skip_defaults: If True, skip parameters that match defaults
        
    Returns:
        str: CLI command string
    """
    model_config = config_mgr.get_model_config(mod)
    
    # Extract feature names safely
    features_list = model_config.get("features", [])
    model_feature_names = set()
    for f in features_list:
        if isinstance(f, dict):
            model_feature_names.add(f['name'])
        else:
            model_feature_names.add(f)
    
    # Get all features and filter to those relevant for this model
    all_feature_instances = get_all_features()
    
    # Global features that apply to ALL or MOST models
    global_feature_names = {
        "batch_size", "max_tokens",
        "overwrite", "recursive", "output_format", "output_json",
        "prefix", "suffix",
        "clean_text", "collapse_newlines", "normalize_text", 
        "remove_chinese", "strip_loop",
        "max_width", "max_height",
        "print_console", "print_status",
        "unload_model",
        "task_prompt", "system_prompt", "prompt_presets",
        "prompt_source", "prompt_file_extension", 
        "prompt_prefix", "prompt_suffix"
    }
    
    # Collect relevant features
    relevant_features = {}
    for name, feature in all_feature_instances.items():
        if name in global_feature_names or name in model_feature_names:
            relevant_features[name] = feature
    
    # Build CLI arguments
    cli_args = []
    cli_args.append(f"--model {mod}")
    cli_args.append(f"--input \"{current_input_path}\"")
    
    # Output directory
    out_dir = args.get("output_dir", "")
    if out_dir:
        cli_args.append(f"--output \"{out_dir}\"")
    
    # Add all feature arguments
    for name, feature in relevant_features.items():
        if not feature.config.include_in_cli:
            continue
        
        # Skip prompt file features if using Instruction Presets
        prompt_source = args.get('prompt_source', 'Instruction Presets')
        if name in ['prompt_file_extension', 'prompt_prefix', 'prompt_suffix']:
            if prompt_source == 'Instruction Presets':
                continue
        
        value = args.get(name, feature.get_default())
        
        # Skip if matches default (except booleans - need explicit flag)
        is_bool = isinstance(value, bool)
        if skip_defaults and value == feature.get_default() and not is_bool:
            continue
        
        # Format based on type
        if is_bool:
            if value:
                cli_args.append(f"--{name.replace('_', '-')}")
        elif isinstance(value, (int, float)):
            cli_args.append(f"--{name.replace('_', '-')} {value}")
        elif isinstance(value, str):
            if not skip_defaults or value:
                cli_args.append(f"--{name.replace('_', '-')} \"{value}\"")
    
    return f"python captioner.py {' '.join(cli_args)}"
