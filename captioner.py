"""
A Thousand Words - CLI Entry Point

This script provides a command-line interface that mirrors the GUI workflow exactly.
Arguments are dynamically generated from the feature registry.

Usage:
    py captioner.py --model smolVLM --input ./images --batch-size 4
    py captioner.py --help  # View all available options
"""
import os
import sys
import argparse
import logging

# Ensure imports work from root directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pathlib import Path
from src.core.config import ConfigManager
from src.core.registry import ModelRegistry
from src.core.loader import DataLoader
import src.features as feature_registry
from src.core.console_kit import console, Fore

# Simplified logging for CLI
# logging.basicConfig(level=logging.INFO, format='%(message)s')
# logger = logging.getLogger(__name__)


def build_argparser():
    """
    Dynamically builds argparse configuration from the feature registry.
    Arguments are grouped by category for readability.
    """
    parser = argparse.ArgumentParser(
        description="A Thousand Words CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  py captioner.py --model smolVLM
  py captioner.py --model smolVLM --input ./input --input-limit 10
  py captioner.py --model mimovl --recursive --batch-size 4
  py captioner.py --model smolVLM --model-version 256M --temperature 0.8
        """
    )
    
    # === REQUIRED ===
    required = parser.add_argument_group("Required")
    required.add_argument("--model", required=True, metavar="ID",
                          help="(str) Model ID. Example: smolVLM, mimovl")
    required.add_argument("--input", required=False, default="input", metavar="PATH",
                          help="(str) Input directory containing images. Default: 'input' folder")
    
    # === INPUT/OUTPUT ===
    io_group = parser.add_argument_group("Input/Output")
    io_group.add_argument("--output", default=None, metavar="PATH",
                          help="(str) Output directory. Default: same as input")
    io_group.add_argument("--input-limit", type=int, default=0, metavar="N",
                          help="(int) Limit captioning to the first N files. 0 = no limit (default)")
    
    # Define category mappings for features
    categories = {
        "Input/Output": ["output_format", "overwrite", "recursive", "output_json", "prefix", "suffix"],
        "Generation": ["batch_size", "max_tokens", "temperature", "top_k", "repetition_penalty"],
        "Model-Specific (varies by model)": ["model_version", "model_mode", "caption_length", "strip_thinking_tags", "flash_attention", "fps"],
        "Prompts": ["task_prompt", "system_prompt", "instruction_template", 
                    "prompt_source", "prompt_file_extension", "prompt_prefix", "prompt_suffix"],
        "Text Processing": ["clean_text", "collapse_newlines", 
                           "normalize_text", "remove_chinese", "strip_loop", "strip_contents_inside", "max_word_length"],
        "Image Processing": ["max_width", "max_height", "image_size", "min_visual_tokens", "max_visual_tokens", "min_video_tokens", "max_video_tokens"],
        "Tagger": ["threshold"],
        "Logging": ["print_console"],
    }
    
    # Create argument groups
    groups = {}
    for cat_name in categories:
        if cat_name != "Input/Output":  # Already created above
            groups[cat_name] = parser.add_argument_group(cat_name)
    groups["Input/Output"] = io_group
    
    # Map feature names to their category
    feature_to_category = {}
    for cat_name, feature_list in categories.items():
        for feat in feature_list:
            feature_to_category[feat] = cat_name
    
    # === Add features to their groups ===
    exclude_features = {"print_status"}
    all_features = feature_registry.get_all_features()
    
    for name, feature in all_features.items():
        if name in exclude_features:
            continue
            
        flag_name = f"--{name.replace('_', '-')}"
        config = feature.config
        desc = config.description if hasattr(config, 'description') else ""
        default = config.default_value
        
        # Get the group for this feature
        cat_name = feature_to_category.get(name, "Other")
        if cat_name not in groups:
            groups[cat_name] = parser.add_argument_group(cat_name)
        group = groups[cat_name]
        
        if config.gui_type == "checkbox":
            # Use store_const with default=None so omitted flags don't override model defaults
            # store_true returns False when omitted, which incorrectly overrides YAML defaults
            group.add_argument(flag_name, action="store_const", const=True, default=None,
                              help=f"(flag) {desc}")
        elif config.gui_type == "slider":
            val_type = float if isinstance(default, float) else int
            type_name = "float" if val_type == float else "int"
            group.add_argument(flag_name, type=val_type, default=None, metavar="N",
                              help=f"({type_name}) {desc} Default: {default}")
        else:
            example = f"Default: {default}" if default else ""
            group.add_argument(flag_name, type=str, default=None, metavar="STR",
                              help=f"(str) {desc} {example}")
    
    return parser


def main():
    parser = build_argparser()
    cli_args = parser.parse_args()
    
    # === Validate Model ===
    config_mgr = ConfigManager()
    available_models = config_mgr.list_models()
    
    if cli_args.model not in available_models:
        console.error(f"Unknown model: {cli_args.model}")
        console.print(f"Available models: {', '.join(available_models)}")
        sys.exit(1)
    
    # === Build Args Dict ===
    # Priority: CLI > Version-specific > Model defaults > Global defaults > Feature defaults
    global_defaults = config_mgr.get_global_settings()
    model_config = config_mgr.get_model_config(cli_args.model)
    
    # Get model_version from CLI first, then model defaults
    cli_version = getattr(cli_args, 'model_version', None)
    raw_defaults = model_config.get('defaults', {})
    
    # Determine version to use
    if cli_version:
        version = cli_version
    elif isinstance(raw_defaults, dict) and 'model_version' in raw_defaults:
        version = raw_defaults['model_version']
    else:
        version = None
    
    # Get version-specific defaults (this handles nested defaults structure)
    model_defaults = config_mgr.get_version_defaults(cli_args.model, version)
    
    args = {}
    all_features = feature_registry.get_all_features()
    
    for name, feature in all_features.items():
        cli_val = getattr(cli_args, name, None)
        if cli_val is not None:
            # CLI value takes priority
            args[name] = cli_val
        elif name in model_defaults:
            # Use version-specific or model default
            args[name] = model_defaults[name]
        elif name in global_defaults:
            # Use global.yaml default
            args[name] = global_defaults[name]
        else:
            # Fall back to feature's hardcoded default
            args[name] = feature.get_default()
    
    args["output_dir"] = cli_args.output if cli_args.output else ""
    args["gpu_vram"] = config_mgr.user_config.get('gpu_vram', 24)

    
    # === Load Dataset ===
    input_path = Path(cli_args.input)
    if not input_path.exists():
        console.error(f"Input path does not exist: {input_path.absolute()}")
        console.print(f"  Tip: Create an 'input' folder or specify a valid path with --input", force=True)
        sys.exit(1)
    
    recursive = args.get("recursive", False)
    dataset = DataLoader.scan_directory(str(input_path), recursive=recursive)
    
    # Apply input limit if specified
    input_limit = getattr(cli_args, 'input_limit', 0)
    if input_limit and input_limit > 0 and len(dataset) > input_limit:
        dataset.images = dataset.images[:input_limit]
        console.print(f"  Input limited to first {input_limit} files", force=True)
    
    if len(dataset) == 0:
        console.error("No images found in input directory.")
        sys.exit(1)
    
    # === Run Inference ===
    try:
        wrapper = ModelRegistry.load_wrapper(cli_args.model)
        wrapper.run(dataset, args)
    except Exception as e:
        console.error(f"Error during inference: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
