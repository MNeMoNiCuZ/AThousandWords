"""Multi-model captioning logic."""

import time
import gradio as gr
import logging
import re

logger = logging.getLogger("GUI.MultiModel")

# Colorama fallback
try:
    from colorama import Fore, Style
except ImportError:
    class Fore:
        CYAN = YELLOW = GREEN = MAGENTA = RED = WHITE = RESET = ''
    class Style:
        BRIGHT = DIM = RESET_ALL = ''


def sanitize_model_name(model_id: str) -> str:
    """
    Produce a filesystem- and extension-safe string from a model identifier by removing all characters except letters, digits, and underscores.
    
    Returns:
        sanitized (str): The input `model_id` with all characters other than A–Z, a–z, 0–9, and underscore removed.
    """
    return re.sub(r'[^a-zA-Z0-9_]', '', model_id)


def load_multi_model_settings(config_mgr, models):
    """
    Load per-model multi-model settings from the user's configuration.
    
    If the `multi_model` section or its keys are missing, defaults are used: models are disabled by default and each model's output format defaults to a sanitized version of its model id.
    
    Parameters:
        models (Iterable[str]): Ordered collection of model identifiers to retrieve settings for.
    
    Returns:
        list[tuple[bool, str]]: A list of (enabled, format_ext) tuples in the same order as `models`.
            `enabled` is `True` when the model is listed in the saved `enabled_models`; otherwise `False`.
            `format_ext` is the configured output format for the model or the sanitized model id when not set.
    """
    config_mgr.user_config = config_mgr._load_yaml(config_mgr.user_config_path)
    
    multi_config = config_mgr.user_config.get('multi_model', {})
    enabled_models = multi_config.get('enabled_models', [])
    output_formats = multi_config.get('output_formats', {})
    
    settings = []
    for model_id in models:
        enabled = model_id in enabled_models
        format_ext = output_formats.get(model_id, sanitize_model_name(model_id))
        settings.append((enabled, format_ext))
    
    return settings


def save_multi_model_settings(config_mgr, models, *inputs):
    """
    Persist multi-model UI settings into the user's configuration and save them to disk.
    
    Updates config_mgr.user_config['multi_model'] with:
    - enabled_models: list of model IDs whose corresponding checkbox input is truthy
    - output_formats: mapping of model_id -> user-specified format for formats that differ from the default sanitized model name
    
    Parameters:
        models (list[str]): Ordered list of model IDs corresponding to the UI rows.
        *inputs: Layout is the first len(models) values as checkbox booleans (enabled flags),
            followed by len(models) values of per-model output format strings.
    
    Returns:
        list: An empty list (returned for Gradio callback compatibility).
    """
    from .constants import filter_user_overrides
    
    num_models = len(models)
    checkboxes = inputs[:num_models]
    formats = inputs[num_models:]
    
    enabled = []
    format_dict = {}
    
    for i, model_id in enumerate(models):
        if checkboxes[i]:
            enabled.append(model_id)
        
        default_format = sanitize_model_name(model_id)
        user_format = formats[i]
        
        if user_format and user_format != default_format:
            format_dict[model_id] = user_format
    
    config_mgr.user_config['multi_model'] = {
        'enabled_models': enabled,
        'output_formats': format_dict
    }
    
    filtered_config = filter_user_overrides(config_mgr.user_config)
    config_mgr._save_yaml(config_mgr.user_config_path, filtered_config)
    
    gr.Info("Multi-model settings saved!")
    return []


def generate_multi_model_commands(app, *inputs):
    """
    Generate a combined CLI script for all enabled models from the current UI inputs.
    
    For each enabled model this constructs a CLI command using the global settings, the model's configured defaults, and the model's specified output format, then returns all commands concatenated with newline separators.
    
    Parameters:
    	inputs (tuple): UI values where the first len(app.models) items are selection checkboxes (truthy to enable a model) and the remaining items are per-model output format strings.
    
    Returns:
    	str: The combined CLI commands separated by newlines, or the string "No models selected!" if no models are enabled.
    """
    num_models = len(app.models)
    checkboxes = inputs[:num_models]
    formats = inputs[num_models:]
    
    enabled_models = [app.models[i] for i in range(num_models) if checkboxes[i]]
    
    if not enabled_models:
        return "No models selected!"
    
    settings = app.config_mgr.get_global_settings()
    
    commands = []
    for model_idx, model_id in enumerate(enabled_models):
        model_config = app.config_mgr.get_model_config(model_id)
        model_defaults = model_config.get('defaults', {})
        
        args = {
            **settings,
            **model_defaults,
            'output_format': formats[app.models.index(model_id)]
        }
        
        cmd = app.generate_cli_command(model_id, args, skip_defaults=False)
        commands.append(f"# Model: {model_id}")
        commands.append(cmd)
        commands.append("")
    
    return "\n".join(commands)


def generate_multi_model_commands_with_settings(app, current_settings, checkboxes, formats):
    """
    Generate a combined CLI script for the enabled models using the provided UI settings.
    
    Parameters:
        app: Application instance providing model list, config manager, and CLI generation helpers.
        current_settings (dict): Overrides to apply to global settings when building per-model arguments.
        checkboxes (Iterable[bool]): Sequence indicating which models (by index in app.models) are enabled.
        formats (Iterable[str]): Per-model output format strings (unused by this function; accepted for API compatibility).
    
    Returns:
        str: A newline-separated string containing a header and CLI command for each enabled model, or
        "No models selected!" if no checkboxes are truthy.
    """
    enabled_models = [app.models[i] for i, c in enumerate(checkboxes) if c]
    
    if not enabled_models:
        return "No models selected!"
    
    global_settings = app.config_mgr.get_global_settings()
    global_settings.update(current_settings)
    
    commands = []
    for model_id in enabled_models:
        model_config = app.config_mgr.get_model_config(model_id)
        model_defaults = model_config.get('defaults', {})
        
        args = {
            **global_settings,
            **model_defaults,
        }
        
        cmd = app.generate_cli_command(model_id, args, skip_defaults=False)
        commands.append(f"# Model: {model_id}")
        commands.append(cmd)
        commands.append("")
    
    return "\n".join(commands)


def run_multi_model_inference(app, *inputs):
    """
    Execute selected models sequentially on the currently loaded dataset and update the UI with progress and a final summary.
    
    Inputs:
    - The varargs `inputs` must be laid out as:
      1) One boolean checkbox per model in `app.models` (selects which models to run),
      2) One output format string per model (same order as `app.models`),
      3) A final `limit_count` value (applies to each model run).
    
    Behavior:
    - If no models are selected or no images are loaded, a Gradio warning is emitted and the gallery is returned unchanged.
    - Runs each enabled model in order, building per-model args from global settings plus the model's selected output format and the provided `limit_count`.
    - Individual model failures are caught and reported; remaining models continue to run.
    - Prints console progress for each model and a final summary, and shows a Gradio Info panel titled "Multi-Model Complete".
    
    Returns:
    - The gallery data from `app._get_gallery_data()` to refresh the UI.
    """
    limit_count = inputs[-1]
    
    num_models = len(app.models)
    checkboxes = inputs[:num_models]
    formats = inputs[num_models:-1]
    
    enabled_models = [app.models[i] for i in range(num_models) if checkboxes[i]]
    
    if not enabled_models:
        gr.Warning("No models selected!")
        return app._get_gallery_data()
    
    if not app.dataset or not app.dataset.images:
        gr.Warning("No images loaded!")
        return app._get_gallery_data()
    
    settings = app.config_mgr.get_global_settings()
    
    start_time = time.time()
    models_completed = []
    total_captions = 0
    
    for model_idx, model_id in enumerate(enabled_models):
        args = {
            **settings,
            'output_format': formats[app.models.index(model_id)],
            'overwrite': settings['overwrite'],
            'print_console': settings['print_console'],
            'limit_count': limit_count
        }
        
        print(f"\n{Fore.MAGENTA}{'='*60}{Style.RESET_ALL}")
        print(f"{Fore.MAGENTA}  MULTI-MODEL PROGRESS: Model {model_idx+1}/{len(enabled_models)}{Style.RESET_ALL}")
        print(f"{Fore.MAGENTA}{'='*60}{Style.RESET_ALL}")
        print(f"  {Fore.YELLOW}Model:{Style.RESET_ALL}  {model_id}")
        print(f"  {Fore.YELLOW}Output:{Style.RESET_ALL} {args['output_format']}")
        print("")
        
        try:
            _, _, stats = app.run_inference(model_id, args)
            models_completed.append(model_id)
            print(f"  {Fore.GREEN} {model_id} completed successfully{Style.RESET_ALL}")
            total_captions += len(app.dataset.images)
        except Exception as e:
            gr.Warning(f"Model {model_id} failed: {str(e)}")
            print(f"  {Fore.RED}✗ {model_id} failed: {e}{Style.RESET_ALL}")
    
    elapsed_time = time.time() - start_time
    
    print(f"\n{Fore.MAGENTA}{'='*60}{Style.RESET_ALL}")
    print(f"{Fore.MAGENTA}         FINISHED MULTI-MODEL PROCESSING{Style.RESET_ALL}")
    print(f"{Fore.MAGENTA}{'='*60}{Style.RESET_ALL}")
    print(f"  {Fore.YELLOW}Models Run:{Style.RESET_ALL}  {', '.join(models_completed)}")
    print(f"  {Fore.YELLOW}Images:{Style.RESET_ALL}      {len(app.dataset.images)}")
    print(f"  {Fore.YELLOW}Total Runs:{Style.RESET_ALL}  {total_captions}")
    print(f"  {Fore.YELLOW}Time:{Style.RESET_ALL}        {elapsed_time:.1f}s")
    print(f"{Fore.MAGENTA}{'='*60}{Style.RESET_ALL}\n")
    
    msg = (
        f"Models: {len(models_completed)}/{len(enabled_models)} completed<br>"
        f"Images: {len(app.dataset.images)}<br>"
        f"Total Captions: {total_captions}<br>"
        f"Time: {elapsed_time:.1f}s"
    )
    gr.Info(msg, title="Multi-Model Complete")
    
    return app._get_gallery_data()