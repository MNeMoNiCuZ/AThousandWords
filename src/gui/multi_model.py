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
    """Convert model ID to sanitized output format extension."""
    return re.sub(r'[^a-zA-Z0-9_]', '', model_id)


def load_multi_model_settings(config_mgr, models):
    """Load multi-model configuration from user_config.yaml"""
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
    """Save multi-model configuration to user_config."""
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
    """Generate CLI commands for all enabled models."""
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
    """Generate CLI commands using current UI settings."""
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
            'output_format': formats[app.models.index(model_id)]
        }
        
        cmd = app.generate_cli_command(model_id, args, skip_defaults=False)
        commands.append(f"# Model: {model_id}")
        commands.append(cmd)
        commands.append("")
    
    return "\n".join(commands)


def run_multi_model_inference(app, *inputs):
    """Run multiple models sequentially on the dataset."""
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
            print(f"  {Fore.RED}[ERR] {model_id} failed: {e}{Style.RESET_ALL}")
    
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
