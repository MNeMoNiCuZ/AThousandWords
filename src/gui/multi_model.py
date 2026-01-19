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
        return app._get_gallery_data(), gr.update(visible=False), gr.update(visible=False)
    
    if not app.dataset or not app.dataset.images:
        gr.Warning("No images loaded!")
        return app._get_gallery_data(), gr.update(visible=False), gr.update(visible=False)
    
    settings = app.config_mgr.get_global_settings()
    
    # Server Mode Handling
    temp_dir_obj = None
    if app.is_server_mode:
        import tempfile
        import os
        # colorama imports removed (using module level)
        temp_dir_obj = tempfile.TemporaryDirectory(dir=os.environ.get("GRADIO_TEMP_DIR"), prefix="multi_model_")
        # Override output_dir in settings
        settings['output_dir'] = temp_dir_obj.name
        print(f"{Fore.CYAN}Server Mode: Multi-model output redirected to {temp_dir_obj.name}{Style.RESET_ALL}")
    
    start_time = time.time()
    models_completed = []
    total_captions = 0
    all_generated_files = []
    
    try:
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
            print(f"  {Fore.YELLOW}Output:{Style.RESET_ALL} {args.get('output_dir', 'Default')} (Format: {args['output_format']})")
            print("")
            
            try:
                # Expecting 4 values: (gallery, dl_btn, stats, files)
                ret = app.run_inference(model_id, args)
                if len(ret) >= 4:
                    generated_files = ret[3]
                    if generated_files:
                        all_generated_files.extend(generated_files)
                
                models_completed.append(model_id)
                print(f"  {Fore.GREEN} {model_id} completed successfully{Style.RESET_ALL}")
                total_captions += len(app.dataset.images) # Approximation, or use stats
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
        
        # Use standard tool finish logic
        from src.gui.inference import tool_finish_processing
        
        # We need to return gallery data + the 3 button updates from tool_finish_processing
        run_btn_upd, dl_grp_upd, dl_btn_upd = tool_finish_processing(
            button_text="Run Captioning",
            generated_files=all_generated_files if all_generated_files else None,
            zip_prefix="multi_model"
        )
        
        if temp_dir_obj:
            temp_dir_obj.cleanup()
            
        return app._get_gallery_data(), run_btn_upd, dl_grp_upd, dl_btn_upd

    except Exception as e:
        if temp_dir_obj:
            temp_dir_obj.cleanup()
        raise e
