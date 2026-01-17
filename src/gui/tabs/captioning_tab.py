"""Captioning tab factory - General and Model Settings accordions."""

from pathlib import Path
import gradio as gr
import src.features as feature_registry


def create_general_settings_accordion(app):
    """Create the General Settings accordion with all processing options."""
    cfg = app.config_mgr.get_global_settings()
    components = {}
    
    with gr.Accordion("⚙️ General Settings", open=False):
        with gr.Row():
            out_dir = gr.Textbox(label="Output Folder", value=cfg['output_dir'], placeholder="Leave empty for same folder as input", info="Directory for captions. Leave empty to save alongside input images.")
            out_fmt = gr.Textbox(label="Output Format", value=cfg['output_format'], info="File extension (e.g., txt, json, caption)")
            
            rec_cfg = feature_registry.get_feature("recursive").get_gui_config()
            g_recursive = gr.Checkbox(label=rec_cfg['label'], value=cfg['recursive'], info=rec_cfg['info'])
            
            ov_cfg = feature_registry.get_feature("overwrite").get_gui_config()
            g_over = gr.Checkbox(label=ov_cfg['label'], value=cfg['overwrite'], info=ov_cfg['info'])

        with gr.Row():
            norm_cfg = feature_registry.get_feature("normalize_text").get_gui_config()
            g_normalize = gr.Checkbox(label=norm_cfg['label'], value=cfg['normalize_text'], info=norm_cfg['info'])
            
            coll_cfg = feature_registry.get_feature("collapse_newlines").get_gui_config()
            g_collapse = gr.Checkbox(label=coll_cfg['label'], value=cfg['collapse_newlines'], info=coll_cfg['info'])

            clean_cfg = feature_registry.get_feature("clean_text").get_gui_config()
            g_clean = gr.Checkbox(label=clean_cfg['label'], value=cfg['clean_text'], info=clean_cfg['info'])

            con_cfg = feature_registry.get_feature("print_console").get_gui_config()
            g_console = gr.Checkbox(label=con_cfg['label'], value=cfg['print_console'], info=con_cfg['info'])

        with gr.Row():
            slp_cfg = feature_registry.get_feature("strip_loop").get_gui_config()
            g_strip_loop = gr.Checkbox(label=slp_cfg['label'], value=cfg['strip_loop'], info=slp_cfg['info'])
            
            rem_cfg = feature_registry.get_feature("remove_chinese").get_gui_config()
            g_remove_chinese = gr.Checkbox(label=rem_cfg['label'], value=cfg['remove_chinese'], info=rem_cfg['info'])
            
            mw_cfg = feature_registry.get_feature("max_width").get_gui_config()
            g_max_width = gr.Number(label=mw_cfg['label'], value=cfg['max_width'], info=mw_cfg['info'])
            
            mh_cfg = feature_registry.get_feature("max_height").get_gui_config()
            g_max_height = gr.Number(label=mh_cfg['label'], value=cfg['max_height'], info=mh_cfg['info'])

        with gr.Row():
            pre_cfg = feature_registry.get_feature("prefix").get_gui_config()
            pre_text = gr.Textbox(label=pre_cfg['label'], value=cfg['prefix'], placeholder="photo of, ", info=pre_cfg['info'])
            
            suf_cfg = feature_registry.get_feature("suffix").get_gui_config()
            suf_text = gr.Textbox(label=suf_cfg['label'], value=cfg['suffix'], placeholder=", high quality", info=suf_cfg['info'])

    components = {
        "out_dir": out_dir,
        "out_fmt": out_fmt,
        "recursive": g_recursive,
        "overwrite": g_over,
        "normalize": g_normalize,
        "collapse": g_collapse,
        "clean": g_clean,
        "console": g_console,
        "strip_loop": g_strip_loop,
        "remove_chinese": g_remove_chinese,
        "max_width": g_max_width,
        "max_height": g_max_height,
        "prefix": pre_text,
        "suffix": suf_text
    }
    return components


def create_model_settings_accordion(app, get_model_description_html_fn):
    """Create the Model Settings accordion with model selection and dynamic features."""
    from ..handlers import create_update_model_settings_handler, create_inference_wrapper
    from ..renderers.features import render_features_content
    from ..logic.model_logic import resolve_model_values, get_initial_model_state
    
    components = {}
    
    with gr.Accordion("🤖 Model Settings", open=True):
        initial_description_html = get_model_description_html_fn(app, app.current_model_id)
        model_description = gr.Markdown(value=initial_description_html, elem_classes="model-description")
        
        presets_tracker = gr.State(value=0)
        
        models_chk = gr.CheckboxGroup(
            choices=app.models, 
            value=app.enabled_models, 
            label="Enabled Models", 
            info="Select which models appear in the model dropdown",
            render=False
        )

        with gr.Row():
            media_type_filter = gr.Dropdown(
                choices=["Image", "Video"], value="Image", label="Media Type",
                info="Filter models by type", scale=0, min_width=150
            )
            model_sel = gr.Dropdown(
                app.enabled_models, label="Model", value=app.current_model_id,
                interactive=True, allow_custom_value=False, filterable=False,
                info="Select captioning model", scale=1, min_width=200
            )
            
            mv_cfg = feature_registry.get_feature("model_version").get_gui_config() if feature_registry.get_feature("model_version") else {}
            model_version_dropdown = gr.Dropdown(choices=[], visible=False, label=mv_cfg.get('label', 'Model Version'), info=mv_cfg.get('info', 'Select model version'), scale=1)
            
            bs_cfg = feature_registry.get_feature("batch_size").get_gui_config() if feature_registry.get_feature("batch_size") else {}
            batch_size_input = gr.Number(label=bs_cfg.get('label', 'Batch Size'), value=bs_cfg.get('value', 1), visible=False, info="Images per batch", scale=1, min_width=180)
            
            mt_cfg = feature_registry.get_feature("max_tokens").get_gui_config() if feature_registry.get_feature("max_tokens") else {}
            max_tokens_input = gr.Number(label=mt_cfg.get('label', 'Max Tokens'), value=mt_cfg.get('value', 512), visible=False, info="Max output length", scale=0, min_width=150)

        model_feature_components = {
            'model_version': model_version_dropdown,
            'batch_size': batch_size_input,
            'max_tokens': max_tokens_input
        }

        settings_state = gr.State({})

        @gr.render(inputs=[model_sel, model_version_dropdown, presets_tracker])
        def render_features(model_id, model_version, tracker):
            render_features_content(app, model_id, model_version, settings_state)

        run_inference_wrapper = create_inference_wrapper(app, settings_state)
        update_model_settings_ui = create_update_model_settings_handler(app, model_feature_components, model_description)

        def initialize_model_state(model_id):
            return get_initial_model_state(app, model_id)

        model_sel.change(fn=initialize_model_state, inputs=[model_sel], outputs=[settings_state])

        def update_static_inputs(model_id, version):
            if not model_id:
                return 1, 1024
            values = resolve_model_values(app, model_id, version)
            return values.get('batch_size', 1), values.get('max_tokens', 1024)

        model_version_dropdown.change(fn=update_static_inputs, inputs=[model_sel, model_version_dropdown], outputs=[batch_size_input, max_tokens_input])

    components = {
        "model_description": model_description,
        "presets_tracker": presets_tracker,
        "models_chk": models_chk,
        "media_type_filter": media_type_filter,
        "model_sel": model_sel,
        "model_version_dropdown": model_version_dropdown,
        "batch_size_input": batch_size_input,
        "max_tokens_input": max_tokens_input,
        "model_feature_components": model_feature_components,
        "settings_state": settings_state,
        "run_inference_wrapper": run_inference_wrapper,
        "update_model_settings_ui": update_model_settings_ui
    }
    return components


def update_prompt_source_visibility(prompt_source_value):
    """Show/hide prompt-related fields based on prompt_source selection."""
    if not prompt_source_value:
        return [gr.update(visible=False)] * 7
    
    if prompt_source_value == "Prompt Presets":
        return [
            gr.update(visible=True),   # prompt_presets
            gr.update(visible=True),   # task_prompt
            gr.update(visible=False),  # prompt_prefix
            gr.update(visible=False),  # prompt_file_extension
            gr.update(visible=False),  # prompt_suffix
            gr.update(visible=True),   # Row: prompt_mode
            gr.update(visible=False),  # Row: file_metadata_mode
        ]
    elif prompt_source_value == "From File":
        return [
            gr.update(visible=False),  # prompt_presets
            gr.update(visible=False),  # task_prompt
            gr.update(visible=True),   # prompt_prefix
            gr.update(visible=True),   # prompt_file_extension
            gr.update(visible=True),   # prompt_suffix
            gr.update(visible=False),  # Row: prompt_mode
            gr.update(visible=True),   # Row: file_metadata_mode
        ]
    elif prompt_source_value == "From Metadata":
        return [
            gr.update(visible=False),  # prompt_presets
            gr.update(visible=False),  # task_prompt
            gr.update(visible=True),   # prompt_prefix
            gr.update(visible=False),  # prompt_file_extension
            gr.update(visible=True),   # prompt_suffix
            gr.update(visible=False),  # Row: prompt_mode
            gr.update(visible=True),   # Row: file_metadata_mode
        ]
    else:
        return [gr.update(visible=False)] * 7


def update_models_by_media_type(app, media_type):
    """Filter models based on selected media type."""
    filtered_models = app.get_models_by_media_type(media_type)
    if app.current_model_id in filtered_models:
        new_value = app.current_model_id
    elif filtered_models:
        new_value = filtered_models[0]
    else:
        new_value = None
    return gr.update(choices=filtered_models, value=new_value)


def create_control_area():
    """Create the control buttons and command output."""
    with gr.Row():
        save_btn = gr.Button("Save Settings", variant="secondary", scale=0)
        generate_command_btn = gr.Button("Generate Command", variant="secondary", scale=0)
        run_btn = gr.Button("Run Captioning", variant="primary", scale=1)
        with gr.Column(visible=False, scale=0, min_width=80, elem_classes="download-btn-wrapper") as download_btn_group:
            download_btn = gr.DownloadButton(
                label="", icon=str(Path(__file__).parent.parent.parent / "core" / "download_white.svg"), visible=True,
                variant="primary", scale=0, elem_classes="download-btn"
            )
    
    command_output = gr.Textbox(label="Generated CLI Command", lines=3, max_lines=5, visible=False, interactive=False)
    
    return {
        "save_btn": save_btn,
        "generate_command_btn": generate_command_btn,
        "run_btn": run_btn,
        "download_btn_group": download_btn_group,
        "download_btn": download_btn,
        "command_output": command_output
    }
