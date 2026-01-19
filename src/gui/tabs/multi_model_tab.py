"""Multi-Model Captioning tab UI factory."""

import gradio as gr


def create_multi_model_tab(app):
    """
    Create the Multi-Model Captioning tab UI.
    
    Args:
        app: CaptioningApp instance
        
    Returns:
        dict: Component references for event wiring
    """
    gr.Markdown("### Multi-Model Processing")
    gr.Markdown("Run multiple models in sequence on the same dataset.")
    
    with gr.Accordion("Model Selection", open=True):
        with gr.Row():
            select_all_btn = gr.Button("Select All", variant="secondary", scale=0)
            deselect_all_btn = gr.Button("Deselect All", variant="secondary", scale=0)
        
        # Header Row
        with gr.Row(elem_classes="table-header"):
            with gr.Column(scale=3):
                gr.Markdown("**Model Selection**")
            with gr.Column(scale=1, min_width=200):
                gr.Markdown("Caption extension per model</span>")
        
        # Model List
        checkboxes = {}
        formats = {}
        
        saved_settings = app.load_multi_model_settings()
        
        for idx, model_id in enumerate(app.models):
            saved_enabled, saved_format = saved_settings[idx]
            
            with gr.Row(elem_classes="clickable-checkbox-row"):
                with gr.Column(scale=3, min_width=200):
                    checkboxes[model_id] = gr.Checkbox(
                        label=model_id, 
                        value=saved_enabled,
                        container=False,
                        elem_classes=["model-checkbox"]
                    )
                with gr.Column(scale=1, min_width=100):
                    formats[model_id] = gr.Textbox(
                        value=saved_format,
                        placeholder="extension",
                        show_label=False,
                        container=False,
                        min_width=80
                    )
    
    # Control buttons
    with gr.Row():
        save_btn = gr.Button("Save Settings", variant="secondary", scale=0)
        gen_cmd_btn = gr.Button("Generate Command", variant="secondary", scale=0)
        run_btn = gr.Button("Run Captioning", variant="primary", scale=1)
        
        # Download button for server mode / batch results
        from pathlib import Path
        with gr.Column(visible=False, scale=0, min_width=80, elem_classes="download-btn-wrapper") as download_btn_group:
            download_btn = gr.DownloadButton(
                label="", 
                icon=str(Path(__file__).parent.parent.parent.parent / "src" / "gui" / "core" / "download_white.svg"),
                visible=True, variant="primary", scale=0, elem_classes="download-btn"
            )
    
    # Command output
    cmd_output = gr.Textbox(
        label="Generated Commands",
        lines=10,
        max_lines=20,
        visible=False,
        interactive=False
    )
    
    gr.Markdown("---")
    
    return {
        "checkboxes": checkboxes,
        "formats": formats,
        "select_all_btn": select_all_btn,
        "deselect_all_btn": deselect_all_btn,
        "save_btn": save_btn,
        "gen_cmd_btn": gen_cmd_btn,
        "run_btn": run_btn,
        "download_btn": download_btn,
        "download_btn_group": download_btn_group,
        "cmd_output": cmd_output
    }


def wire_multi_model_events(app, components, gal, limit_count, multi_run_valid_state):
    """
    Wire event handlers for the Multi-Model tab.
    
    Args:
        app: CaptioningApp instance
        components: Dict from create_multi_model_tab
        gal: Gallery component
        limit_count: Limit count input
        multi_run_valid_state: State for validating run
    """
    import gradio as gr
    
    checkboxes = components["checkboxes"]
    formats = components["formats"]
    
    checkbox_list = list(checkboxes.values())
    format_list = list(formats.values())
    all_inputs = checkbox_list + format_list
    
    # Select All
    def select_all():
        return [gr.update(value=True) for _ in app.models]
    
    components["select_all_btn"].click(
        fn=select_all,
        inputs=[],
        outputs=checkbox_list
    )
    
    # Deselect All
    def deselect_all():
        return [gr.update(value=False) for _ in app.models]
    
    components["deselect_all_btn"].click(
        fn=deselect_all,
        inputs=[],
        outputs=checkbox_list
    )
    
    # Save Settings
    components["save_btn"].click(
        fn=app.save_multi_model_settings,
        inputs=all_inputs,
        outputs=[]
    )
    
    # Generate Command with toggle
    cmd_visible_state = gr.State(value=False)
    
    def gen_cmd_toggle(current_visible, *args):
        if current_visible:
            return gr.update(value="", visible=False), False, gr.update(value="Generate Command")
        cmd = app.generate_multi_model_commands(*args)
        return gr.update(value=cmd, visible=True), True, gr.update(value="Generate Command ▼")
    
    components["gen_cmd_btn"].click(
        fn=gen_cmd_toggle,
        inputs=[cmd_visible_state] + all_inputs,
        outputs=[components["cmd_output"], cmd_visible_state, components["gen_cmd_btn"]]
    )
    
    # Run Captioning with validation
    def validate_run():
        if not app.dataset or not app.dataset.images:
            gr.Warning("No media loaded. Please load a folder first.")
            return False
        return True
    
    def start_processing(is_valid):
        # Hide download button at start
        if not is_valid:
            return gr.update(value="Run Captioning", interactive=True), gr.update(visible=False), gr.update(visible=False)
        return gr.update(value="Processing...", interactive=False), gr.update(visible=False), gr.update(visible=False)
    

    def run_wrapper(*inputs):
        is_valid = inputs[-1]
        real_inputs = inputs[:-1]
        if not is_valid:
            # Return no-op updates matching signature: (gallery, run_btn, dl_grp, dl_btn)
            return gr.update(), gr.update(), gr.update(visible=False), gr.update(visible=False)
        
        # Returns (gallery_data, run_btn_update, dl_group_update, dl_btn_update)
        return app.run_multi_model_inference(*real_inputs)
    
    run_inputs = checkbox_list + format_list + [limit_count, multi_run_valid_state]
    
    components["run_btn"].click(
        fn=validate_run,
        inputs=[],
        outputs=[multi_run_valid_state]
    ).then(
        fn=start_processing,
        inputs=[multi_run_valid_state],
        outputs=[components["run_btn"], components["download_btn_group"], components["download_btn"]]
    ).then(
        fn=run_wrapper,
        inputs=run_inputs,
        outputs=[gal, components["run_btn"], components["download_btn_group"], components["download_btn"]]
    )


def get_multi_model_reload_handler(app, components):
    """Return a function to reload multi-model settings on page load."""
    import gradio as gr
    
    checkboxes = components["checkboxes"]
    formats = components["formats"]
    
    def load_settings():
        saved_settings = app.load_multi_model_settings()
        updates = []
        for idx in range(len(app.models)):
            enabled, _ = saved_settings[idx]
            updates.append(gr.update(value=enabled))
        for idx in range(len(app.models)):
            _, format_ext = saved_settings[idx]
            updates.append(gr.update(value=format_ext))
        return updates
    
    outputs = list(checkboxes.values()) + list(formats.values())
    return load_settings, outputs

