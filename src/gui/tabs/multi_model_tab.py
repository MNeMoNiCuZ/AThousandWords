"""Multi-Model Captioning tab UI factory."""

import gradio as gr


def create_multi_model_tab(app):
    """
    Builds the UI for the Multi-Model Captioning tab and returns references to its interactive components.
    
    Parameters:
        app (CaptioningApp): Application instance used to populate available models and load saved multi-model settings.
    
    Returns:
        dict: Mapping of component names to Gradio component references:
            - "checkboxes": dict[str, gr.Checkbox] — per-model enable/disable checkboxes keyed by model_id.
            - "formats": dict[str, gr.Textbox] — per-model caption extension textboxes keyed by model_id.
            - "select_all_btn": gr.Button — button that selects all model checkboxes.
            - "deselect_all_btn": gr.Button — button that deselects all model checkboxes.
            - "save_btn": gr.Button — button that saves current multi-model settings.
            - "gen_cmd_btn": gr.Button — button that toggles generation/visibility of the command output.
            - "run_btn": gr.Button — primary button that starts the multi-model captioning run.
            - "cmd_output": gr.Textbox — read-only textbox used to display generated commands.
    """
    gr.Markdown("### 🤖 Multi-Model Processing")
    gr.Markdown("Run multiple models in sequence on the same dataset.")
    
    with gr.Accordion("🤖 Model Selection", open=True):
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
        "cmd_output": cmd_output
    }


def wire_multi_model_events(app, components, gal, limit_count, multi_run_valid_state):
    """
    Wire interactive event handlers connecting the Multi-Model tab UI to app logic.
    
    Parameters:
        app (CaptioningApp): Application instance providing model list, persistence, command generation, and inference methods.
        components (dict): Mapping of UI component references returned by create_multi_model_tab; expected keys include
            "checkboxes" (dict of per-model Checkbox components), "formats" (dict of per-model Textbox components),
            "select_all_btn", "deselect_all_btn", "save_btn", "gen_cmd_btn", "run_btn", and "cmd_output".
        gal (gr.components.Gallery): Gallery component to receive inference results.
        limit_count: UI input controlling the maximum runs per model (passed through to inference).
        multi_run_valid_state (gr.State): State component used to carry validation status into the run chain.
    
    Behavior:
        - Binds Select All / Deselect All buttons to toggle every model checkbox.
        - Connects Save Settings button to persist current checkbox and format inputs via app.save_multi_model_settings.
        - Toggles generation and visibility of the command output via the Generate Command button, using app.generate_multi_model_commands.
        - Validates that media is loaded before running; updates Run button state and then invokes app.run_multi_model_inference with all model inputs, writing results to the provided Gallery.
    """
    import gradio as gr
    
    checkboxes = components["checkboxes"]
    formats = components["formats"]
    
    checkbox_list = list(checkboxes.values())
    format_list = list(formats.values())
    all_inputs = checkbox_list + format_list
    
    # Select All
    def select_all():
        """
        Set every model checkbox to checked.
        
        Returns:
            list: A list of Gradio update objects that set each model checkbox to checked.
        """
        return [gr.update(value=True) for _ in app.models]
    
    components["select_all_btn"].click(
        fn=select_all,
        inputs=[],
        outputs=checkbox_list
    )
    
    # Deselect All
    def deselect_all():
        """
        Create UI updates that set every model checkbox to False.
        
        Returns:
            list: A list of Gradio update objects, one per model, each setting the checkbox `value` to `False`.
        """
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
        """
        Toggle the generated command output visibility and update associated UI elements.
        
        Parameters:
            current_visible (bool): Whether the command output is currently visible.
            *args: Values forwarded to app.generate_multi_model_commands to produce the command text.
        
        Returns:
            tuple: (cmd_output_update, visible_state, gen_btn_update)
                - cmd_output_update: Gradio update for the command textbox (sets `value` and `visible`).
                - visible_state (bool): `True` if the command output is visible after the toggle, `False` otherwise.
                - gen_btn_update: Gradio update for the Generate Command button label.
        """
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
        """
        Check that the application has a loaded dataset containing images.
        
        Displays a Gradio warning if no media is loaded.
        
        Returns:
            bool: `True` if a dataset with images is available, `False` otherwise.
        """
        if not app.dataset or not app.dataset.images:
            gr.Warning("No media loaded. Please load a folder first.")
            return False
        return True
    
    def start_processing(is_valid):
        """
        Update the Run button's label and interactivity based on whether a run is valid.
        
        Parameters:
            is_valid (bool): True if run validation passed; False otherwise.
        
        Returns:
            gr.update: An update object that sets the button to "Processing..." and disables interaction when `is_valid` is True; otherwise resets the label to "Run Captioning" and makes it interactive.
        """
        if not is_valid:
            return gr.update(value="Run Captioning", interactive=True)
        return gr.update(value="Processing...", interactive=False)
    
    def run_wrapper(*inputs):
        """
        Execute the multi-model inference call when the preceding validation step indicates the run is allowed.
        
        Parameters:
            *inputs: A sequence where the last element is a boolean `is_valid` flag produced by validation, and the preceding elements are the actual arguments to pass to `app.run_multi_model_inference`.
        
        Returns:
            If `is_valid` is false, a Gradio update object that makes no UI changes. Otherwise, the value returned by `app.run_multi_model_inference` (typically the output to update the gallery).
        """
        is_valid = inputs[-1]
        real_inputs = inputs[:-1]
        if not is_valid:
            return gr.update()
        return app.run_multi_model_inference(*real_inputs)
    
    run_inputs = checkbox_list + format_list + [limit_count, multi_run_valid_state]
    
    components["run_btn"].click(
        fn=validate_run,
        inputs=[],
        outputs=[multi_run_valid_state]
    ).then(
        fn=start_processing,
        inputs=[multi_run_valid_state],
        outputs=[components["run_btn"]]
    ).then(
        fn=run_wrapper,
        inputs=run_inputs,
        outputs=[gal]
    ).then(
        fn=lambda: gr.update(value="Run Captioning", interactive=True),
        outputs=[components["run_btn"]]
    )


def get_multi_model_reload_handler(app, components):
    """
    Create a loader function that repopulates the multi-model UI from saved settings.
    
    Parameters:
        components (dict): Mapping with keys "checkboxes" and "formats", each mapping model_id to the corresponding Gradio component to update.
    
    Returns:
        tuple:
            - load_settings (callable): A zero-argument function that loads saved multi-model settings and returns a list of Gradio update objects: first the checkbox values (enabled state) for each model in app.models order, then the per-model format strings.
            - outputs (list): List of the Gradio components (checkboxes followed by format textboxes) that the loader updates.
    """
    import gradio as gr
    
    checkboxes = components["checkboxes"]
    formats = components["formats"]
    
    def load_settings():
        """
        Load saved multi-model settings and produce Gradio update objects to restore each model's enabled state and per-model format.
        
        Returns:
            list: A list of Gradio update objects where the first N entries set each model checkbox's value (enabled state) and the next N entries set each model format textbox's value (format extension), in the same order as app.models.
        """
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
