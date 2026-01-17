"""
Presets tab factory.

Creates the User Presets tab with add/delete functionality.
"""

import gradio as gr


PRESET_CSS = """
<style>
.preset-row {
    margin: 0 !important;
    padding: 0px !important;
    border-bottom: 1px solid #374151;
    display: flex;
    align-items: center;
}
.preset-row .form {
    border: none !important;
    background: transparent !important;
}
.preset-row > div {
     padding-top: 4px !important; 
     padding-bottom: 4px !important;
     min-height: auto !important;
     display: flex;
     align-items: center;
}
.preset-cell-markdown {
    padding-left: 12px !important;
    padding-right: 12px !important;
    height: 100%;
    width: 100%;
    display: flex;
    align-items: center;
}
.preset-cell-markdown p {
    margin-bottom: 0px !important;
    font-size: 0.9em;
    line-height: normal;
}

/* Delete Button Styling - Red and Compact */
.preset-trash-btn {
    min-width: 0 !important;
    width: 32px !important;
    height: 32px !important;
    padding: 0 !important;
    background-color: #450a0a !important;
    color: #fca5a5 !important;
    border: 1px solid #7f1d1d !important;
    border-radius: 6px !important;
    display: flex !important;
    align-items: center;
    justify-content: center;
    margin: 0 auto !important;
}
.preset-trash-btn:hover {
    background-color: #7f1d1d !important;
    color: #fff !important;
    border-color: #991b1b !important;
}
</style>
"""


def create_presets_tab(app, presets_tracker: gr.State) -> dict:
    """
    Create and return Gradio UI components for the "User Prompt Presets" tab.
    
    Args:
        app: Application instance that provides preset management methods (e.g., get_preset_eligible_models,
            get_user_presets_dataframe, save_user_preset, delete_user_preset, refresh_models).
        presets_tracker: gr.State used to trigger re-rendering of the presets list when its value changes.
    
    Returns:
        dict: References to key UI components with keys:
            - "model_dd": model dropdown component
            - "name_txt": preset name textbox
            - "prompt_txt": preset prompt textbox
            - "save_btn": save/add preset button
            - "tracker": the provided presets_tracker state
    """
    gr.Markdown("### 📚 User Prompt Presets")
    
    with gr.Accordion("Add Preset", open=True):
        with gr.Row():
            preset_model_dd = gr.Dropdown(
                choices=["All Models"] + [(m, m) for m in app.get_preset_eligible_models()], 
                value="All Models", 
                label="Model"
            )
        preset_name_txt = gr.Textbox(label="Preset Name")
        preset_prompt_txt = gr.Textbox(label="Prompt Text", lines=3)
        preset_save_btn = gr.Button("💾 Add Preset", variant="primary")
    
    with gr.Accordion("Presets", open=True):
        gr.HTML(PRESET_CSS)
        
        @gr.render(inputs=[presets_tracker])
        def render_preset_list(tracker):
            """
            Render the User Presets list into the current Gradio layout.
            
            Retrieves user presets from the application and renders a header row plus one UI row per preset
            (model, name, and prompt text) with a per-row delete button that deletes the preset and increments
            the provided tracker to trigger a UI refresh. If no presets exist, returns a Markdown component
            containing the message "*No presets found.*".
            
            Parameters:
            	tracker (int or gr.State): Tracker value/state that is incremented after a delete to signal the UI to re-render.
            
            Returns:
            	gr.Markdown: A Markdown component with the message "*No presets found.*" when there are no presets; otherwise returns None.
            """
            rows = app.get_user_presets_dataframe()
            
            if not rows:
                return gr.Markdown("*No presets found.*")
            
            # Header
            with gr.Row(elem_classes="preset-row", variant="compact"):
                with gr.Column(scale=2, min_width=200):
                    gr.Markdown("**Model**", elem_classes="preset-cell-markdown")
                with gr.Column(scale=2, min_width=200):
                    gr.Markdown("**Name**", elem_classes="preset-cell-markdown")
                with gr.Column(scale=6):
                    gr.Markdown("**Prompt Text**", elem_classes="preset-cell-markdown")
                with gr.Column(scale=0, min_width=60):
                    pass
            
            # Rows
            for row_data in rows:
                p_model = row_data[0]
                p_name = row_data[1]
                p_text = row_data[2]
                
                with gr.Row(elem_classes="preset-row", variant="compact"):
                    with gr.Column(scale=2, min_width=200):
                        gr.Markdown(p_model, elem_classes="preset-cell-markdown")
                    with gr.Column(scale=2, min_width=200):
                        gr.Markdown(f"**{p_name}**", elem_classes="preset-cell-markdown")
                    with gr.Column(scale=6):
                        gr.Markdown(p_text, elem_classes="preset-cell-markdown")
                    with gr.Column(scale=0, min_width=60):
                        del_btn = gr.Button("🗑️", elem_classes="preset-trash-btn", size="sm")
                        
                        def do_delete(m=p_model, n=p_name):
                            """
                            Delete the user preset identified by the given model and name and trigger a UI refresh.
                            
                            Parameters:
                                m (str): Model identifier associated with the preset to delete.
                                n (str): Name of the preset to delete.
                            
                            Returns:
                                int: Updated tracker value (previous tracker value plus one).
                            """
                            app.delete_user_preset(m, n)
                            return tracker + 1
                        
                        del_btn.click(
                            fn=do_delete,
                            inputs=[],
                            outputs=[presets_tracker]
                        )
    
    return {
        "model_dd": preset_model_dd,
        "name_txt": preset_name_txt,
        "prompt_txt": preset_prompt_txt,
        "save_btn": preset_save_btn,
        "tracker": presets_tracker
    }


def wire_presets_events(app, components: dict, model_sel, models_chk):
    """
    Wire UI events for the Presets tab.
    
    Binds the Add Preset button to save a user preset, increment the presets tracker to trigger a re-render, and refresh the model selection controls afterward.
    
    Parameters:
        app: Application object exposing preset-related methods (expected to implement `save_user_preset(model, name, text)` and `refresh_models()`).
        components (dict): Dictionary returned by `create_presets_tab` containing UI components (keys: "model_dd", "name_txt", "prompt_txt", "save_btn", "tracker").
        model_sel: Model selection dropdown component to be refreshed after saving a preset.
        models_chk: Models checkbox group component to be refreshed after saving a preset.
    """
    def handle_save_preset(model, name, text, tracker_val):
        """
        Save a user preset and increment the presets tracker to trigger a UI refresh.
        
        Calls into the application to persist the preset identified by `model`, `name`, and `text`, then returns `tracker_val + 1` to signal that the presets list should be re-rendered.
        
        Returns:
            int: The updated tracker value (`tracker_val + 1`).
        """
        app.save_user_preset(model, name, text)
        return tracker_val + 1
    
    components["save_btn"].click(
        handle_save_preset,
        inputs=[components["model_dd"], components["name_txt"], 
                components["prompt_txt"], components["tracker"]],
        outputs=[components["tracker"]]
    ).then(
        fn=lambda: app.refresh_models(),
        outputs=[model_sel, models_chk]
    )