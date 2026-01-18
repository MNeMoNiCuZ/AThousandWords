"""
Settings tab factory.

Creates the System Settings tab.
"""

import gradio as gr


def get_system_ram_gb() -> int:
    """Get system RAM in GB."""
    try:
        import psutil
        return int(psutil.virtual_memory().total / (1024 ** 3))
    except Exception:
        return 32


def create_settings_tab(app, cfg: dict, models_chk) -> dict:
    """Create Settings tab components.
    
    Args:
        app: CaptioningApp instance
        cfg: Global settings config dict
        models_chk: Models checkbox group (to render in this tab)
        
    Returns:
        dict of component references
    """
    with gr.Accordion("System & Gallery Settings", open=True):
        with gr.Row():
            vram_inp = gr.Number(
                label="GPU VRAM (GB)", 
                value=cfg['gpu_vram'], 
                precision=0, 
                info="Your GPU's VRAM for batch size recommendations"
            )
            
            system_ram_default = cfg.get('system_ram', get_system_ram_gb())
            system_ram_inp = gr.Number(
                label="System RAM (GB)", 
                value=system_ram_default, 
                precision=0, 
                info="Your System RAM for batch size recommendations"
            )
            
            theme_mode_val = cfg.get('theme_mode', 'Dark')
            theme_mode = gr.Dropdown(
                choices=["Dark", "Light", "System"],
                label="Theme",
                value=theme_mode_val,
                info="UI color theme <b>(Requires Restart)</b>",
                interactive=True
            )

            unload_val = cfg.get('unload_model', True)
            g_unload_model = gr.Checkbox(
                label="Unload Model", 
                value=unload_val, 
                info="Unload model from VRAM immediately after finishing."
            )

        with gr.Row():
            gal_cols = gr.Slider(
                2, 16, step=1, 
                label="Gallery Columns", 
                value=app.gallery_columns, 
                info="Number of columns in the image gallery"
            )
            gal_rows_slider = gr.Slider(
                0, 20, step=1, 
                label="Gallery Rows", 
                value=app.gallery_rows, 
                info="Rows to display (0 = hide)"
            )

            items_per_page = gr.Number(
                label="Gallery Items Per Page", 
                value=app.gallery_items_per_page, 
                precision=0, 
                minimum=1, 
                info="Images per gallery page (pagination)"
            )
    
    with gr.Accordion("Model Management", open=True):
        with gr.Row():
            with gr.Column(scale=1):
                models_chk.render()
            
            with gr.Column(scale=1):
                current_order = app.sorted_models
                
                model_order_state = gr.State(value=current_order)
                
                model_order_radio = gr.Radio(
                    choices=current_order,
                    label="Model Display Order",
                    value=None,
                    info="Select a model, then use Move Up/Down to reorder",
                    interactive=True
                )
                
                with gr.Row():
                    move_up_btn = gr.Button("Move Up", variant="secondary", scale=1)
                    move_down_btn = gr.Button("Move Down", variant="secondary", scale=1)
                
                model_order_textbox = gr.Textbox(
                    value="\n".join(current_order),
                    visible=False
                )
    
    with gr.Accordion("Tool Management", open=True):
        with gr.Row():
            with gr.Column(scale=1):
                # enabled_tools is a list of strings
                tools_chk = gr.CheckboxGroup(
                    choices=app.tools,
                    value=app.enabled_tools,
                    label="Enabled Tools",
                    info="Uncheck to hide tools from the interface"
                )
            
            with gr.Column(scale=1):
                current_tool_order = app.sorted_tools
                
                tool_order_state = gr.State(value=current_tool_order)
                
                tool_order_radio = gr.Radio(
                    choices=current_tool_order,
                    label="Tool Display Order",
                    value=None,
                    info="Select a tool, then used Move Left/Right to reorder <b>(Requires Restart)</b>",
                    interactive=True
                )
                
                with gr.Row():
                    tool_move_left_btn = gr.Button("Move Left", variant="secondary", scale=1)
                    tool_move_right_btn = gr.Button("Move Right", variant="secondary", scale=1)
                
                tool_order_textbox = gr.Textbox(
                    value="\n".join(current_tool_order),
                    visible=False
                )

    with gr.Row():
        settings_save_btn = gr.Button("Save Settings", variant="primary", scale=0)
        settings_reset_btn = gr.Button("Reset", variant="secondary", scale=0)
        settings_reset_confirm_btn = gr.Button(
            "Confirm Reset", variant="stop", scale=0, visible=False
        )
        settings_reset_cancel_btn = gr.Button(
            "Cancel", variant="secondary", scale=0, visible=False
        )
    
    return {
        "vram": vram_inp,
        "system_ram": system_ram_inp,
        "unload": g_unload_model,
        "items_per_page": items_per_page,
        "gal_cols": gal_cols,
        "gal_rows": gal_rows_slider,
        "theme_mode": theme_mode,
        "model_order_state": model_order_state,
        "model_order_radio": model_order_radio,
        "model_order_textbox": model_order_textbox,
        "move_up_btn": move_up_btn,
        "move_down_btn": move_down_btn,
        "save_btn": settings_save_btn,
        "reset_btn": settings_reset_btn,
        "reset_confirm_btn": settings_reset_confirm_btn,
        "reset_cancel_btn": settings_reset_cancel_btn,
        "reset_confirm_btn": settings_reset_confirm_btn,
        "reset_cancel_btn": settings_reset_cancel_btn,
        "tools_chk": tools_chk,
        "tool_order_state": tool_order_state,
        "tool_order_radio": tool_order_radio,
        "tool_order_textbox": tool_order_textbox,
        "tool_move_left_btn": tool_move_left_btn,
        "tool_move_right_btn": tool_move_right_btn,
    }


def wire_settings_events(app, components: dict, model_sel, models_chk, 
                          multi_components: dict = None):
    """Wire settings events after components created.
    
    Args:
        app: CaptioningApp instance
        components: dict from create_settings_tab
        model_sel: Model selection dropdown
        models_chk: Models checkbox group
        multi_components: Optional dict of multi-model components
    """
    # Move Up/Down handlers
    components["move_up_btn"].click(
        app.move_model_up,
        inputs=[components["model_order_radio"], components["model_order_state"]],
        outputs=[components["model_order_radio"], components["model_order_textbox"], 
                 components["model_order_state"]]
    )
    
    components["move_down_btn"].click(
        app.move_model_down,
        inputs=[components["model_order_radio"], components["model_order_state"]],
        outputs=[components["model_order_radio"], components["model_order_textbox"], 
                 components["model_order_state"]]
    )

    # Tool Move Left / Right handlers
    components["tool_move_left_btn"].click(
        app.move_tool_up,
        inputs=[components["tool_order_radio"], components["tool_order_state"]],
        outputs=[components["tool_order_radio"], components["tool_order_textbox"], 
                 components["tool_order_state"]]
    )
    
    components["tool_move_right_btn"].click(
        app.move_tool_down,
        inputs=[components["tool_order_radio"], components["tool_order_state"]],
        outputs=[components["tool_order_radio"], components["tool_order_textbox"], 
                 components["tool_order_state"]]
    )
    
    # Reset confirmation flow
    def show_confirm():
        return gr.update(visible=False), gr.update(visible=True), gr.update(visible=True)
    
    def hide_confirm():
        return gr.update(visible=True), gr.update(visible=False), gr.update(visible=False)
    
    def do_reset():
        success, message = app.reset_to_defaults()
        if success:
            gr.Info(message)
        else:
            gr.Warning(message)
        return gr.update(visible=True), gr.update(visible=False), gr.update(visible=False)
    
    components["reset_btn"].click(
        show_confirm,
        outputs=[components["reset_btn"], components["reset_confirm_btn"], 
                 components["reset_cancel_btn"]]
    )
    
    components["reset_cancel_btn"].click(
        hide_confirm,
        outputs=[components["reset_btn"], components["reset_confirm_btn"], 
                 components["reset_cancel_btn"]]
    )
    
    components["reset_confirm_btn"].click(
        do_reset,
        outputs=[components["reset_btn"], components["reset_confirm_btn"], 
                 components["reset_cancel_btn"]]
    )
