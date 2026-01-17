# gui/main.py
"""Main UI creation function for A Thousand Words. Keep this script clean. Write actual functionalities elsewhere  and import them here only when necessary"""

import gradio as gr
import os
import glob
import functools
from pathlib import Path
import logging

from .app import CaptioningApp
from .styles import CSS
from .js import JS
from .handlers import (
    create_update_model_settings_handler,
    create_auto_save_handler,
    create_inference_wrapper,
    create_gallery_cols_saver,
    create_version_change_handler,
    get_system_ram_gb,
    get_model_description_html,
)
from .dataset_gallery import create_dataset_gallery
from .model_info import create_model_info_tab
from .tabs import create_tools_tab, create_presets_tab, create_settings_tab, create_multi_model_tab, wire_multi_model_events, get_multi_model_reload_handler, create_general_settings_accordion, create_model_settings_accordion, create_control_area, update_prompt_source_visibility, update_models_by_media_type
from .tabs.tools import wire_tool_events
from .tabs.presets import wire_presets_events
from .tabs.settings import wire_settings_events
from .sections import create_input_source, create_gallery_section, create_viewer_section, wire_dataset_events, wire_gallery_settings
from .inference import build_inference_args, validate_run_state, validate_dataset_only, start_processing, run_with_dynamic_state
from .renderers.features import render_features_content
from .logic.model_logic import resolve_model_values, get_initial_model_state
import src.features as feature_registry

# Setup Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("GUI")
if not logger.handlers:
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

# Suppress "We will use 90% of the memory..." from accelerate
logging.getLogger("accelerate.utils.modeling").setLevel(logging.WARNING)



def create_ui(startup_message=None):
    """Create and return the Gradio Blocks interface."""
    app = CaptioningApp()
    

    ui_auto_save = create_auto_save_handler(app)
    save_gallery_cols = create_gallery_cols_saver(app)
    version_change_handler = create_version_change_handler(app)
    
    with gr.Blocks(title="A Thousand Words") as demo:
        
        # Header
        with gr.Row(elem_classes="header-row"):
            gr.Markdown("# A Thousand Words", elem_id="header-title")


        with gr.Tabs():            
            with gr.Tab("Captioning"):
                # General Settings Accordion (extracted to factory)
                general_components = create_general_settings_accordion(app)
                out_dir = general_components["out_dir"]
                out_fmt = general_components["out_fmt"]
                g_recursive = general_components["recursive"]
                g_over = general_components["overwrite"]
                g_normalize = general_components["normalize"]
                g_collapse = general_components["collapse"]
                g_clean = general_components["clean"]
                g_console = general_components["console"]
                g_strip_loop = general_components["strip_loop"]
                g_remove_chinese = general_components["remove_chinese"]
                g_max_width = general_components["max_width"]
                g_max_height = general_components["max_height"]
                pre_text = general_components["prefix"]
                suf_text = general_components["suffix"]

                # Model Settings Accordion (extracted to factory)
                model_components = create_model_settings_accordion(app, get_model_description_html)
                model_description = model_components["model_description"]
                presets_tracker = model_components["presets_tracker"]
                models_chk = model_components["models_chk"]
                media_type_filter = model_components["media_type_filter"]
                model_sel = model_components["model_sel"]
                model_version_dropdown = model_components["model_version_dropdown"]
                batch_size_input = model_components["batch_size_input"]
                max_tokens_input = model_components["max_tokens_input"]
                model_feature_components = model_components["model_feature_components"]
                settings_state = model_components["settings_state"]
                update_model_settings_ui = model_components["update_model_settings_ui"]

                # Control Area (extracted to factory)
                ctrl = create_control_area()
                save_btn = ctrl["save_btn"]
                generate_command_btn = ctrl["generate_command_btn"]
                run_btn = ctrl["run_btn"]
                download_btn_group = ctrl["download_btn_group"]
                download_btn = ctrl["download_btn"]
                command_output = ctrl["command_output"]



            # Tab 2: Multi Model Captioning
            with gr.Tab("Multi Model Captioning"):
                multi_model_components = create_multi_model_tab(app)

                


            # Tab 3: Tools
            with gr.Tab("Tools"):
                tool_components = create_tools_tab(app)


            # Tab 3.5: Presets
            with gr.Tab("Presets"):
                preset_components = create_presets_tab(app, presets_tracker)

            # Tab 4: Settings
            with gr.Tab("Settings"):
                cfg = app.config_mgr.get_global_settings()
                settings_components = create_settings_tab(app, cfg, models_chk)
                # Extract components for later event wiring
                vram_inp = settings_components["vram"]
                system_ram_inp = settings_components["system_ram"]
                g_unload_model = settings_components["unload"]
                items_per_page = settings_components["items_per_page"]
                gal_cols = settings_components["gal_cols"]
                gal_rows_slider = settings_components["gal_rows"]
                model_order_state = settings_components["model_order_state"]
                model_order_radio = settings_components["model_order_radio"]
                model_order_textbox = settings_components["model_order_textbox"]
                move_up_btn = settings_components["move_up_btn"]
                move_down_btn = settings_components["move_down_btn"]
                settings_save_btn = settings_components["save_btn"]
                settings_reset_btn = settings_components["reset_btn"]
                settings_reset_confirm_btn = settings_components["reset_confirm_btn"]
                settings_reset_cancel_btn = settings_components["reset_cancel_btn"]


            # Tab 5: Model Information
            with gr.Tab("Model Information"):
                create_model_info_tab(app.config_mgr)


        # Shared Input Source, Gallery, and Viewer (reusable across tabs)
        input_components = create_input_source(app)
        gallery_components = create_gallery_section(app)
        viewer_components = create_viewer_section()
        
        # Extract frequently used components for event bindings
        input_files = input_components["input_files"]
        input_path_text = input_components["input_path_text"]
        image_count = input_components["image_count"]
        load_source_btn = input_components["load_source_btn"]
        limit_count = input_components["limit_count"]
        clear_gallery_btn = input_components["clear_gallery_btn"]
        
        gallery_group = gallery_components["gallery_group"]
        pagination_row = gallery_components["pagination_row"]
        prev_btn = gallery_components["prev_btn"]
        page_number_input = gallery_components["page_number_input"]
        total_pages_label = gallery_components["total_pages_label"]
        next_btn = gallery_components["next_btn"]
        gal = gallery_components["gal"]
        
        inspector_group = viewer_components["inspector_group"]
        insp_tabs = viewer_components["insp_tabs"]
        insp_img = viewer_components["insp_img"]
        insp_video = viewer_components["insp_video"]
        insp_cap = viewer_components["insp_cap"]
        save_cap_btn = viewer_components["save_cap_btn"]
        insp_remove_btn = viewer_components["insp_remove_btn"]
        close_insp_btn = viewer_components["close_insp_btn"]


        # Event Bindings
        
        # Model selection change handler
        model_settings_outputs = [model_description]
        model_settings_outputs.extend(model_feature_components.values())
        
        # Model Selection Handler: Updates static component visibility/values
        model_sel.change(
            update_model_settings_ui,
            inputs=[model_sel],
            outputs=model_settings_outputs
        )
        

        model_version_dropdown.change(
            version_change_handler,
            inputs=[model_sel, model_version_dropdown],
            outputs=[batch_size_input]
        )
        
        # Media type filter change handler (uses extracted function)
        media_type_filter.change(
            lambda mt: update_models_by_media_type(app, mt),
            inputs=[media_type_filter],
            outputs=[model_sel]
        )
        
        # Prompt Source visibility is handled by update_prompt_source_visibility (imported)


        def _build_inference_args(model_id, model_ver, batch, tokens, dynamic_settings, 
                                 pre, suf, over, rec, con, unload, clean, collapse, norm, rm_cn, loop, 
                                 w, h, limit, out_dir_glob, out_fmt):
            """Wrapper for build_inference_args that passes app."""
            return build_inference_args(
                app, model_id, model_ver, batch, tokens, dynamic_settings,
                pre, suf, over, rec, con, unload, clean, collapse, norm, rm_cn, loop,
                w, h, limit, out_dir_glob, out_fmt
            )

        def _run_with_dynamic_state(*args):
            """Wrapper that passes app and build function to extracted runner."""
            return run_with_dynamic_state(app, _build_inference_args, *args)


        # GENERATE COMMAND LOGIC
        command_visible_state = gr.State(value=False)

        def generate_cli_wrapper(current_visible, *args):
            # Toggle logic
            if current_visible:
                # If currently visible, hide it params: (value, visible)
                return gr.update(visible=False, value=""), False, gr.update(value="Generate Command")
            
            # Build unified args
            args_dict = _build_inference_args(*args)
            
            # Generate CLI string
            model_id = args_dict.get('model_id') or app.current_model_id
            
            # Generate full command including defaults for transparency
            cmd = app.generate_cli_command(model_id, args_dict, skip_defaults=False)
            
            return gr.update(value=cmd, visible=True), True, gr.update(value="Generate Command ▼")

        # State for validation flow control
        run_valid_state = gr.State(value=True)
        multi_run_valid_state = gr.State(value=True)

        def _validate_run_state(model_id):
            """Wrapper for validate_run_state that passes app."""
            return validate_run_state(app, model_id)

        # 1. Validate
        run_btn.click(
            _validate_run_state,
            inputs=[model_sel],
            outputs=[run_valid_state]
        ).then(
            start_processing,
            inputs=[run_valid_state],
            outputs=[run_btn, download_btn_group, download_btn]
        ).then(
            _run_with_dynamic_state,
            inputs=[
                model_sel, 
                model_version_dropdown, batch_size_input, max_tokens_input,
                settings_state, # DYNAMIC STATE
                # Globals must match build_inference_args signature order
                pre_text, suf_text, g_over, g_recursive, g_console, g_unload_model,
                g_clean, g_collapse, g_normalize, g_remove_chinese, g_strip_loop,
                g_max_width, g_max_height, limit_count, out_dir, out_fmt,
                run_valid_state # VALIDATION STATE (last arg)
            ],
            outputs=[gal, run_btn, download_btn_group, download_btn],
            show_progress="hidden"
        )

        # Wire Up Generate Command Button
        generate_command_btn.click(
            generate_cli_wrapper,
            inputs=[
                command_visible_state, # Current visibility state
                model_sel, model_version_dropdown, batch_size_input, max_tokens_input,
                settings_state,
                pre_text, suf_text, g_over, g_recursive, g_console, g_unload_model,
                g_clean, g_collapse, g_normalize, g_remove_chinese, g_strip_loop,
                g_max_width, g_max_height, limit_count, out_dir, out_fmt
            ],
            outputs=[command_output, command_visible_state, generate_command_btn]
        )
        
        # Save Settings button on Captioning tab
        save_btn.click(
            app.save_settings,
            inputs=[
                vram_inp, models_chk, gal_cols, gal_rows_slider, limit_count,
                out_dir, out_fmt, g_over, g_recursive, g_console, g_unload_model,
                pre_text, suf_text, g_clean, g_collapse, g_normalize, g_remove_chinese, g_strip_loop,
                g_max_width, g_max_height, model_sel, model_version_dropdown, 
                batch_size_input, max_tokens_input, settings_state, items_per_page
            ],
            outputs=[]
        )
        
        # Generator wrapper for multi-model processing UI update
        def start_multi_processing(is_valid):
            if not is_valid:
                return gr.update(value="Run Captioning", interactive=True)
            return gr.update(value="Processing...", interactive=False)

        # Wrapper for multi run to check validity
        def run_multi_wrapper(*inputs):
             # Inputs: [*checkboxes, *formats, limit, is_valid]
             is_valid = inputs[-1]
             real_inputs = inputs[:-1]
             
             if not is_valid:
                 return gr.update() # No gallery update
             
             return app.run_multi_model_inference(*real_inputs)


        # Run Captioning
        def validate_dataset_only():
             if not app.dataset or not app.dataset.images:
                gr.Warning("No media found. Please load a folder or add images to the 'Input Source'.")
                return False
             return True

        # Model Selection Change Event
        # Note: feature rendering is handled by @gr.render above. We DO NOT want to call update_model_ui because it returns a fixed list of 11 updates which blindly target dynamic components by index.
        
        model_sel.change(
            fn=lambda m: gr.update(value=get_model_description_html(app, m)),
            inputs=[model_sel],
            outputs=[model_description]
        )
        
        
        

        # Wire dataset, gallery, pagination, and viewer events
        wire_dataset_events(app, input_components, gallery_components, viewer_components,
                           recursive_checkbox=g_recursive, items_per_page_input=items_per_page)
        wire_gallery_settings(app, gal_cols, gal_rows_slider, gallery_group, gal)

        
        # Session persistence on page reload
        demo.load(
            update_model_settings_ui,
            inputs=[model_sel],
            outputs=model_settings_outputs  # Use same outputs list as model_sel.change
        )
        
        demo.load(
            app.load_settings,
            inputs=None,
            outputs=[
                vram_inp, models_chk, gal_cols, gal_rows_slider, limit_count,  # System
                out_dir, out_fmt, g_over,  # Output
                g_recursive, g_console, g_unload_model, g_clean, g_collapse,  # Options

                g_normalize, g_remove_chinese, g_strip_loop,  # Text Processing
                g_max_width, g_max_height,  # Image
                pre_text, suf_text,  # Pre/Suffix
                model_sel,  # Model
                model_order_textbox,  # Model Order (hidden)
                model_order_radio,  # Model Order Radio (NEW)
                items_per_page,
                pagination_row
            ]
        )
        
        # Multi-Model Tab Event Wiring (Extracted to factory)
        wire_multi_model_events(app, multi_model_components, gal, limit_count, multi_run_valid_state)
        
        # Multi-model settings reload on page refresh
        multi_reload_fn, multi_reload_outputs = get_multi_model_reload_handler(app, multi_model_components)
        demo.load(multi_reload_fn, inputs=None, outputs=multi_reload_outputs)



        # Settings Tab Buttons
        # Build multi-model outputs from the components dict
        multi_checkboxes_list = list(multi_model_components["checkboxes"].values())
        multi_formats_list = list(multi_model_components["formats"].values())
        
        settings_save_btn.click(
            app.save_settings_simple,
            inputs=[vram_inp, system_ram_inp, models_chk, gal_cols, gal_rows_slider, g_unload_model, model_order_textbox, items_per_page],
            outputs=[model_sel, models_chk, model_order_radio] + multi_checkboxes_list + multi_formats_list
        )

        
        # Reset to Defaults - requires manual page refresh after reset
        def request_reset_confirmation():
            """Hide reset button, show confirmation buttons."""
            return (
                gr.update(visible=False), # reset_btn
                gr.update(visible=True),  # confirm
                gr.update(visible=True)   # cancel
            )

        def cancel_reset_confirmation():
            """Hide confirmation buttons, show reset button."""
            return (
                gr.update(visible=True),  # reset_btn
                gr.update(visible=False), # confirm
                gr.update(visible=False)  # cancel
            )

        def execute_reset():
            """Execute logic and restore buttons."""
            success, message = app.reset_to_defaults()
            if success:
                gr.Info(message)
            else:
                gr.Warning(message)
            
            # Restore UI state
            return (
                gr.update(visible=True),  # reset_btn
                gr.update(visible=False), # confirm
                gr.update(visible=False)  # cancel
            )
        
        # 1. Click Reset -> Show Confirmation
        settings_reset_btn.click(
            request_reset_confirmation,
            inputs=[],
            outputs=[settings_reset_btn, settings_reset_confirm_btn, settings_reset_cancel_btn]
        )

        # 2. Click Cancel -> Restore
        settings_reset_cancel_btn.click(
            cancel_reset_confirmation,
            inputs=[],
            outputs=[settings_reset_btn, settings_reset_confirm_btn, settings_reset_cancel_btn]
        )

        # 3. Click Confirm -> Execute and Restore
        settings_reset_confirm_btn.click(
            execute_reset,
            inputs=[],
            outputs=[settings_reset_btn, settings_reset_confirm_btn, settings_reset_cancel_btn]
        )

        # Wire Up Tools (Dynamic - auto-discovered tools)
        for tool_name, components in tool_components.items():
            components["tool"].wire_events(
                app,
                components["run_btn"],
                components["inputs"],
                gal,
                limit_count  # Pass limit from Input Source
            )
        
        # Wire Preset Events
        wire_presets_events(app, preset_components, model_sel, models_chk)


        if startup_message:
            def show_startup_msg():
                # Track if message shown using function attribute
                if not hasattr(show_startup_msg, "shown"):
                    gr.Info(startup_message)
                    show_startup_msg.shown = True
            
            demo.load(show_startup_msg, outputs=[])

    return demo
