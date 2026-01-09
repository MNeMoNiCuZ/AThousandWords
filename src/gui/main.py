# gui/main.py
"""
Main UI creation function for the A Thousand Words.
Refactored for simplified settings workflow (Phase 3).
"""

import gradio as gr
import logging

from .app import CaptioningApp
from .app import CaptioningApp
from .styles import CSS
from .js import JS
from .handlers import (
    create_update_model_settings_handler,
    create_auto_save_handler,
    create_inference_wrapper,
    create_gallery_cols_saver,
    create_version_change_handler,
)
from .dataset_gallery import create_dataset_gallery
from .model_info import create_model_info_tab
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
    
    # Note: Handlers that need model_feature_components will be created later
    # after components are defined
    ui_auto_save = create_auto_save_handler(app)
    save_gallery_cols = create_gallery_cols_saver(app)
    version_change_handler = create_version_change_handler(app)
    
    with gr.Blocks(title="A Thousand Words") as demo:
        
        # --- Header ---
        with gr.Row(elem_classes="header-row"):
            gr.Markdown("# A Thousand Words", elem_id="header-title")


        # --- Tabs ---
        with gr.Tabs():
            
            # =========================================
            # Tab 1: Captioning (DEFAULT)
            # =========================================
            with gr.Tab("Captioning"):
                # Feature UI Component Maps
                feature_inputs = {}

                # ==============================
                # GENERAL SETTINGS
                # ==============================
                with gr.Accordion("⚙️ General Settings", open=False, elem_id="general-settings-accordion"):
                    # Get MERGED config (global.yaml + user_config.yaml)
                    # This is the SINGLE SOURCE OF TRUTH - no hardcoded fallbacks
                    cfg = app.config_mgr.get_global_settings()
                    
                    # Row 1: Output Settings
                    with gr.Row(elem_classes="general-settings-row"):
                        out_dir = gr.Textbox(label="Output Folder", value=cfg['output_dir'], placeholder="Leave empty for same folder as input", info="Directory for captions. Leave empty to save alongside input images.")
                        out_fmt = gr.Textbox(label="Output Format", value=cfg['output_format'], info="File extension (e.g., txt, json, caption)")
                        
                        rec_cfg = feature_registry.get_feature("recursive").get_gui_config()
                        g_recursive = gr.Checkbox(label=rec_cfg['label'], value=cfg['recursive'], info=rec_cfg['info'])
                        
                        ov_cfg = feature_registry.get_feature("overwrite").get_gui_config()
                        g_over = gr.Checkbox(label=ov_cfg['label'], value=cfg['overwrite'], info=ov_cfg['info'])
                        
                        feature_inputs["output_format"] = out_fmt
                        feature_inputs["recursive"] = g_recursive
                        feature_inputs["overwrite"] = g_over

                    # Row 2: Text Processing Options
                    with gr.Row(elem_classes="general-settings-row"):
                        # Logical Pipeline Order: Normalize -> Collapse -> Clean
                        norm_cfg = feature_registry.get_feature("normalize_text").get_gui_config()
                        g_normalize = gr.Checkbox(label=norm_cfg['label'], value=cfg['normalize_text'], info=norm_cfg['info'])
                        
                        coll_cfg = feature_registry.get_feature("collapse_newlines").get_gui_config()
                        g_collapse = gr.Checkbox(label=coll_cfg['label'], value=cfg['collapse_newlines'], info=coll_cfg['info'])

                        clean_cfg = feature_registry.get_feature("clean_text").get_gui_config()
                        g_clean = gr.Checkbox(label=clean_cfg['label'], value=cfg['clean_text'], info=clean_cfg['info'])

                        con_cfg = feature_registry.get_feature("print_console").get_gui_config()
                        g_console = gr.Checkbox(label=con_cfg['label'], value=cfg['print_console'], info=con_cfg['info'])
                        
                        feature_inputs["normalize_text"] = g_normalize
                        feature_inputs["collapse_newlines"] = g_collapse
                        feature_inputs["clean_text"] = g_clean
                        feature_inputs["print_console"] = g_console

                    # Row 3: Advanced Processing & Image
                    with gr.Row(elem_classes="general-settings-row"):
                        slp_cfg = feature_registry.get_feature("strip_loop").get_gui_config()
                        g_strip_loop = gr.Checkbox(label=slp_cfg['label'], value=cfg['strip_loop'], info=slp_cfg['info'])
                        
                        rem_cfg = feature_registry.get_feature("remove_chinese").get_gui_config()
                        g_remove_chinese = gr.Checkbox(label=rem_cfg['label'], value=cfg['remove_chinese'], info=rem_cfg['info'])
                        
                        mw_cfg = feature_registry.get_feature("max_width").get_gui_config()
                        g_max_width = gr.Number(label=mw_cfg['label'], value=cfg['max_width'], info=mw_cfg['info'])
                        
                        mh_cfg = feature_registry.get_feature("max_height").get_gui_config()
                        g_max_height = gr.Number(label=mh_cfg['label'], value=cfg['max_height'], info=mh_cfg['info'])
                        
                        feature_inputs["strip_loop"] = g_strip_loop
                        feature_inputs["remove_chinese"] = g_remove_chinese
                        feature_inputs["max_width"] = g_max_width
                        feature_inputs["max_height"] = g_max_height

                    # Row 4: Prefix/Suffix
                    with gr.Row(elem_classes="general-settings-row"):
                        pre_cfg = feature_registry.get_feature("prefix").get_gui_config()
                        pre_text = gr.Textbox(label=pre_cfg['label'], value=cfg['prefix'], placeholder="photo of, ", info=pre_cfg['info'])
                        
                        suf_cfg = feature_registry.get_feature("suffix").get_gui_config()
                        suf_text = gr.Textbox(label=suf_cfg['label'], value=cfg['suffix'], placeholder=", high quality", info=suf_cfg['info'])
                        
                        feature_inputs["prefix"] = pre_text
                        feature_inputs["suffix"] = suf_text



                # ==============================
                # MODEL SETTINGS
                # ==============================
                with gr.Accordion("🤖 Model Settings", open=True):
                    # Model Description - Full width display of model info from YAML config
                    # Get initial description from current model's config
                    initial_model_config = app.config_mgr.get_model_config(app.current_model_id) if app.current_model_id else {}
                    initial_description = initial_model_config.get("description", "")
                    model_description = gr.Markdown(
                        value=initial_description,
                        elem_classes="model-description"
                    )
                    
                    # Model Selection Row (includes media_type_filter, model dropdown, model_version, batch_size, max_tokens)
                    with gr.Row():
                        media_type_filter = gr.Dropdown(
                            choices=["Image", "Video"],
                            value="Image",
                            label="Media Type",
                            info="Filter models by type",
                            scale=0,
                            min_width=150
                        )
                        model_sel = gr.Dropdown(app.enabled_models, label="Model", value=app.current_model_id, interactive=True, allow_custom_value=False, filterable=False, info="Select captioning model", scale=1, min_width=200)
                        
                        # Model version (will be populated by model selection handler)
                        mv_cfg = feature_registry.get_feature("model_version").get_gui_config() if feature_registry.get_feature("model_version") else {}
                        model_version_dropdown = gr.Dropdown(choices=[], visible=False, label=mv_cfg.get('label', 'Model Version'), info=mv_cfg.get('info', 'Select model version'), scale=1)
                        
                        # Batch size (takes remaining space)
                        bs_cfg = feature_registry.get_feature("batch_size").get_gui_config() if feature_registry.get_feature("batch_size") else {}
                        batch_size_input = gr.Number(label=bs_cfg.get('label', 'Batch Size'), value=bs_cfg.get('value', 1), visible=False, info="Images per batch", scale=1, min_width=180)
                        
                        # Max tokens (same width as Media Type)
                        mt_cfg = feature_registry.get_feature("max_tokens").get_gui_config() if feature_registry.get_feature("max_tokens") else {}
                        max_tokens_input = gr.Number(label=mt_cfg.get('label', 'Max Tokens'), value=mt_cfg.get('value', 512), visible=False, info="Max output length", scale=0, min_width=150)

                    # Dynamic Feature Component Pool
                    # Create ALL possible model-specific feature components organized into rows
                    from .dynamic_components import create_component_from_feature_config
                    
                    # Initialize Static Components Dict (for handlers)
                    model_feature_components = {
                        'model_version': model_version_dropdown,
                        'batch_size': batch_size_input,
                        'max_tokens': max_tokens_input
                    }

                    # Model Settings State - Tracks current values of all dynamic features
                    # We use a State dict to persist values across renders
                    # Keys: feature_name, Values: current value
                    settings_state = gr.State({})

                    # Dynamic Rendering of Model Features
                    from .dynamic_components import create_component_from_feature_config
                    
                    @gr.render(inputs=[model_sel, model_version_dropdown])
                    def render_features(model_id, model_version):
                        if not model_id:
                            return
                        
                        # 1. Resolve Layout
                        # Use ConfigManager to get resolved feature_rows or fallback
                        resolved_rows = app.config_mgr.resolve_feature_rows(model_id)
                        if resolved_rows is None:
                            resolved_rows = []

                        # 2. Get Current Config & Defaults
                        # Helper to merge defaults with user overrides
                        config = app.config_mgr.get_model_config(model_id)
                        
                        # Use shared helper to ensure consistency with state initializer
                        # Pass model_version from dropdown if available (for re-render on version change)
                        current_values = _resolve_values(model_id, model_version)

                        # 3. Render Rows
                        # Create a map to track created components for wiring events later
                        component_map = {}
                        
                        for row_features in resolved_rows:
                            if not row_features: continue
                            
                            # check if any features in this row are NOT model_version/batch/tokens
                            valid_features = [f for f in row_features if f not in ['model_version', 'batch_size', 'max_tokens']]
                            if not valid_features: continue

                            with gr.Row():
                                for feature_name in valid_features:
                                    feature = feature_registry.get_feature(feature_name)
                                    if not feature: continue
                                    
                                    # Create config with current value
                                    feat_conf = feature.get_gui_config()
                                    
                                    # Override value if exists in current settings
                                    if feature_name in current_values:
                                        feat_conf['value'] = current_values[feature_name]
                                        
                                    # Apply overrides from YAML (e.g. tooltips)
                                    overrides = config.get('feature_overrides', {}).get(feature_name, {})
                                    feat_conf.update(overrides)

                                    # Check visibility dependency for THIS component
                                    # If this component depends on prompt_source, check that value
                                    ps_val = current_values.get("prompt_source", "Instruction Template")
                                    
                                    # Visibility Logic (Mirroring the handler)
                                    is_visible = True
                                    if feature_name in ["instruction_template", "task_prompt"]:
                                        if ps_val != "Instruction Template": is_visible = False
                                    elif feature_name == "prompt_file_extension":
                                        if ps_val != "From File": is_visible = False
                                    elif feature_name in ["prompt_prefix", "prompt_suffix"]:
                                        if ps_val not in ["From File", "From Metadata"]: is_visible = False
                                        
                                    # Force visible=True relative to config, BUT respect logic
                                    # If config says hidden (e.g. dev flag), keep hidden. 
                                    # But generally features in YAML are meant to be seen.
                                    feat_conf['visible'] = is_visible
                                    


                                    # Populate Dynamic Choices
                                    # 1. Prompt Source (Always static 3 options)
                                    if feature_name == "prompt_source":
                                        feat_conf['choices'] = ["Instruction Template", "From File", "From Metadata"]
                                        feat_conf['allow_custom_value'] = False # Force selection from list
                                    
                                    # 2. Instruction Template (Version-Specific)
                                    elif feature_name == "instruction_template":
                                        # Get current version for proper preset resolution
                                        current_version = current_values.get('model_version')
                                        presets = app.config_mgr.get_version_instruction_presets(model_id, current_version)
                                        choices = list(presets.keys()) if presets else []
                                        # Add Custom choice
                                        if "Custom" not in choices:
                                            choices.insert(0, "Custom")
                                        feat_conf['choices'] = choices
                                    
                                    # 3. Model Version (From Model Config)
                                    elif feature_name == "model_version":
                                        feat_conf['choices'] = config.get('model_versions', [])
                                    
                                    # 4. Model Mode (From Model Config)
                                    elif feature_name == "model_mode":
                                        feat_conf['choices'] = config.get('model_modes', [])

                                    # Create Component
                                    comp = create_component_from_feature_config(feat_conf)
                                    
                                    # Register component in map
                                    component_map[feature_name] = comp

                                    # Bind Change Event to Update State
                                    # We use a closure to capture feature_name
                                    def update_state(val, current_state, name=feature_name):
                                        # Handle case where state might be None
                                        if current_state is None:
                                            current_state = {}
                                        current_state[name] = val
                                        return current_state

                                    comp.change(fn=update_state, inputs=[comp, settings_state], outputs=[settings_state])
                        
                        # 4. Wire Up Conditional Visibility (After all components created)
                        # Prompt Source Visibility Logic
                        if "prompt_source" in component_map:
                            ps_comp = component_map["prompt_source"]
                            
                            # Identify target components that exist for this model
                            target_features = [
                                "instruction_template", "task_prompt",           # Instruction Mode
                                "prompt_prefix", "prompt_file_extension", "prompt_suffix", # File Mode
                                "prompt_prefix", "prompt_suffix"                 # Metadata Mode
                            ] # Note: redundancy handled by unique check below
                            
                            # Find which of these actually exist in the current layout
                            existing_targets = []
                            target_map = {} # name -> component
                            
                            for name in target_features:
                                if name in component_map and name not in target_map:
                                    target_map[name] = component_map[name]
                                    existing_targets.append(name)
                                    
                            if existing_targets:
                                # Define handler specifically for this set of components
                                def update_visibility_dynamic(source_val):
                                    if not source_val:
                                        return [gr.update(visible=False)] * len(existing_targets)
                                        
                                    updates = []
                                    for name in existing_targets:
                                        visible = False
                                        if source_val == "Instruction Template":
                                            if name in ["instruction_template", "task_prompt"]: visible = True
                                        elif source_val == "From File":
                                            if name in ["prompt_prefix", "prompt_file_extension", "prompt_suffix"]: visible = True
                                        elif source_val == "From Metadata":
                                            # Metadata uses prefix/suffix but NOT extension
                                            if name in ["prompt_prefix", "prompt_suffix"]: visible = True
                                        
                                        updates.append(gr.update(visible=visible))
                                    return updates

                                # Bind the handler
                                ps_comp.change(
                                    fn=update_visibility_dynamic,
                                    inputs=[ps_comp],
                                    outputs=list(target_map.values())
                                )
                                
                                # Trigger initial state (to set correct visibility on load)
                                # We need to use the current value of prompt_source
                                initial_ps_val = current_values.get("prompt_source", "Instruction Template")
                                # We can't easily trigger the event here without a load event, but 
                                # render is effectively a re-creation. We can set initial visibility in config?
                                # No, simpler to just rely on initial render state logic if possible,
                                # but create_component sets visible=True by default.
                                # Let's manually set the initial visibility on the components!
                                initial_updates = update_visibility_dynamic(initial_ps_val)
                                for comp, update in zip(target_map.values(), initial_updates):
                                    # This is a bit hacky - interacting with Gradio object directly
                                    # But we are in render function.
                                    # Actually, better to just set it during creation?
                                    # Too late now, they are created. 
                                    # We can use the update dict to modify the component props potentially?
                                    # No, gr.update returns a dict.
                                    # Wait, we are defining the component in the loop above.
                                    # We should move the visibility logic TO the creation loop or 
                                    # use a load event?
                                    # 'comp' object usage here is just for binding. 
                                    pass 
                                    # Actually, we SHOULD execute the visibility update immediately via js or load?
                                    # Or just let the user toggle. 
                                    # Better: Apply initial visibility based on current_values inside the creation loop?
                                    # Yes, but "prompt_source" value is needed.
                                    pass

                                    # actually, we SHOULD execute the visibility update immediately via js or load?
                                    # Or just let the user toggle. 
                                    # Better: Apply initial visibility based on current_values inside the creation loop?
                                    # Yes, but "prompt_source" value is needed.
                                    pass

                        # Wire Up Instruction Template -> Task Prompt
                        if "instruction_template" in component_map and "task_prompt" in component_map:
                            it_comp = component_map["instruction_template"]
                            tp_comp = component_map["task_prompt"]
                            
                            # Get version-specific presets for this model
                            current_version = current_values.get('model_version')
                            presets = app.config_mgr.get_version_instruction_presets(model_id, current_version)
                            
                            def update_task_prompt_from_template(template_name, current_state=settings_state.value):
                                if template_name == "Custom":
                                    return gr.update() # Legacy/Custom mode - preserve current text
                                
                                if not template_name or template_name not in presets:
                                    return gr.update() # No change
                                
                                new_prompt = presets[template_name]
                                
                                # Update State as well (so it's ready for inference)
                                current_state['task_prompt'] = new_prompt
                                
                                return new_prompt

                            it_comp.change(
                                fn=update_task_prompt_from_template,
                                inputs=[it_comp],
                                outputs=[tp_comp]
                            )
                        
                        # Correct Approach:
                        # We have `current_values` dictionary. We can determine visibility 
                        # for *every* component during the creation loop!
                        # We just need to check `current_values.get('prompt_source')`.

                                    
                    # Create generic handlers (simplified)
                    # We pass settings_state to them instead of components dict
                    run_inference_wrapper = create_inference_wrapper(app, settings_state)
                    
                    # Create model selection handler (for static components + description)
                    update_model_settings_ui = create_update_model_settings_handler(app, model_feature_components, model_description)

                    # State Synchronization Handler
                    def initialize_model_state(model_id):
                        """
                        Resets settings_state to a fresh dictionary containing ONLY the current model's defaults.
                        This prevents:
                        1. Stale values from previous models persisting if the user clicks Run immediately.
                        2. Feature pollution (accumulation of keys from all verified models).
                        """
                        if not model_id:
                            return {}
                        
                        # Use the same resolution logic as render_features
                        # No version override - will use saved or default
                        return _resolve_values(model_id, version_override=None)

                    # Helper for value resolution (Shared logic)
                    def _resolve_values(model_id, version_override=None):
                        config = app.config_mgr.get_model_config(model_id)
                        # Load full model settings structure
                        saved_model_root = app.config_mgr.user_config.get('model_settings', {}).get(model_id, {})
                        
                        # Determine if we have nested version structure (New Style) vs Flat (Old Style)
                        has_versions = 'versions' in saved_model_root
                        
                        # 1. Determine Current Version
                        # Priority: Override (Dropdown) > Saved Active Version > Defaults
                        defaults_raw = config.get('defaults', {})
                        
                        if version_override:
                            current_version = version_override
                        else:
                            # Try to get saved active version
                            # In new style, it's at root. In old style, it's also at root.
                            saved_ver = saved_model_root.get('model_version')
                            if saved_ver:
                                current_version = saved_ver
                            else:
                                # Fallback to Defaults
                                if isinstance(defaults_raw, dict):
                                    # Check if nested defaults (has string keys with dict values)
                                    first_key = next(iter(defaults_raw), None)
                                    if first_key and isinstance(first_key, str) and isinstance(defaults_raw.get(first_key), dict):
                                        # Nested structure - use first version as default
                                        current_version = first_key
                                    else:
                                        # Flat structure - extract version from defaults
                                        current_version = defaults_raw.get('model_version')
                                else:
                                    current_version = None

                        # 2. Get Defaults for this version
                        defaults = app.config_mgr.get_version_defaults(model_id, current_version)
                        
                        # 3. Get Saved Settings Diff for this version
                        saved_diff = {}
                        if has_versions:
                            # New Style: key into specific version
                            versions_dict = saved_model_root.get('versions', {})
                            # Only apply if this specific version has saved data
                            if current_version in versions_dict:
                                saved_diff = versions_dict[current_version].copy()
                                # Ensure model_version is set in diff (it was removed on save)
                                saved_diff['model_version'] = current_version
                        else:
                            # Old Style: Flat
                            # Only apply if the saved version matches (or if we have no versioning concept)
                            # Logic: If flat settings exist, they apply. 
                            # But wait, if I have saved "Alpha" settings (flat) and I switch to "Beta", 
                            # I don't want to apply Alpha settings!
                            # So we check if the saved version matches current_version
                            saved_ver = saved_model_root.get('model_version')
                            if saved_ver == current_version or not saved_ver:
                                saved_diff = saved_model_root.copy()

                        # 4. Merge
                        values = {**defaults, **saved_diff}
                        
                        # Smart Resolution: Ensure Task Prompt matches Instruction Template
                        # ... smart resolution logic continues below ...
                        
                        # Smart Resolution: Ensure Task Prompt matches Instruction Template
                        # If the user hasn't explicitly saved a task_prompt, we should enforce 
                        # the prompt corresponding to the selected template (or a fallback for Custom).
                        
                        task_prompt_is_saved = "task_prompt" in saved_diff
                        
                        if "instruction_template" in values and not task_prompt_is_saved:
                            template = values["instruction_template"]
                            
                            if template == "Custom":
                                # Fallback for Custom if no specific prompt saved
                                values["task_prompt"] = "Caption this image."
                            else:
                                # Resolve from presets
                                current_version = values.get('model_version')
                                presets = app.config_mgr.get_version_instruction_presets(model_id, current_version)
                                if template in presets:
                                    values["task_prompt"] = presets[template]
                                
                        return values

                    # Bind state reset to model change
                    # This ensures settings_state is fresh as soon as model changes
                    model_sel.change(
                        fn=initialize_model_state,
                        inputs=[model_sel],
                        outputs=[settings_state]
                    )

                    # Bind update for static inputs (batch_size, max_tokens) when version changes
                    # These are outside render_features so they need explicit update
                    def update_static_inputs(model_id, version):
                        if not model_id:
                            return 1, 1024
                        values = _resolve_values(model_id, version)
                        # Default fallbacks if keys missing
                        return values.get('batch_size', 1), values.get('max_tokens', 1024)

                    model_version_dropdown.change(
                        fn=update_static_inputs,
                        inputs=[model_sel, model_version_dropdown],
                        outputs=[batch_size_input, max_tokens_input]
                    )


                # Control Area: Save & Run
                with gr.Row():
                    save_btn = gr.Button("Save Settings", variant="secondary", scale=0)
                    gen_cmd_btn = gr.Button("Generate Command", variant="secondary", scale=0)
                    run_btn = gr.Button("Run Captioning", variant="primary", scale=1)
                    # Wrap download button in Column for proper visibility control
                    with gr.Column(visible=False, scale=0, min_width=80, elem_classes="download-btn-wrapper") as download_btn_group:
                        download_btn = gr.DownloadButton(
                            label="", 
                            icon="src/core/download_white.svg",
                            visible=True,  # Always visible within wrapper
                            variant="primary",
                            scale=0,
                            elem_classes="download-btn"
                        )
                
                # Command output textbox (hidden by default)
                cmd_output = gr.Textbox(
                    label="Generated CLI Command",
                    lines=3,
                    max_lines=5,
                    visible=False,
                    interactive=False
                )




            # =========================================
            # Tab 2: Multi Model Captioning
            # =========================================
            # =========================================
            # Tab 2: Multi Model Captioning
            # =========================================
            with gr.Tab("Multi Model Captioning"):
                gr.Markdown("### 🤖 Multi-Model Processing")
                gr.Markdown("Run multiple models in sequence on the same dataset.")
                
                with gr.Accordion("🤖 Model Selection", open=True):
                    # Select All / Deselect All buttons
                    with gr.Row():
                        multi_select_all_btn = gr.Button("Select All", variant="secondary", scale=0)
                        multi_deselect_all_btn = gr.Button("Deselect All", variant="secondary", scale=0)
                    

                    # Header Row
                    with gr.Row(elem_classes="table-header"):
                        with gr.Column(scale=3):
                            gr.Markdown("**Model Selection**")
                        with gr.Column(scale=1, min_width=200):
                            gr.Markdown(
                                "**Output Extension**\n"
                                "<span style='font-size: 0.8em; color: gray; font-weight: normal;'>Caption extension per model</span>"
                            )
                    
                    # Model List
                    multi_model_checkboxes = {}
                    multi_model_formats = {}
                    
                    # Load saved settings for initial values
                    saved_settings = app.load_multi_model_settings()
                    
                    for idx, model_id in enumerate(app.models):
                        # Get saved values for this model
                        saved_enabled, saved_format = saved_settings[idx]
                        
                        with gr.Row(elem_classes="clickable-checkbox-row"):
                            with gr.Column(scale=3, min_width=200):
                                multi_model_checkboxes[model_id] = gr.Checkbox(
                                    label=model_id, 
                                    value=saved_enabled,
                                    container=False,
                                    elem_classes=["model-checkbox"]
                                )
                            with gr.Column(scale=1, min_width=100):
                                multi_model_formats[model_id] = gr.Textbox(
                                    value=saved_format,
                                    placeholder="extension",
                                    show_label=False,
                                    container=False,
                                    min_width=80
                                )

                # Control buttons
                with gr.Row():
                    multi_save_btn = gr.Button("Save Settings", variant="secondary", scale=0)
                    multi_gen_cmd_btn = gr.Button("Generate Command", variant="secondary", scale=0)
                    multi_run_btn = gr.Button("Run Captioning", variant="primary", scale=1)
                
                # Command output textbox (hidden by default)
                multi_cmd_output = gr.Textbox(
                    label="Generated Commands",
                    lines=10,
                    max_lines=20,
                    visible=False,
                    interactive=False
                )
                
                gr.Markdown("---")
                



            # =========================================
            # Tab 3: Tools
            # =========================================
            with gr.Tab("Tools"):
                with gr.Tabs():
                    with gr.Tab("Metadata"):
                        gr.Markdown("Extract metadata from images.")
                        meta_src = gr.Dropdown(["all", "png_info", "exif"], label="Source", value="all", info="Which metadata field to search for existing captions.")
                        meta_upd = gr.Checkbox(label="Overwrite existing captions", value=True, info="If checked, existing captions in the tool will be replaced.")
                        meta_run = gr.Button("Extract Metadata")
                    with gr.Tab("Resize"):
                        gr.Markdown("Resize images to a maximum dimension.")
                        resize_px = gr.Number(label="Max Dimension (px)", value=1024, info="The maximum width or height. Aspect ratio is preserved.")
                        resize_run = gr.Button("Resize Images")


            # =========================================
            # Tab 4: Settings (System)
            # =========================================
            with gr.Tab("Settings"):
                 gr.Markdown("### 🖥️ System Settings")
                 with gr.Row():
                     vram_inp = gr.Number(label="GPU VRAM (GB)", value=cfg['gpu_vram'], precision=0, info="Your GPU's VRAM for batch size recommendations")
                     
                     vram_inp = gr.Number(label="GPU VRAM (GB)", value=cfg['gpu_vram'], precision=0, info="Your GPU's VRAM for batch size recommendations")
                     
                     unload_val = cfg.get('unload_model', True) # Default to True if missing
                     g_unload_model = gr.Checkbox(label="Unload Model", value=unload_val, info="Unload model from VRAM immediately after finishing.")
                     
                     items_per_page = gr.Number(label="Items Per Page", value=app.gallery_items_per_page, precision=0, minimum=1, info="Images per gallery page")
                     gal_cols = gr.Slider(2, 16, step=1, label="Gallery Columns", value=app.gallery_columns, info="Number of columns in the image gallery")
                     gal_rows_slider = gr.Slider(0, 20, step=1, label="Gallery Rows", value=app.gallery_rows, info="Rows to display (0 = hide)")
                 
                 gr.Markdown("### 📦 Model Management")
                 
                 with gr.Row():
                     with gr.Column(scale=1):
                         models_chk = gr.CheckboxGroup(choices=app.models, value=app.enabled_models, label="Enabled Models", info="Select which models appear in the model dropdown")
                     
                     with gr.Column(scale=1):
                         # Get current order (user config overrides global)
                         current_order = app.config_mgr.user_config.get('model_order', app.config_mgr.global_config.get('model_order', app.models))
                         
                         # State to track current order (allows multiple button clicks)
                         model_order_state = gr.State(value=current_order)
                         
                         # Radio for selecting which model to move
                         model_order_radio = gr.Radio(
                             choices=current_order,
                             label="Model Display Order",
                             value=None,
                             info="Select a model, then use Move Up/Down to reorder"
                         )
                         
                         # Move buttons
                         with gr.Row():
                             move_up_btn = gr.Button("⬆️ Move Up", variant="secondary", scale=1)
                             move_down_btn = gr.Button("⬇️ Move Down", variant="secondary", scale=1)
                         
                         # Hidden textbox to store the order (for saving)
                         model_order_textbox = gr.Textbox(
                             value="\n".join(current_order),
                             visible=False
                         )
                 
                 with gr.Row():
                     settings_save_btn = gr.Button("💾 Save Settings", variant="primary", scale=0)
                     settings_reset_btn = gr.Button("🗑️ Reset", variant="secondary", scale=0)


            # =========================================
            # Tab 5: Model Information
            # =========================================
            with gr.Tab("Model Information"):
                create_model_info_tab(app.config_mgr)


        # =========================================
        # Shared Input & Gallery (Outside Tabs)
        # =========================================
        # Moved here to persist state across tabs
        with gr.Accordion("📂 Input Source", open=True):
            with gr.Row():
                # Column 1: Drag & Drop
                with gr.Column(scale=1):
                    input_files = gr.File(
                        label="Drop Images or Folders",
                        file_count="multiple",
                        type="filepath",
                        height=130
                    )

                # Column 2: Manual Path Input + Image Count
                with gr.Column(scale=1):
                    input_path_text = gr.Textbox(
                        label="Input Folder Path", 
                        placeholder="C:/Path/To/Images", 
                        value="",
                        lines=1
                    )
                    image_count = gr.Markdown(f"<center><b style='font-size: 1.2em'>{len(app.dataset.images)} images</b></center>")

                # Column 3: Load & Clear Buttons
                with gr.Column(scale=1, min_width=200):
                    load_source_btn = gr.Button(
                        "Load Images From Input", 
                        variant="primary", 
                        size="lg"
                    )
                    with gr.Row():
                        limit_count = gr.Textbox(
                            placeholder="Limit", 
                            show_label=False, 
                            container=False, 
                            min_width=80,
                            scale=1
                        )
                        
                        clear_gallery_btn = gr.Button(
                            "Clear Dataset Gallery", 
                            variant="secondary", 
                            size="lg",
                            min_width=200,
                            scale=3
                        )
            
            # Warning Text on separate row to avoid cutting off
            with gr.Row():
                gr.Markdown(
                    "<div style='font-size: 0.8em; color: gray; margin-left: 10px;'>"
                    "⚠️ <b>Note:</b> Dragged files are saved to a temporary location. "
                    "Outputs will be placed in the configured output folder (default: /output)."
                    "</div>"
                )
        
        with gr.Group(visible=app.gallery_rows > 0, elem_id="gallery_group") as gallery_group:
            with gr.Accordion("🖼️ Dataset Gallery", open=True) as gallery_accordion:
                # Compact Pagination Controls
                with gr.Row(elem_classes="pagination-row compact-pagination") as pagination_row:
                    prev_btn = gr.Button("◀", variant="secondary", size="sm", elem_classes="pagination-btn")
                    
                    # Page Number Input (User can type to jump)
                    page_number_input = gr.Number(
                        value=app.current_page,
                        label=None,
                        show_label=False,
                        precision=0,
                        minimum=1,
                        container=False, # No box around it
                        elem_classes="pagination-input",
                        scale=0,
                        min_width=60
                    )
                    
                    # Just the "/ X" part
                    total_pages_label = gr.Markdown(value=f"/ {app.get_total_pages()}", elem_classes="pagination-label")
                    
                    next_btn = gr.Button("▶", variant="secondary", size="sm", elem_classes="pagination-btn")

                gal = gr.Gallery(
                    label=None,
                    columns=app.gallery_columns,
                    height=app.calc_gallery_height(),
                    object_fit="contain",
                    allow_preview=False,
                    show_label=True,
                    elem_classes="gallery-section",
                    elem_id="main_gallery"
                )


        # --- Inspector Section ---
        with gr.Group(visible=False) as inspector_group:
            gr.Markdown("### 🔍 Viewer")
            with gr.Column(elem_classes="input-section"):
                with gr.Row():
                    with gr.Tabs() as insp_tabs:
                        with gr.Tab("Image", id="img_tab") as img_tab:
                            insp_img = gr.Image(label=None, interactive=False, height=600, show_label=False)
                        with gr.Tab("Video", id="vid_tab") as vid_tab:
                            insp_video = gr.Video(label=None, interactive=False, height=600, show_label=False, autoplay=False)
                    with gr.Column():
                        insp_cap = gr.TextArea(label="Caption", lines=15)
                        with gr.Row():
                            save_cap_btn = gr.Button("💾 Save Caption", variant="primary", scale=1)
                            insp_remove_btn = gr.Button("🗑️ Remove", variant="stop", scale=0, min_width=120)
                        close_insp_btn = gr.Button("Close Viewer", variant="secondary")


        # =============================================
        # EVENT BINDINGS (After all components defined)
        # =============================================
        
        # Model selection change handler
        # Build outputs list explicitly: description first, then static feature components (version/batch/tokens)
        model_settings_outputs = [model_description]
        model_settings_outputs.extend(model_feature_components.values())
        
        # Model Selection Handler: Updates static component visibility/values
        model_sel.change(
            update_model_settings_ui,
            inputs=[model_sel],
            outputs=model_settings_outputs
        )
        
        # New: Version change handler (updates batch size)
        model_version_dropdown.change(
            version_change_handler,
            inputs=[model_sel, model_version_dropdown],
            outputs=[batch_size_input]
        )
        
        # Media type filter change handler
        def update_models_by_media_type(media_type):
            """Filter models based on selected media type."""
            filtered_models = app.get_models_by_media_type(media_type)
            # Update model dropdown with filtered list
            # If current model is not in filtered list, select first available
            if app.current_model_id in filtered_models:
                new_value = app.current_model_id
            elif filtered_models:
                new_value = filtered_models[0]
            else:
                new_value = None
            
            return gr.update(choices=filtered_models, value=new_value)
        
        media_type_filter.change(
            update_models_by_media_type,
            inputs=[media_type_filter],
            outputs=[model_sel]
        )
        
        # Prompt Source conditional visibility handler
        def update_prompt_source_visibility(prompt_source_value):
            """Show/hide prompt-related fields based on prompt_source selection."""
            if not prompt_source_value:
                # Hide all if no selection
                return [gr.update(visible=False)] * 7
            
            # Determine which fields to show based on mode
            if prompt_source_value == "Instruction Template":
                # Show: instruction_template, task_prompt
                # Hide: prompt_prefix, prompt_file_extension, prompt_suffix
                # Rows: instruction_mode VISIBLE, file_metadata_mode HIDDEN
                return [
                    gr.update(visible=True),   # instruction_template
                    gr.update(visible=True),   # task_prompt
                    gr.update(visible=False),  # prompt_prefix
                    gr.update(visible=False),  # prompt_file_extension
                    gr.update(visible=False),  # prompt_suffix
                    gr.update(visible=True),   # Row: instruction_mode
                    gr.update(visible=False),  # Row: file_metadata_mode
                ]
            elif prompt_source_value == "From File":
                # Show: prompt_prefix, prompt_file_extension, prompt_suffix
                # Hide: instruction_template, task_prompt
                # Rows: instruction_mode HIDDEN, file_metadata_mode VISIBLE
                return [
                    gr.update(visible=False),  # instruction_template
                    gr.update(visible=False),  # task_prompt
                    gr.update(visible=True),   # prompt_prefix
                    gr.update(visible=True),   # prompt_file_extension
                    gr.update(visible=True),   # prompt_suffix
                    gr.update(visible=False),  # Row: instruction_mode
                    gr.update(visible=True),   # Row: file_metadata_mode
                ]
            elif prompt_source_value == "From Metadata":
                # Show: prompt_prefix, prompt_suffix (NO file extension for metadata)
                # Hide: instruction_template, task_prompt, prompt_file_extension
                # Rows: instruction_mode HIDDEN, file_metadata_mode VISIBLE (but one child hidden)
                return [
                    gr.update(visible=False),  # instruction_template
                    gr.update(visible=False),  # task_prompt
                    gr.update(visible=True),   # prompt_prefix
                    gr.update(visible=False),  # prompt_file_extension
                    gr.update(visible=True),   # prompt_suffix
                    gr.update(visible=False),  # Row: instruction_mode
                    gr.update(visible=True),   # Row: file_metadata_mode
                ]
            else:
                # Unknown mode - hide all
                return [gr.update(visible=False)] * 7  # 5 components + 2 rows
        


        # Run Button with state management
        def run_with_button_state(model_id, *all_inputs):
            """
            Wrapper to handle button state and call inference.
            
            Args:
                model_id: Selected model  
                *all_inputs: Dynamic feature values + global settings
            """
            # Split inputs: model features come first, then global settings at end
            num_model_features = len(model_feature_components)
            feature_values = all_inputs[:num_model_features]
            global_values = all_inputs[num_model_features:]
            
            # Build complete args dict
            args = {}
            
            # Add model-specific features
            for feature_name, value in zip(model_feature_components.keys(), feature_values):
                args[feature_name] = value
            
            # Add global features (in order: prefix, suffix, overwrite, recursive, console, unload, clean, collapse, normalize, remove_chinese, strip_loop, max_width, max_height, limit_count, out_dir, out_fmt)
            if len(global_values) >= 16:
                args['prefix'] = global_values[0]
                args['suffix'] = global_values[1]
                args['overwrite'] = global_values[2]
                args['recursive'] = global_values[3]
                args['print_console'] = global_values[4]
                args['unload_model'] = global_values[5]
                args['clean_text'] = global_values[6]
                args['collapse_newlines'] = global_values[7]
                args['normalize_text'] = global_values[8]
                args['remove_chinese'] = global_values[9]
                args['strip_loop'] = global_values[10]
                args['max_width'] = global_values[11]
                args['max_height'] = global_values[12]
                args['limit_count'] = global_values[13]
                args['output_dir'] = global_values[14]
                args['output_format'] = global_values[15]
            
            # Get merged settings
            settings = app.config_mgr.get_global_settings()
            args["gpu_vram"] = settings['gpu_vram']

            
            # Now returns (gallery_data, download_btn_update, stats)
            result = app.run_inference(model_id, args)
            
            stats = {}
            # Unpack result
            if isinstance(result, tuple) and len(result) == 3:
                gallery_data, download_update, stats = result
            elif isinstance(result, tuple) and len(result) == 2:
                # Fallback for legacy return
                gallery_data, download_update = result
            else:
                # Fallback for error cases or if signature mismatches (safety)
                gallery_data = result
                download_update = gr.update(visible=False)
            
                
            # Show completion notification if we have stats
            # Show completion notification if we have stats
            if stats:
                model_name = stats.get('model_name', 'Unknown')
                processed = stats.get('processed', 0)
                skipped = stats.get('skipped', 0)
                total = stats.get('total', 0)
                time_val = stats.get('time', 0)
                peak_vram = stats.get('peak_vram', 0)
                empty_count = stats.get('empty_count', 0)

                # Format VRAM string
                vram_str = f"{peak_vram:.2f} GB" if peak_vram else "N/A"

                if processed > 0:
                    msg = (
                        f"Model: {model_name} | "
                        f"Captioned: {processed} files | "
                        f"Time: {time_val:.2f}s | "
                        f"Peak VRAM: {vram_str}"
                    )
                    # Only show skipped if non-zero
                    if skipped > 0:
                        msg += f" | Skipped: {skipped}"
                        
                    gr.Info(msg, title="✅ Success!")
                else:
                    # Special case for 0 processed (e.g. all skipped or error)
                    if skipped > 0:
                        msg = f"Skipped all {skipped} files (already exist). Check 'Overwrite' to re-caption."
                        gr.Info(msg, title="ℹ️ No Changes")
                    else:
                        msg = "No files processed."
                        gr.Warning(msg)
                
                # Report empty captions as an Error/Warning
                if empty_count > 0:
                    gr.Error(
                        f"⚠️ {empty_count} of {total} captions were returned empty by {model_name}!", 
                        duration=10
                    )
                
            # Handle download button - show with download icon when ready
            if isinstance(download_update, dict):
                # Prepare updates for both group visibility and button value
                if download_update.get('visible') or download_update.get('value'):
                    # Show group, set button value and styling
                    download_group_update = gr.update(visible=True)
                    download_btn_update = gr.update(
                        value=str(download_update.get('value')),
                        variant='primary',
                        interactive=True,
                        icon="src/core/download_white.svg",
                        elem_classes="download-btn"  # Remove processing class
                    )
                else:
                    # Hide group
                    download_group_update = gr.update(visible=False)
                    download_btn_update = gr.update(value=None)
            else:
                # Fallback - hide
                download_group_update = gr.update(visible=False)
                download_btn_update = gr.update(value=None)
                
            return gallery_data, gr.update(value="Run Captioning", interactive=True), download_group_update, download_btn_update
        
        def start_processing():
            """
            Update UI on run start:
            1. Run Button -> "Processing...", Disabled
            2. Download Button -> Show wrapper, set button to Processing state (spinner)
            """
            return (
                gr.update(value="Processing...", interactive=False),  # run_btn
                gr.update(visible=True),  # download_btn_group - SHOW IT so spinner is visible
                gr.update(
                    value=None,
                    variant="secondary",
                    interactive=False,
                    icon=None,  # Icon hidden by CSS anyway, but explicit is good
                    elem_classes="download-btn processing"  # Adds spinner
                )  # download_btn
            )

        # Run Logic with Dynamic State
        def run_with_dynamic_state(model_id, 
                                   model_ver, batch, tokens,
                                   dynamic_settings,
                                   # Global args
                                   pre, suf, over, rec, con, unload, clean, collapse, norm, rm_cn, loop, w, h, limit, out_dir_glob, out_fmt):
            
            # Base args from dynamic state
            args = dynamic_settings.copy() if dynamic_settings else {}
            
            # Add static overrides
            args['model_version'] = model_ver
            args['batch_size'] = int(batch) if batch else 1
            args['max_tokens'] = int(tokens) if tokens else 512
            
            # Add globals
            args.update({
                'prefix': pre, 'suffix': suf, 
                'overwrite': over, 'recursive': rec, 
                'print_console': con, 'unload_model': unload,
                'clean_text': clean, 'collapse_newlines': collapse, 
                'normalize_text': norm, 'remove_chinese': rm_cn, 
                'strip_loop': loop,
                'max_width': int(w) if w else None, 
                'max_height': int(h) if h else None,
                'limit_count': limit,
                'output_dir': out_dir_glob,
                'output_format': out_fmt
            })
            
            # Add global settings (VRAM)
            g_set = app.config_mgr.get_global_settings()
            args['gpu_vram'] = g_set.get('gpu_vram', 24)
            
            # Run Inference
            # app.run_inference uses self.dataset so we don't pass files here
            result = app.run_inference(model_id, args)
            
            # Handle Result Unpacking
            stats = {}
            if isinstance(result, tuple) and len(result) == 3:
                gallery_data, download_update, stats = result
            elif isinstance(result, tuple) and len(result) == 2:
                gallery_data, download_update = result
            else:
                gallery_data = result
                download_update = gr.update(visible=False)
            
            # Notifications
            if stats:
                processed = stats.get('processed', 0)
                skipped = stats.get('skipped', 0)
                time_val = stats.get('time', 0)
                empty = stats.get('empty_count', 0)
                peak_vram = stats.get('peak_vram', 0)
                vram_str = f"{peak_vram:.2f} GB" if peak_vram else "N/A"
                
                if processed > 0:
                    model_name = stats.get('model_name', 'Unknown')
                    msg = (
                        f"Model: {model_name}<br>"
                        f"Captioned: {processed} files<br>"
                        f"Time: {time_val:.2f}s<br>"
                        f"Peak VRAM: {vram_str}"
                    )
                    if skipped > 0:
                        msg += f"<br>Skipped: {skipped}"
                    gr.Info(msg, title="Success")
                elif skipped > 0:
                     gr.Info(f"Skipped all {skipped} files (already exist).<br>Check 'Overwrite' to re-caption.", title="No Changes")
                else:
                    # Fallback if 0 processed 0 skipped - this is an error condition
                    gr.Warning("No files were processed. The dataset may be empty.")

                if empty > 0:
                    gr.Warning(f"{empty} captions were empty!")
            else:
                # No stats returned - fallback error
                gr.Warning("Processing completed but no statistics were returned.")
            
            # Download Button Logic
            dl_grp = gr.update(visible=False)
            dl_btn = gr.update(visible=False)
            if isinstance(download_update, dict) and download_update.get('value'):
                # If download_update has a value (zip file path), show the button
                dl_grp = gr.update(visible=True)
                dl_btn = gr.update(
                    value=download_update.get('value'),
                    visible=True,
                    interactive=True,
                    variant="primary",  # Keep green
                    icon="src/core/download_white.svg",
                    elem_classes="download-btn"  # Remove "processing" class to stop spinner
                )

            return gallery_data, gr.update(value="Run Captioning", interactive=True), dl_grp, dl_btn


        run_btn.click(
            start_processing,
            inputs=None,
            outputs=[run_btn, download_btn_group, download_btn]  # Group visibility + button value
        ).then(
            run_with_dynamic_state,
            inputs=[
                model_sel, 
                model_version_dropdown, batch_size_input, max_tokens_input,
                settings_state, # DYNAMIC STATE
                # Globals
                pre_text, suf_text, g_over, g_recursive, g_console, g_unload_model,
                g_clean, g_collapse, g_normalize, g_remove_chinese, g_strip_loop,
                g_max_width, g_max_height, limit_count, out_dir, out_fmt
            ],
            outputs=[gal, run_btn, download_btn_group, download_btn],
            show_progress="minimal"
        )
        
        # Save Settings Button (Phase 3)
        # Note: We need to update handlers list in imports if we move logic there, 
        # or bind directly to app.save_settings.
        # Since we modified the inputs significantly, we need to update app.save_settings to match this.
        save_btn.click(
            app.save_settings,
            inputs=[
                # System inputs
                vram_inp, models_chk, gal_cols, gal_rows_slider, limit_count,
                # Output Settings
                out_dir, out_fmt, g_over, g_recursive, g_console, g_unload_model,
                # Pre/Suffix
                pre_text, suf_text, 
                # Text Processing
                g_clean, g_collapse, g_normalize, g_remove_chinese, g_strip_loop,
                # Image
                g_max_width, g_max_height,
                # Model Params (for model-specific defaults) - now using dynamic components
                model_sel, *model_feature_components.values(),
                # State needed for dynamic features
                settings_state,
                # items_per_page added at end to minimize diff
                items_per_page
            ],
            outputs=[] # Just save to disk / notification
        )
        
        # Generate Command Button - toggles command display textbox
        # Generate Command Button - toggles command display textbox
        def generate_command_handler(model_id, current_visibility, settings_dict, *all_inputs):
            """Toggle command display and generate CLI command."""
            # Split inputs same way as run_with_button_state
            num_model_features = len(model_feature_components)
            feature_values = all_inputs[:num_model_features]
            global_values = all_inputs[num_model_features:]
            
            # If currently visible, hide it
            if current_visibility:
                return gr.update(visible=False, value=""), gr.update(value="Generate Command")
            
            # Build complete args dict
            args = {}
            
            # Add dynamic features from state (Priority)
            if settings_dict:
                args.update(settings_dict)
                
            # Add static model-specific features (overrides state if duplicate, but usually disjoint)
            for feature_name, value in zip(model_feature_components.keys(), feature_values):
                args[feature_name] = value
            
            # Add global features
            if len(global_values) >= 14:
                args['prefix'] = global_values[0]
                args['suffix'] = global_values[1]
                args['overwrite'] = global_values[2]
                args['recursive'] = global_values[3]
                args['print_console'] = global_values[4]
                args['clean_text'] = global_values[5]
                args['collapse_newlines'] = global_values[6]
                args['normalize_text'] = global_values[7]
                args['remove_chinese'] = global_values[8]
                args['strip_loop'] = global_values[9]
                args['max_width'] = global_values[10]
                args['max_height'] = global_values[11]
                args['output_dir'] = global_values[12]
                args['output_format'] = global_values[13]
            
            # Generate command with ALL parameters (skip_defaults=False)
            cli_command = app.generate_cli_command(model_id, args, skip_defaults=False)
            
            # Show textbox with command
            return gr.update(visible=True, value=cli_command), gr.update(value="Close Command")
        
        gen_cmd_btn.click(
            generate_command_handler,
            inputs=[
                model_sel, cmd_output, settings_state,
                # Model Params
                *model_feature_components.values(),
                # Globals
                pre_text, suf_text, g_over, g_recursive, g_console,
                g_clean, g_collapse, g_normalize, g_remove_chinese, g_strip_loop,
                g_max_width, g_max_height, out_dir, out_fmt
            ],
            outputs=[cmd_output, gen_cmd_btn]
        )

        # Inspector events
        gal.select(app.open_inspector, outputs=[inspector_group, insp_tabs, insp_img, insp_video, insp_cap])
        save_cap_btn.click(app.save_and_close, inputs=[insp_cap], outputs=[gal, inspector_group])
        close_insp_btn.click(app.close_inspector, outputs=[inspector_group])

        # Tools
        meta_run.click(app.run_metadata, inputs=[meta_src, meta_upd], outputs=[gal])
        resize_run.click(app.run_resize, inputs=[resize_px], outputs=[gal])
        
        # Multi-Model Captioning Events
        # Select All / Deselect All
        multi_select_all_btn.click(
            lambda: [gr.update(value=True) for _ in app.models],
            outputs=list(multi_model_checkboxes.values())
        )
        multi_deselect_all_btn.click(
            lambda: [gr.update(value=False) for _ in app.models],
            outputs=list(multi_model_checkboxes.values())
        )
        
        # Save Settings
        multi_save_btn.click(
            app.save_multi_model_settings,
            inputs=[*multi_model_checkboxes.values(), *multi_model_formats.values()],
            outputs=[]
        )
        
        # Generate Command (Toggle)
        def multi_gen_cmd_handler(current_visibility, *inputs):
            # inputs = [feature values... checkboxes... formats...]
            # Split: feature values, then multi-model checkboxes, then formats
            num_features = len(model_feature_components)
            num_models = len(app.models)
            
            feature_values = inputs[:num_features]
            checkboxes = inputs[num_features:num_features + num_models]
            formats = inputs[num_features + num_models:]
            
            if current_visibility:
                return gr.update(visible=False, value=""), gr.update(value="Generate Command")
            else:
                # Build feature dict from current UI values
                current_settings = {}
                for i, (name, component) in enumerate(model_feature_components.items()):
                    current_settings[name] = feature_values[i]
                
                # Pass current settings + checkboxes + formats
                commands = app.generate_multi_model_commands_with_settings(current_settings, checkboxes, formats)
                return gr.update(visible=True, value=commands), gr.update(value="Close Command")
        
        multi_gen_cmd_btn.click(
            multi_gen_cmd_handler,
            inputs=[multi_cmd_output, *model_feature_components.values(), *multi_model_checkboxes.values(), *multi_model_formats.values()],
            outputs=[multi_cmd_output, multi_gen_cmd_btn]
        )
        
        # Run Captioning
        multi_run_btn.click(
            lambda: gr.update(value="Processing...", interactive=False),
            outputs=[multi_run_btn]
        ).then(
            app.run_multi_model_inference,
            inputs=[*multi_model_checkboxes.values(), *multi_model_formats.values()],
            outputs=[gal]
        ).then(
            lambda: gr.update(value="Run Captioning", interactive=True),
            outputs=[multi_run_btn]
        )
        
        
        
        # Data Loading (Shared)
        # Data Loading (Shared)
        load_source_btn.click(app.load_input_source, inputs=[input_path_text, g_recursive, limit_count], outputs=[gal, input_files, page_number_input, total_pages_label, pagination_row])
        clear_gallery_btn.click(app.clear_gallery, outputs=[gal, inspector_group, page_number_input, total_pages_label, pagination_row])
        input_files.change(app.load_files, inputs=[input_files], outputs=[gal, input_files, page_number_input, total_pages_label, pagination_row])
        
        # Helper to update image count displays
        def get_image_count():
            count = len(app.dataset.images) if app.dataset else 0
            return f"<center><b style='font-size: 1.2em'>{count} images</b></center>"
        
        # Chain image count updates after all load/clear actions
        load_source_btn.click(get_image_count, outputs=[image_count])
        clear_gallery_btn.click(get_image_count, outputs=[image_count])
        input_files.change(get_image_count, outputs=[image_count])

        # Gallery columns refresh (No Auto-Save)
        def refresh_cols(val):
            # Update memory only
            app.gallery_columns = val
            # Recalculate height since columns changed
            return gr.update(columns=val, height=app.calc_gallery_height(), value=app._get_gallery_data())

        gal_cols.change(refresh_cols, inputs=[gal_cols], outputs=[gal])
        
        # Gallery rows update gallery visibility with refresh and height calc (No Auto-Save)
        def refresh_vis_rows(val):
            # Update memory only
            app.gallery_rows = val
            
            is_visible = val > 0
            pixel_height = app.calc_gallery_height()
            
            # IMPORTANT: Return current data to prevent clearing when reshaping
            return (
                gr.update(visible=is_visible),  # gallery_group
                gr.update(height=pixel_height, value=app._get_gallery_data()),   # gal
            )
        
        gal_rows_slider.change(refresh_vis_rows, inputs=[gal_rows_slider], outputs=[gallery_group, gal])
        
        # Preset selection trigger
        if 'instruction_template' in model_feature_components and 'task_prompt' in model_feature_components:
            model_feature_components['instruction_template'].change(
                app.apply_preset, 
                inputs=[model_sel, model_feature_components['instruction_template']], 
                outputs=[model_feature_components['task_prompt']]
            )

        # Pagination Events
        # We need app methods that return (gallery, page_num, total_label, row_vis)
        # Note: We update visibility on all paging actions to be safe/consistent
        prev_btn.click(app.prev_page, inputs=None, outputs=[gal, page_number_input, total_pages_label, pagination_row])
        next_btn.click(app.next_page, inputs=None, outputs=[gal, page_number_input, total_pages_label, pagination_row])
        
        # Jump to page
        page_number_input.submit(app.jump_to_page, inputs=[page_number_input], outputs=[gal, page_number_input, total_pages_label, pagination_row])
        
        # Update items per page (from Settings tab) causes refresh
        items_per_page.change(app.update_items_per_page, inputs=[items_per_page], outputs=[gal, page_number_input, total_pages_label, pagination_row])

        # Inspector actions
        gal.select(app.open_inspector, outputs=[inspector_group, insp_tabs, insp_img, insp_video, insp_cap])
        save_cap_btn.click(app.save_and_close, inputs=[insp_cap], outputs=[gal, inspector_group])
        close_insp_btn.click(app.close_inspector, outputs=[inspector_group])
        insp_remove_btn.click(
            app.remove_from_gallery,
            outputs=[gal, inspector_group, page_number_input, total_pages_label, pagination_row]
        ).then(
            get_image_count,
            outputs=[image_count]
        )
        
        # =========================================
        # SESSION PERSISTENCE (Page Refresh)
        # =========================================
        # Force re-read config and update all UI components when the page loads.
        # Prompt source visibility toggles without needing server restart.
        # IMPORTANT: Including model_sel as output triggers the change event, loading model features
        demo.load(
            update_model_settings_ui,
            inputs=[model_sel],
            outputs=model_settings_outputs  # Use same outputs list as model_sel.change (includes model_description)
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
        
        
        # Multi-model settings reload on page refresh
        def load_multi_model_ui_settings():
            """Load saved multi-model settings for UI components."""
            saved_settings = app.load_multi_model_settings()
            updates = []
            # First all checkboxes (enabled status)
            for idx in range(len(app.models)):
                enabled, _ = saved_settings[idx]
                updates.append(gr.update(value=enabled))
            # Then all format inputs
            for idx in range(len(app.models)):
                _, format_ext = saved_settings[idx]
                updates.append(gr.update(value=format_ext))
            return updates
        
        demo.load(
            load_multi_model_ui_settings,
            inputs=None,
            outputs=[*multi_model_checkboxes.values(), *multi_model_formats.values()]
        )

        # Model Order Move Buttons
        move_up_btn.click(
            app.move_model_up,
            inputs=[model_order_radio, model_order_state],
            outputs=[model_order_radio, model_order_textbox, model_order_state]
        )
        
        move_down_btn.click(
            app.move_model_down,
            inputs=[model_order_radio, model_order_state],
            outputs=[model_order_radio, model_order_textbox, model_order_state]
        )

        # Settings Tab Buttons
        settings_save_btn.click(
            app.save_settings_simple,
            inputs=[vram_inp, models_chk, gal_cols, gal_rows_slider, g_unload_model, model_order_textbox, items_per_page],
            outputs=[model_sel, models_chk, model_order_radio] + list(multi_model_checkboxes.values()) + list(multi_model_formats.values())
        )
        
        # Reset to Defaults - requires manual page refresh after reset
        def reset_handler():
            success, message = app.reset_to_defaults()
            if success:
                gr.Info(message)
            else:
                gr.Warning(message)
        
        settings_reset_btn.click(
            reset_handler,
            outputs=[]
        )

        if startup_message:
            def show_startup_msg():
                # Use a global/module attribute to track if we've shown the message
                # attached to the app instance or similar, but since we are in a closure...
                # We can use a distinct attribute on the function itself or a mutable default
                if not hasattr(show_startup_msg, "shown"):
                    gr.Info(startup_message)
                    show_startup_msg.shown = True
            
            demo.load(show_startup_msg, outputs=[])

    return demo
