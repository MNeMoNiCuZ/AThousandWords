# gui/main.py
"""
Main UI creation function for the A Thousand Words.
Refactored for simplified settings workflow (Phase 3).
"""

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
                    initial_description_html = get_model_description_html(app, app.current_model_id)
                    model_description = gr.Markdown(
                        value=initial_description_html,
                        elem_classes="model-description"
                    )
                    # Presets Tracker State (Global for UI)
                    presets_tracker = gr.State(value=0)

                    # Pre-define models_chk to avoid UnboundLocalError in Presets Tab
                    # We render it later in the Settings tab
                    models_chk = gr.CheckboxGroup(
                        choices=app.models, 
                        value=app.enabled_models, 
                        label="Enabled Models", 
                        info="Select which models appear in the model dropdown",
                        render=False  # IMPORTANT: Do not render yet
                    )

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
                    
                    @gr.render(inputs=[model_sel, model_version_dropdown, presets_tracker])
                    def render_features(model_id, model_version, tracker):
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
                        
                        # Build set of supported features for this model
                        supported_features = set()
                        for f in config.get('features', []):
                            if isinstance(f, str):
                                supported_features.add(f)
                            elif isinstance(f, dict) and 'name' in f:
                                supported_features.add(f['name'])
                        
                        # Use shared helper to ensure consistency with state initializer
                        # Pass model_version from dropdown if available (for re-render on version change)
                        current_values = _resolve_values(model_id, model_version)
                        
                        # Initialize settings_state with current values for this model
                        # This ensures dynamic features are available when generating commands
                        settings_state.value = current_values.copy()
                        
                        # 3. Render Rows
                        # Create a map to track created components for wiring events later
                        # Component Map for debugging/access
                        component_map = {}
                        
                        # Universal State Update Handler
                        def update_state_handler(val, current_state, name):
                            if current_state is None:
                                current_state = {}
                            
                            
                            current_state[name] = val
                            
                            return current_state
                        


                        for row_features in resolved_rows:
                            if not row_features: continue
                            
                            # check if any features in this row are NOT model_version/batch/tokens
                            # CRITICAL: Always exclude universal/static features from dynamic rendering
                            IGNORED_FEATURES = {'model_version', 'batch_size', 'max_tokens'}
                            valid_features = [f for f in row_features if f.strip() not in IGNORED_FEATURES]
                            if not valid_features: continue

                            # Pre-process feature definitions for this model to support lookups
                            feature_defs = {f['name']: f for f in config.get('features', []) if isinstance(f, dict) and 'name' in f}

                            with gr.Row():
                                for feature_name in valid_features:
                                    # CRITICAL: Only render features supported by the model
                                    # Double check against IGNORED just in case
                                    if feature_name in IGNORED_FEATURES:
                                        continue
                                        
                                    # Relax check: Allow layout to dictate features (e.g. system_prompt in presets)
                                    # if feature_name not in supported_features:
                                    #    continue

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
                                    
                                    # --- INITIAL VISIBILITY LOGIC ---
                                    # Check generic 'visible_if' from configuration
                                    feat_def = feature_defs.get(feature_name, {})
                                    if 'visible_if' in feat_def:
                                        condition = feat_def['visible_if']
                                        # Simple parser for initial state
                                        import re
                                        match = re.match(r"(\w+)\s*(==|!=)\s*['\"](.+)['\"]", condition)
                                        if match:
                                            source, op, val = match.groups()
                                            source_val = current_values.get(source)
                                            # Default to visible if source value missing (safe)
                                            if source_val is not None:
                                                if op == '==':
                                                    feat_conf['visible'] = (str(source_val) == val)
                                                elif op == '!=':
                                                    feat_conf['visible'] = (str(source_val) != val)

                                    # SAFETY: Explicitly force separate labels for Prompts to avoid confusion
                                    if feature_name == "task_prompt":
                                        feat_conf['label'] = "Task Prompt"
                                        # Check if model supports custom prompts
                                        if config.get('supports_custom_prompts', True) is False:
                                            feat_conf['interactive'] = False
                                            # Use Code component to ensure special tokens (like <DETAILED_CAPTION>) are visible
                                            feat_conf['type'] = 'code'
                                            
                                            # Optional: Add info
                                            # feat_conf['info'] = "This model requires specific prompts. You cannot edit this field manually."
                                    elif feature_name == "system_prompt":
                                        feat_conf['label'] = "System Prompt"

                                    # Check visibility dependency for THIS component
                                    # If this component depends on prompt_source, check that value
                                    ps_val = current_values.get("prompt_source", "Prompt Presets")
                                    
                                    # Visibility Logic (Mirroring the handler)
                                    is_visible = True
                                    if feature_name in ["prompt_presets", "task_prompt"]:
                                        if ps_val != "Prompt Presets": is_visible = False
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
                                        feat_conf['choices'] = ["Prompt Presets", "From File", "From Metadata"]
                                        feat_conf['allow_custom_value'] = False # Force selection from list
                                    
                                    # 2. Prompt Presets (Version-Specific)
                                    elif feature_name == "prompt_presets":
                                        # Get current version for proper preset resolution
                                        current_version = current_values.get('model_version')
                                        presets = app.config_mgr.get_version_prompt_presets(model_id, current_version)
                                        choices = list(presets.keys()) if presets else []
                                        
                                        feat_conf['choices'] = choices
                                    
                                    # Populate Dynamic Choices
                                    elif feature_name == "model_mode":
                                        feat_conf['choices'] = config.get('model_modes', [])
                                    elif feature_name == "caption_length":
                                        feat_conf['choices'] = config.get('caption_lengths', [])
                                    elif feature_name == "model_version":
                                        # Already handled but good for completeness
                                        feat_conf['choices'] = list(config.get('model_versions', {}).keys())

                                    # Create Component
                                    # Create Component
                                    comp = create_component_from_feature_config(feat_conf)
                                    
                                    # Register component in map
                                    component_map[feature_name] = comp

                                    # Bind Change Event to Update State
                                    # Use functools.partial to safely capture feature_name loop variable
                                    if feature_name != "prompt_presets":
                                        comp.change(
                                            fn=functools.partial(update_state_handler, name=feature_name),
                                            inputs=[comp, settings_state],
                                            outputs=[settings_state]
                                        )
                        
                        # 4. Wire Up Conditional Visibility (Generic Handler)
                        # This handles "visible_if" properties defined in model YAML
                        # Format: "dependency == 'value'" or "dependency != 'value'"
                        
                        # 1. Build Dependency Map: source -> [(target_name, condition_str), ...]
                        visibility_deps = {}
                        feature_config_list = config.get('features', [])
                        
                        # Convert config list to dict for easier lookup if needed, 
                        # but we just need to iterate to find visible_if rules.
                        # We also need to inspect feature_rows since some features might be in presets (like prompt_control)
                        # Actually, 'features' list in YAML is the definition source.
                        
                        for feat_def in feature_config_list:
                            if isinstance(feat_def, dict) and 'visible_if' in feat_def:
                                target_name = feat_def['name']
                                condition = feat_def['visible_if']
                                
                                # Parse generic condition: "source op 'value'"
                                # Simple parser for == and !=
                                import re
                                match = re.match(r"(\w+)\s*(==|!=)\s*['\"](.+)['\"]", condition)
                                if match:
                                    source, op, val = match.groups()
                                    if source in component_map and target_name in component_map:
                                        if source not in visibility_deps:
                                            visibility_deps[source] = []
                                        visibility_deps[source].append({
                                            'target': target_name,
                                            'op': op,
                                            'val': val
                                        })

                        # 2. Wire Events for each Source
                        for source_name, dependants in visibility_deps.items():
                            source_comp = component_map[source_name]
                            targets = [component_map[d['target']] for d in dependants]
                            
                            # Closure to capture the specific dependants list for this source
                            def update_visibility_generic(source_val, deps=dependants):
                                updates = []
                                for dep in deps:
                                    op = dep['op']
                                    target_val = dep['val']
                                    
                                    visible = False
                                    if op == '==':
                                        visible = (str(source_val) == target_val)
                                    elif op == '!=':
                                        visible = (str(source_val) != target_val)
                                    
                                    updates.append(gr.update(visible=visible))
                                return updates

                            source_comp.change(
                                fn=update_visibility_generic,
                                inputs=[source_comp],
                                outputs=targets
                            )
                            
                            # 3. Trigger Initial State
                            # We need to apply the visibility immediately based on current value
                            current_source_val = current_values.get(source_name)
                            if current_source_val is not None:
                                initial_updates = update_visibility_generic(current_source_val)
                                # We can't return these updates during render, but we can set 
                                # the initial 'visible' property on the components since we have references!
                                # Wait, component_map values are already created Gradio components. 
                                # We cannot change their properties directly after creation easily without an update?
                                # Actually, we can just rely on the fact that we set 'visible' 
                                # during creation in the earlier loop IF we evaluate it there.
                                # But we didn't evaluate generic rules there yet.
                                
                                # BETTER APPROACH for INITIAL STATE:
                                # Re-evaluate the 'visible' logic in the creation loop?
                                # OR: Just accept they might flicker? No, flickering is bad.
                                # We should update the 'visible' prop on the component object if possible.
                                # Gradio components are objects. `comp.visible` might be settable?
                                # No, usually read-only or requires update().
                                
                                # Let's try to set it via update() on load? No load event here.
                                # The cleanest way is to evaluate this logic *inside* the creation loop 
                                # to set the initial `visible` flag correctly.
                                pass

                        # 4. Wire Up Conditional Visibility (Legacy/Specific Handler)
                        if "prompt_source" in component_map:
                            ps_comp = component_map["prompt_source"]
                            
                            # Identify target components that exist for this model
                            target_features = [
                                "prompt_presets", "task_prompt",           # Prompt Presets Mode
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
                                        if source_val == "Prompt Presets":
                                            if name in ["prompt_presets", "task_prompt"]: visible = True
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
                                initial_ps_val = current_values.get("prompt_source", "Prompt Presets")
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

                        # Wire Up Prompt Presets -> Task Prompt
                        if "prompt_presets" in component_map and "task_prompt" in component_map:
                            it_comp = component_map["prompt_presets"]
                            tp_comp = component_map["task_prompt"]
                            
                            # Get version-specific presets for this model
                            current_version = current_values.get('model_version')
                            presets = app.config_mgr.get_version_prompt_presets(model_id, current_version)
                            
                            def handle_preset_change(template_name, current_state):
                                if current_state is None:
                                    current_state = {}
                                
                                logger.info(f"Preset Change: {template_name}")
                                
                                # ALWAYS update the prompt_presets value in state first
                                # This replaces the separate update_state handler to avoid race conditions
                                current_state['prompt_presets'] = template_name

                                if not template_name or template_name not in presets:
                                    return gr.update(), current_state # No change
                                
                                new_prompt = presets[template_name]
                                current_state['task_prompt'] = new_prompt
                                
                                return new_prompt, current_state

                            it_comp.change(
                                fn=handle_preset_change,
                                inputs=[it_comp, settings_state],
                                outputs=[tp_comp, settings_state]
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
                        
                        # Build supported features set
                        supported_features = set()
                        for f in config.get('features', []):
                            if isinstance(f, str):
                                supported_features.add(f)
                            elif isinstance(f, dict) and 'name' in f:
                                supported_features.add(f['name'])
                                
                        IGNORED_FEATURES = {'model_version', 'batch_size', 'max_tokens'}
                        
                        # VALIDATION: Check if model actually supports versions
                        model_versions = config.get('model_versions', {})
                        model_supports_versions = bool(model_versions)
                        valid_versions = list(model_versions.keys()) if model_supports_versions else []
                        
                        # 1. Determine Current Version with VALIDATION
                        # Priority: Override (Dropdown) > Saved Active Version > Defaults
                        defaults_raw = config.get('defaults', {})
                        
                        # Fix for NameError: Load clean defaults raw for fallback logic
                        clean_config = app.config_mgr.get_model_defaults(model_id)
                        clean_defaults_raw = clean_config.get('defaults', {})
                        
                        if version_override:
                            current_version = version_override
                        else:
                            # Try to get saved active version
                            # In new style, it's at root. In old style, it's also at root.
                            saved_ver = saved_model_root.get('model_version')
                            
                            # CRITICAL VALIDATION: Check if saved version is valid
                            if saved_ver:
                                if not model_supports_versions:
                                    # Model doesn't support versions, clear the stale value
                                    logging.warning(f"Model '{model_id}' does not support versions, but 'model_version: {saved_ver}' was saved. Clearing invalid value.")
                                    # Remove from user_config
                                    if 'model_settings' in app.config_mgr.user_config and model_id in app.config_mgr.user_config['model_settings']:
                                        app.config_mgr.user_config['model_settings'][model_id].pop('model_version', None)
                                        app.config_mgr.save_user_config()
                                    saved_ver = None  # Ignore it
                                elif saved_ver not in valid_versions:
                                    # Saved version is invalid for this model
                                    logging.warning(f"Invalid model_version '{saved_ver}' for model '{model_id}'. Valid versions: {valid_versions}. Resetting to default.")
                                    # Remove from user_config
                                    if 'model_settings' in app.config_mgr.user_config and model_id in app.config_mgr.user_config['model_settings']:
                                        app.config_mgr.user_config['model_settings'][model_id].pop('model_version', None)
                                        app.config_mgr.save_user_config()
                                    saved_ver = None  # Ignore it
                            
                            if saved_ver:
                                current_version = saved_ver
                            else:
                                # Fallback to Defaults
                                if isinstance(clean_defaults_raw, dict):
                                    # Check if nested defaults (has string keys with dict values)
                                    first_key = next(iter(clean_defaults_raw), None)
                                    if first_key and isinstance(first_key, str) and isinstance(clean_defaults_raw.get(first_key), dict):
                                        # Nested structure - use first version as default
                                        current_version = first_key
                                    else:
                                        # Flat structure - extract version from defaults
                                        current_version = clean_defaults_raw.get('model_version')
                                else:
                                    current_version = None

                        # 2. Get Defaults for this version (from the clean config)
                        defaults = app.config_mgr._resolve_version_specific(clean_defaults_raw, current_version)
                        if isinstance(defaults, dict):
                            defaults = defaults.copy() # CRITICAL: Copy to avoid mutating global config
                        else:
                            defaults = {}
                            
                        
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
                        
                        # 4.5 Ensure ALL supported features have a value (from FeatureRegistry defaults if needed)
                        # This feature scrubbing ensures robust state initialization
                        for feature_name in supported_features:
                            if feature_name not in values and feature_name not in IGNORED_FEATURES:
                                feature = feature_registry.get_feature(feature_name)
                                if feature:
                                    values[feature_name] = feature.get_default()

                        # Smart Resolution: Enforce Task Prompt Rules
                        # 1. If 'prompt_presets' is supported but missing/empty (no saved value, no default), 
                        #    default to the first available preset option.
                        # 2. If 'task_prompt' is not strictly saved by user (even if empty in saved diff),
                        #    overwrite it with the text from the active preset.
                        
                        task_prompt_is_saved = "task_prompt" in saved_diff
                        
                        # Handle missing/empty prompt_presets by picking first available
                        current_preset_val = values.get("prompt_presets")
                        if not current_preset_val:
                             has_presets_feature = "prompt_presets" in supported_features
                             
                             if has_presets_feature:
                                 current_version = values.get('model_version')
                                 presets = app.config_mgr.get_version_prompt_presets(model_id, current_version)
                                 
                                 if presets:
                                     first_preset = next(iter(presets))
                                     values["prompt_presets"] = first_preset
                        
                        # Derive Task Prompt from Preset if not explicitly saved (or if saved as empty/cleared)
                        # Note: We check 'task_prompt_val' to treat an empty saved prompt as "reset to preset"
                        task_prompt_val = values.get("task_prompt", "")
                        
                        # Enforce Custom Prompt Support Rules
                        supports_custom_prompts = config.get('supports_custom_prompts', True)
                        
                        # If custom prompts are NOT supported, we MUST ignore any saved value that might contradict the preset
                        # effectively forcing the prompt to match the preset.
                        if not supports_custom_prompts:
                            task_prompt_is_saved = False # Pretend it's not saved to force overwrite
                        
                        if "prompt_presets" in values and (not task_prompt_is_saved or not task_prompt_val):
                            template = values["prompt_presets"]
                            
                            current_version = values.get('model_version')
                            presets = app.config_mgr.get_version_prompt_presets(model_id, current_version)
                            if presets and template in presets:
                                resolved_prompt = presets[template]
                                values["task_prompt"] = resolved_prompt
                                # logger.debug(f"Auto-resolved Task Prompt for '{model_id}': Preset='{template}'")
                                
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
                    generate_command_btn = gr.Button("Generate Command", variant="secondary", scale=0)
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
                command_output = gr.Textbox(
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
                        gr.Markdown("""
                        ### 🏷️ Create Captions from Metadata
                        Extract metadata (User Comment, PNG Info, etc.) from images to create captions.
                        Useful for restoring prompts from generated images.
                        """)
                        with gr.Row():
                            meta_src = gr.Dropdown(["all", "png_info", "exif"], label="Source", value="all", info="Which metadata field to search.")
                            meta_out_dir = gr.Textbox(label="Output Directory", placeholder="Optional. Leave empty to save in same folder.", info="Path to save text files.")
                            meta_ext = gr.Textbox(label="Output Extension", value="txt", placeholder="txt", info="Extension for caption files.")
                            meta_upd = gr.Checkbox(label="Overwrite existing captions", value=True)
                        
                        with gr.Row():
                            meta_clean = gr.Checkbox(label="Clean Text", value=True, info="Remove extra spaces")
                            meta_collapse = gr.Checkbox(label="Collapse Newlines", value=True, info="Merge paragraphs")
                            meta_norm = gr.Checkbox(label="Normalize Text", value=True, info="Fix punctuation")
                            
                        with gr.Row():
                            meta_pre = gr.Textbox(label="Prefix", placeholder="Added to start...", lines=1)
                            meta_suf = gr.Textbox(label="Suffix", placeholder="Added to end...", lines=1)

                        meta_run = gr.Button("Extract Metadata", variant="primary")
                        
                    with gr.Tab("Resize"):
                        gr.Markdown("""
                        ### 📐 Resize Images
                        Batch resize all loaded images to a maximum dimension while preserving aspect ratio.
                        Images smaller than the target are NOT upscaled.
                        
                        > [!CAUTION]
                        > **Irreversible Action**: Resizing images is a destructive operation if you overwrite the originals. 
                        > It is highly recommended to use a separate Output Directory or Prefix/Suffix.
                        """)
                        with gr.Row():
                            resize_out_dir = gr.Textbox(label="Output Directory", placeholder="Optional. Leave empty to save in same folder.", info="Path to save resized images.")
                            resize_pre = gr.Textbox(label="Output Filename Prefix", placeholder="Added to start...", lines=1)
                            resize_suf = gr.Textbox(label="Output Filename Suffix", placeholder="Added to end...", lines=1)
                            resize_ext = gr.Textbox(label="Output Extension", value="", placeholder="Keep Original", info="Ext (jpg, png). Empty = Keep Original.")
                            
                        with gr.Row():
                            resize_px = gr.Number(label="Max Dimension (px)", value=1024, precision=0)
                            resize_overwrite = gr.Checkbox(label="Overwrite", value=True, info="Overwrite if file exists")
                            
                        resize_run = gr.Button("Resize Images", variant="primary")


            # =========================================
            # Tab 3.5: User Presets Library
            # =========================================
            with gr.Tab("Presets"):
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
                    # CSS for compact rows and table-like styling
                    gr.HTML("""
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
                            /* No vertical separator as per user request */
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
                            background-color: #450a0a !important; /* Dark Red Background */
                            color: #fca5a5 !important; /* Light Red Icon */
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
                    """)

                    # Render Presets List Dynamically with Native Components
                    @gr.render(inputs=[presets_tracker])
                    def render_preset_list(tracker):
                        # Get data
                        rows = app.get_user_presets_dataframe()
                        
                        if not rows:
                            return gr.Markdown("*No presets found.*")
                        
                        # Header
                        with gr.Row(elem_classes="preset-row", variant="compact"):
                            # Using scale to enforce widths
                            # Model
                            with gr.Column(scale=2, min_width=200):
                                gr.Markdown("**Model**", elem_classes="preset-cell-markdown")
                            # Name
                            with gr.Column(scale=2, min_width=200):
                                gr.Markdown("**Name**", elem_classes="preset-cell-markdown")
                            # Prompt Text
                            with gr.Column(scale=6):
                                gr.Markdown("**Prompt Text**", elem_classes="preset-cell-markdown")
                            # Action
                            with gr.Column(scale=0, min_width=60): # Slightly wider container for padding
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
                                    # Markdown automatically wraps
                                    gr.Markdown(p_text, elem_classes="preset-cell-markdown")
                                with gr.Column(scale=0, min_width=60):
                                    del_btn = gr.Button("🗑️", elem_classes="preset-trash-btn", size="sm")
                                    
                                    # Bind Delete - Capturing loop variables properly
                                    def do_delete(m=p_model, n=p_name):
                                        app.delete_user_preset(m, n)
                                        return tracker + 1
                                    
                                    del_btn.click(
                                        fn=do_delete,
                                        inputs=[],
                                        outputs=[presets_tracker]
                                    )
                
                # Update Save Button to trigger tracker
                def handle_save_preset(model, name, text, tracker_val):
                     app.save_user_preset(model, name, text)
                     return tracker_val + 1
                
                preset_save_btn.click(
                    handle_save_preset,
                    inputs=[preset_model_dd, preset_name_txt, preset_prompt_txt, presets_tracker],
                    outputs=[presets_tracker]
                ).then(
                    fn=lambda: app.refresh_models(),
                    outputs=[model_sel, models_chk]
                )

            # =========================================
            # Tab 4: Settings (System)
            # =========================================
            with gr.Tab("Settings"):
                 gr.Markdown("### 🖥️ System Settings")
                 with gr.Row():
                     vram_inp = gr.Number(label="GPU VRAM (GB)", value=cfg['gpu_vram'], precision=0, info="Your GPU's VRAM for batch size recommendations")
                     
                     # System RAM for ONNX models
                     system_ram_default = cfg.get('system_ram', get_system_ram_gb())
                     system_ram_inp = gr.Number(label="System RAM (GB)", value=system_ram_default, precision=0, info="Your System RAM for batch size recommendations")
                     
                     unload_val = cfg.get('unload_model', True) # Default to True if missing
                     g_unload_model = gr.Checkbox(label="Unload Model", value=unload_val, info="Unload model from VRAM immediately after finishing.")
                     
                     items_per_page = gr.Number(label="Items Per Page", value=app.gallery_items_per_page, precision=0, minimum=1, info="Images per gallery page")
                     gal_cols = gr.Slider(2, 16, step=1, label="Gallery Columns", value=app.gallery_columns, info="Number of columns in the image gallery")
                     gal_rows_slider = gr.Slider(0, 20, step=1, label="Gallery Rows", value=app.gallery_rows, info="Rows to display (0 = hide)")

                 
                 gr.Markdown("### 📦 Model Management")
                 
                 with gr.Row():
                     with gr.Column(scale=1):
                         models_chk.render() # Render the pre-defined component here
                     
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
                     settings_reset_confirm_btn = gr.Button("⚠️ Confirm Reset", variant="stop", scale=0, visible=False)
                     settings_reset_cancel_btn = gr.Button("Cancel", variant="secondary", scale=0, visible=False)


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
                    "⚠️ <b>Note:</b> Dragged files are saved to a temporary location and outputs will be placed in the configured output folder (default: /output)."
                    "</div>"
                )
        
        with gr.Group(visible=app.gallery_rows > 0, elem_id="gallery_group") as gallery_group:
            with gr.Accordion("🖼️ Dataset Gallery", open=True) as gallery_accordion:
                # Compact Pagination Controls
                with gr.Row(elem_classes="pagination-row compact-pagination", visible=False) as pagination_row:
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
            if prompt_source_value == "Prompt Presets":
                # Show: prompt_presets, task_prompt
                # Hide: prompt_prefix, prompt_file_extension, prompt_suffix
                # Rows: prompt_mode VISIBLE, file_metadata_mode HIDDEN
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
                # Show: prompt_prefix, prompt_file_extension, prompt_suffix
                # Hide: prompt_presets, task_prompt
                # Rows: prompt_mode HIDDEN, file_metadata_mode VISIBLE
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
                # Show: prompt_prefix, prompt_suffix (NO file extension for metadata)
                # Hide: prompt_presets, task_prompt, prompt_file_extension
                # Rows: prompt_mode HIDDEN, file_metadata_mode VISIBLE (but one child hidden)
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
        
        def start_processing(is_valid):
            """
            Update UI on run start:
            1. Run Button -> "Processing...", Disabled
            2. Download Button -> Show wrapper, set button to Processing state (spinner)
            """
            if not is_valid:
                # Don't change anything, or ensure it's in ready state
                return (
                    gr.update(value="Run Captioning", interactive=True), # Keep/Revert to normal
                    gr.update(visible=False), # Hide download group if it was gonna show spinner
                    gr.update() # download btn
                )

            return (
                gr.update(value="Processing...", interactive=False),  # run_btn
                gr.update(visible=True),  # download_btn_group - SHOW IT so spinner is visible
                gr.update(
                    value=None,
                    variant="secondary",
                    interactive=False,
                    elem_classes="processing-btn" # Add processing class back?
                )
            )


        # Run Logic with Dynamic State
        # =======================================================
        # SHARED ARGUMENT BUILDER (SINGLE SOURCE OF TRUTH)
        # =======================================================
        def build_inference_args(model_id, model_ver, batch, tokens, dynamic_settings, 
                                 pre, suf, over, rec, con, unload, clean, collapse, norm, rm_cn, loop, 
                                 w, h, limit, out_dir_glob, out_fmt):
            """
            Construct the final arguments dictionary for both proper runs and CLI generation.
            Ensures 100% consistency between what is run and what is generated.
            """
            
            # 1. Get Model Config and Defaults
            model_config = app.config_mgr.get_model_config(model_id)
            model_defaults = model_config.get('defaults', {})
            
            # 2. Start with model defaults
            args_dict = model_defaults.copy()
            
            # 3. Update with dynamic settings from state (overrides defaults)
            if dynamic_settings:
                # Filter out any None values that might have snuck in (though dict.update handles them, good to be clean)
                clean_dynamic = {k: v for k, v in dynamic_settings.items() if v is not None}
                args_dict.update(clean_dynamic)
            
            # 4. Add static overrides / Global UI controls
            # These typically override model defaults as they are explicit user inputs on the main UI
            
            # Model Version
            if model_ver: 
                args_dict['model_version'] = model_ver
            
            # Batch Size & Tokens (Validation handled by inputs, but ensure int)
            if batch: args_dict['batch_size'] = int(batch)
            if tokens: args_dict['max_tokens'] = int(tokens)
            
            # Global Text Processing
            if pre: args_dict['prefix'] = pre
            if suf: args_dict['suffix'] = suf
            if clean is not None: args_dict['clean_text'] = clean
            if collapse is not None: args_dict['collapse_newlines'] = collapse
            if norm is not None: args_dict['normalize_text'] = norm
            if rm_cn is not None: args_dict['remove_chinese'] = rm_cn
            if loop is not None: args_dict['strip_loop'] = loop
            
            # Global Image Processing
            args_dict['max_width'] = int(w) if w else None
            args_dict['max_height'] = int(h) if h else None
            
            # Global Execution Flags
            if over is not None: args_dict['overwrite'] = over
            if rec is not None: args_dict['recursive'] = rec
            if con is not None: args_dict['print_console'] = con
            if unload is not None: args_dict['unload_model'] = unload
            
            # Output Settings
            if out_dir_glob: args_dict['output_dir'] = out_dir_glob
            if out_fmt: args_dict['output_format'] = out_fmt
            if limit: args_dict['limit_count'] = limit

            # Add Global Settings (VRAM)
            g_set = app.config_mgr.get_global_settings()
            args_dict['gpu_vram'] = g_set.get('gpu_vram', 24)
            
            return args_dict

        # =======================================================
        # RUN LOGIC
        # =======================================================
        def run_with_dynamic_state(*args):
            # Args unpacking (Last arg is validation state)
            is_valid = args[-1]
            if not is_valid:
                 # Return "Reset" state essentially, mimicking the return of a successful run but without doing work
                 # Return signature: gallery_data, run_btn_update, dl_grp_update, dl_btn_update
                 return gr.update(), gr.update(value="Run Captioning", interactive=True), gr.update(visible=False), gr.update(visible=False)

            # Unpack inputs corresponding to build_inference_args signature
            # Slicing: all except last (validation state)
            input_args = args[:-1]
            
            # Build the unified args dictionary
            args_dict = build_inference_args(*input_args)
            
            # Run Inference
            # app.run_inference uses self.dataset so we don't pass files here
            result = app.run_inference(args_dict.get('model_id') or app.current_model_id, args_dict)
            
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
                    gr.Warning("No files were processed. The dataset may be empty.")

                if empty > 0:
                    gr.Warning(f"{empty} captions were empty!")
            else:
                gr.Warning("Processing completed but no statistics were returned.")
            
            # Download Button Logic
            dl_grp = gr.update(visible=False)
            dl_btn = gr.update(visible=False)
            if isinstance(download_update, dict) and download_update.get('value'):
                dl_grp = gr.update(visible=True)
                dl_btn = gr.update(
                    value=download_update.get('value'),
                    visible=True,
                    interactive=True,
                    variant="primary",
                    icon="src/core/download_white.svg",
                    elem_classes="download-btn"
                )

            return gallery_data, gr.update(value="Run Captioning", interactive=True), dl_grp, dl_btn


        # =======================================================
        # GENERATE COMMAND LOGIC
        # =======================================================
        command_visible_state = gr.State(value=False)

        def generate_cli_wrapper(current_visible, *args):
            # Toggle logic
            if current_visible:
                # If currently visible, hide it params: (value, visible)
                return gr.update(visible=False, value=""), False, gr.update(value="Generate Command")
            
            # Build unified args
            args_dict = build_inference_args(*args)
            
            # Generate CLI string
            model_id = args_dict.get('model_id') or app.current_model_id
            
            # Generate full command including defaults for transparency
            cmd = app.generate_cli_command(model_id, args_dict, skip_defaults=False)
            
            return gr.update(value=cmd, visible=True), True, gr.update(value="Generate Command ▼")

        # State for validation flow control
        run_valid_state = gr.State(value=True)
        multi_run_valid_state = gr.State(value=True)

        def validate_run_state(model_id):
            """
            Validate strict requirements before starting the run flow.
            Returns False and shows Warning if conditions aren't met.
            """
            if not app.dataset or not app.dataset.images:
                gr.Warning("No media found. Please load a folder or add images to the 'Input Source'.")
                return False
            
            # Check for media type compatibility
            config = app.config_mgr.get_model_config(model_id)
            supported_media = config.get('media_type', ["Image"]) # Default to Image
            if isinstance(supported_media, str):
                supported_media = [supported_media]

            has_images = any(img.media_type == "image" for img in app.dataset.images)
            has_videos = any(img.media_type == "video" for img in app.dataset.images)
            
            # 1. Check if dataset has ANY valid files
            if not has_images and not has_videos:
                 gr.Warning("Dataset contains no valid image or video files.")
                 return False

            supports_image = "Image" in supported_media
            supports_video = "Video" in supported_media

            # RELAXED VALIDATION:
            # We trust the base wrapper to handle conversion (Video -> Image) if needed.
            # So we don't strictly block Video input for Image-only models anymore.
            
            # However, if Model is Video-Only (rare/theoretical) and we only have Images, 
            # we typically can't "convert" Image -> Video easily/meaningfully for captioning in the same way.
            # But mostly VLM models that support Video also support Image.
            
            if supports_video and not supports_image and not has_videos and has_images:
                 gr.Warning(f"Model '{model_id}' only supports Video, but dataset contains only Images.")
                 return False
            
            return True

        # Wire Up Run Button
        run_btn.click(
            validate_run_state,
            inputs=[model_sel],
            outputs=[run_valid_state]
        ).then(
            start_processing,
            inputs=[run_valid_state],
            outputs=[run_btn, download_btn_group, download_btn]
        ).then(
            run_with_dynamic_state,
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
            show_progress="minimal"
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
        # NOTE: feature rendering is handled by @gr.render above.
        # We only need to update the default values/description in the static UI parts.
        
        # We DO NOT want to call update_model_ui because it returns a fixed list of 11 updates
        # which blindly target dynamic components by index, causing label corruption.
        # The re-render logic handles all feature initialization correctly.
        
        model_sel.change(
            fn=lambda m: gr.update(value=get_model_description_html(app, m)),
            inputs=[model_sel],
            outputs=[model_description]
        )
        
        
        
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

        # Main Save Settings Button (Captioning Tab)
        save_btn.click(
            app.save_settings,
            inputs=[
                vram_inp, models_chk, gal_cols, gal_rows_slider, limit_count,
                out_dir, out_fmt, g_over, g_recursive, g_console, g_unload_model,
                pre_text, suf_text, g_clean, g_collapse, g_normalize, g_remove_chinese, g_strip_loop,
                g_max_width, g_max_height,
                model_sel, model_version_dropdown, batch_size_input, max_tokens_input,
                settings_state, items_per_page
            ],
            outputs=[]
        )

        # =========================================
        # Multi-Model Tab Event Bindings
        # =========================================
        
        # Select/Deselect All
        def set_all_multi_models(val):
            return [gr.update(value=val) for _ in multi_model_checkboxes]
            
        multi_select_all_btn.click(
            fn=lambda: set_all_multi_models(True),
            inputs=[],
            outputs=list(multi_model_checkboxes.values())
        )
        
        multi_deselect_all_btn.click(
            fn=lambda: set_all_multi_models(False),
            inputs=[],
            outputs=list(multi_model_checkboxes.values())
        )

        # Save Settings
        multi_save_btn.click(
            app.save_multi_model_settings,
            inputs=[*multi_model_checkboxes.values(), *multi_model_formats.values()],
            outputs=[]
        )

        # Generate Command
        def gen_multi_cmd_wrapper(*args):
             cmd = app.generate_multi_model_commands(*args)
             return gr.update(value=cmd, visible=True)
             
        multi_gen_cmd_btn.click(
            gen_multi_cmd_wrapper,
            inputs=[*multi_model_checkboxes.values(), *multi_model_formats.values()],
            outputs=[multi_cmd_output]
        )

        # Run Captioning
        def validate_multi_run():
            if not app.dataset or not app.dataset.images:
                gr.Warning("No media loaded. Please load a folder first.")
                return False
            return True

        multi_run_btn.click(
            validate_multi_run,
            inputs=[],
            outputs=[multi_run_valid_state]
        ).then(
            start_multi_processing,
            inputs=[multi_run_valid_state],
            outputs=[multi_run_btn]
        ).then(
            run_multi_wrapper,
            # Inputs: checkboxes..., formats..., limit_count, valid_state
            inputs=[*multi_model_checkboxes.values(), *multi_model_formats.values(), limit_count, multi_run_valid_state],
            outputs=[gal]
        ).then(
            lambda: gr.update(value="Run Captioning", interactive=True),
            outputs=[multi_run_btn]
        )

        # Settings Tab Buttons
        settings_save_btn.click(
            app.save_settings_simple,
            inputs=[vram_inp, system_ram_inp, models_chk, gal_cols, gal_rows_slider, g_unload_model, model_order_textbox, items_per_page],
            outputs=[model_sel, models_chk, model_order_radio] + list(multi_model_checkboxes.values()) + list(multi_model_formats.values())
        )

        # =========================================
        # Multi-Model Tab Event Bindings
        # =========================================
        
        # Select/Deselect All
        def set_all_multi_models(val):
            return [gr.update(value=val) for _ in multi_model_checkboxes]
            
        multi_select_all_btn.click(
            fn=lambda: set_all_multi_models(True),
            inputs=[],
            outputs=list(multi_model_checkboxes.values())
        )
        
        multi_deselect_all_btn.click(
            fn=lambda: set_all_multi_models(False),
            inputs=[],
            outputs=list(multi_model_checkboxes.values())
        )

        # Save Settings
        multi_save_btn.click(
            app.save_multi_model_settings,
            inputs=[*multi_model_checkboxes.values(), *multi_model_formats.values()],
            outputs=[]
        )

        # Generate Command
        multi_command_visible_state = gr.State(value=False)

        def gen_multi_cmd_wrapper(current_visible, *args):
             if current_visible:
                 return gr.update(value="", visible=False), False, gr.update(value="Generate Command")
             
             cmd = app.generate_multi_model_commands(*args)
             return gr.update(value=cmd, visible=True), True, gr.update(value="Generate Command ▼")
             
        multi_gen_cmd_btn.click(
            gen_multi_cmd_wrapper,
            inputs=[multi_command_visible_state, *multi_model_checkboxes.values(), *multi_model_formats.values()],
            outputs=[multi_cmd_output, multi_command_visible_state, multi_gen_cmd_btn]
        )

        # Run Captioning
        def validate_multi_run():
            if not app.dataset or not app.dataset.images:
                gr.Warning("No media loaded. Please load a folder first.")
                return False
            return True

        multi_run_btn.click(
            validate_multi_run,
            inputs=[],
            outputs=[multi_run_valid_state]
        ).then(
            start_multi_processing,
            inputs=[multi_run_valid_state],
            outputs=[multi_run_btn]
        ).then(
            run_multi_wrapper,
            # Inputs: checkboxes..., formats..., limit_count, valid_state
            inputs=[*multi_model_checkboxes.values(), *multi_model_formats.values(), limit_count, multi_run_valid_state],
            outputs=[gal]
        ).then(
            lambda: gr.update(value="Run Captioning", interactive=True),
            outputs=[multi_run_btn]
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

        # Wire Up Tools
        meta_run.click(
            app.run_metadata,
            inputs=[meta_src, meta_upd, meta_pre, meta_suf, meta_clean, meta_collapse, meta_norm, meta_out_dir, meta_ext],
            outputs=[gal]
        )
        resize_run.click(
            app.run_resize,
            inputs=[resize_px, resize_out_dir, resize_pre, resize_suf, resize_ext, resize_overwrite],
            outputs=[gal]
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
