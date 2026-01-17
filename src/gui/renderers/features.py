"""Dynamic feature renderer for model-specific UI components."""

import gradio as gr
import functools
import logging
import re
import src.features as feature_registry
from src.gui.dynamic_components import create_component_from_feature_config
from src.gui.logic.model_logic import resolve_model_values

logger = logging.getLogger("GUI.Renderers")

def render_features_content(app, model_id, model_version, tracker, settings_state):
    """
    Constructs and wires a dynamic set of Gradio UI components for the given model's feature layout and initializes feature state.
    
    Builds per-feature GUI configurations from model config and current values, creates components, applies feature overrides and initial visibility rules (including `visible_if` conditions and prompt-source-specific visibility), and registers change handlers to:
    - update a shared settings state when feature values change,
    - dynamically toggle visibility of dependent components,
    - propagate selected prompt presets into the task prompt.
    
    Parameters:
        app: Application object providing config manager and registries used to resolve feature rows, model config, and prompt presets.
        model_id (str): Identifier of the model whose features should be rendered. If falsy, the function returns early.
        model_version (str): Model version used to resolve current values and presets.
        tracker: Telemetry/tracking object (passed through but not documented further).
        settings_state: Mutable Gradio state object that will be initialized with a copy of the current feature values and updated by component callbacks.
    
    """
    if not model_id:
        return
    
    # Resolve layout
    resolved_rows = app.config_mgr.resolve_feature_rows(model_id)
    if resolved_rows is None:
        resolved_rows = []
    
    # Get Config & Defaults
    config = app.config_mgr.get_model_config(model_id)
    
    # Resolve current values
    current_values = resolve_model_values(app, model_id, model_version)
    
    # Initialize state
    settings_state.value = current_values.copy()
    
    # Render Rows
    component_map = {}
    
    # Universal State Update Handler
    def update_state_handler(val, current_state, name):
        """
        Update the settings state by assigning a value for a specific feature name.
        
        Parameters:
            val: The value to set for the feature.
            current_state (dict or None): Existing state mapping feature names to values. If None, a new dict is created.
            name (str): The feature name/key to update in the state.
        
        Returns:
            state (dict): The updated state dictionary with `name` set to `val`.
        """
        if current_state is None:
            current_state = {}
        current_state[name] = val
        return current_state

    # Helper to parse visibility conditions
    def parse_condition(cond_str):
        # simple parser for "source op 'value'"
        """
        Parse a simple condition string of the form "source op 'value'" into its components.
        
        Parameters:
        	cond_str (str): Condition expression using a single-word source, `==` or `!=` operator, and a quoted value.
        
        Returns:
        	tuple: `(source, op, value)` where `source` is the identifier, `op` is either `"=="` or `"!="`, and `value` is the unquoted string; `None` if the input does not match the expected pattern.
        """
        match = re.match(r"(\w+)\s*(==|!=)\s*['\"](.+)['\"]", cond_str)
        if match:
            return match.groups()
        return None

    # Pre-process feature definitions for lookups
    feature_defs = {f['name']: f for f in config.get('features', []) if isinstance(f, dict) and 'name' in f}

    for row_features in resolved_rows:
        if not row_features: continue
        
        IGNORED_FEATURES = {'model_version', 'batch_size', 'max_tokens'}
        valid_features = [f for f in row_features if f.strip() not in IGNORED_FEATURES]
        if not valid_features: continue

        with gr.Row():
            for feature_name in valid_features:
                if feature_name in IGNORED_FEATURES: continue
                
                feature = feature_registry.get_feature(feature_name)
                if not feature: continue
                
                # Create config with current value
                feat_conf = feature.get_gui_config()
                
                if feature_name in current_values:
                    feat_conf['value'] = current_values[feature_name]
                    
                overrides = config.get('feature_overrides', {}).get(feature_name, {})
                feat_conf.update(overrides)
                
                # Initial Visibility Check (simple generic)
                feat_def = feature_defs.get(feature_name, {})
                if 'visible_if' in feat_def:
                    parts = parse_condition(feat_def['visible_if'])
                    if parts:
                        source, op, val = parts
                        source_val = current_values.get(source)
                        if source_val is not None:
                            if op == '==':
                                feat_conf['visible'] = (str(source_val) == val)
                            elif op == '!=':
                                feat_conf['visible'] = (str(source_val) != val)

                # Special Cases for Labels/Visibility
                if feature_name == "task_prompt":
                    feat_conf['label'] = "Task Prompt"
                    if config.get('supports_custom_prompts', True) is False:
                        feat_conf['interactive'] = False
                        feat_conf['type'] = 'code'
                elif feature_name == "system_prompt":
                    feat_conf['label'] = "System Prompt"

                # Prompt Source Visibility Logic (Initial)
                ps_val = current_values.get("prompt_source", "Prompt Presets")
                is_visible = True
                if feature_name in ["prompt_presets", "task_prompt"]:
                    if ps_val != "Prompt Presets": is_visible = False
                elif feature_name == "prompt_file_extension":
                    if ps_val != "From File": is_visible = False
                elif feature_name in ["prompt_prefix", "prompt_suffix"]:
                    if ps_val not in ["From File", "From Metadata"]: is_visible = False
                
                # Combine visibility
                if 'visible' not in feat_conf:
                    feat_conf['visible'] = is_visible
                else:
                    feat_conf['visible'] = feat_conf['visible'] and is_visible

                # Dynamic Choices
                if feature_name == "prompt_source":
                    feat_conf['choices'] = ["Prompt Presets", "From File", "From Metadata"]
                    feat_conf['allow_custom_value'] = False
                elif feature_name == "prompt_presets":
                    presets = app.config_mgr.get_version_prompt_presets(model_id, current_values.get('model_version'))
                    feat_conf['choices'] = list(presets.keys()) if presets else []
                elif feature_name == "model_mode":
                    feat_conf['choices'] = config.get('model_modes', [])
                elif feature_name == "caption_length":
                    feat_conf['choices'] = config.get('caption_lengths', [])
                elif feature_name == "model_version":
                    feat_conf['choices'] = list(config.get('model_versions', {}).keys())

                # Create Component
                comp = create_component_from_feature_config(feat_conf)
                component_map[feature_name] = comp

                # Bind Change Event
                if feature_name != "prompt_presets":
                    comp.change(
                        fn=functools.partial(update_state_handler, name=feature_name),
                        inputs=[comp, settings_state],
                        outputs=[settings_state]
                    )

    # 4. Wire Up Conditional Visibility (Generic Handler)
    visibility_deps = {}
    feature_config_list = config.get('features', [])
    for feat_def in feature_config_list:
        if isinstance(feat_def, dict) and 'visible_if' in feat_def:
            target_name = feat_def['name']
            parts = parse_condition(feat_def['visible_if'])
            if parts:
                source, op, val = parts
                if source in component_map and target_name in component_map:
                    if source not in visibility_deps:
                        visibility_deps[source] = []
                    visibility_deps[source].append({
                        'target': target_name, 'op': op, 'val': val
                    })

    for source_name, dependants in visibility_deps.items():
        source_comp = component_map[source_name]
        targets = [component_map[d['target']] for d in dependants]
        
        def update_visibility_generic(source_val, deps=dependants):
            """
            Compute visibility updates for a set of dependent UI targets based on a source value.
            
            Parameters:
                source_val: The current value of the source feature used for comparisons.
                deps (list[dict]): A list of dependency descriptors. Each descriptor must contain:
                    - 'op' (str): Comparison operator; supported values are '==' and '!='.
                    - 'val' (str): Target value to compare against the string form of source_val.
            
            Returns:
                list: A list of gr.update objects where each update sets `visible` to `true` if the comparison
                between `str(source_val)` and the descriptor's 'val' satisfies the descriptor's 'op', `false` otherwise.
            """
            updates = []
            for dep in deps:
                op = dep['op']
                target_val = dep['val']
                visible = False
                if op == '==': visible = (str(source_val) == target_val)
                elif op == '!=': visible = (str(source_val) != target_val)
                updates.append(gr.update(visible=visible))
            return updates

        source_comp.change(
            fn=update_visibility_generic,
            inputs=[source_comp],
            outputs=targets
        )

    # 5. Wire Up Conditional Visibility (Prompt Source)
    if "prompt_source" in component_map:
        ps_comp = component_map["prompt_source"]
        target_features = [
            "prompt_presets", "task_prompt",
            "prompt_prefix", "prompt_file_extension", "prompt_suffix"
        ]
        
        target_map = {}
        for name in target_features:
            if name in component_map:
                target_map[name] = component_map[name]
                
        if target_map:
            def update_visibility_dynamic(source_val):
                """
                Compute visibility updates for a set of target components based on the selected prompt source.
                
                Parameters:
                    source_val (str): Selected prompt source; expected values: "Prompt Presets", "From File", or "From Metadata".
                
                Returns:
                    list: A list of gr.update objects, in the same order as target_map.keys(), where each update sets `visible` to `True` or `False` according to the selected source.
                """
                updates = []
                for name in target_map.keys():
                    visible = False
                    if source_val == "Prompt Presets":
                        if name in ["prompt_presets", "task_prompt"]: visible = True
                    elif source_val == "From File":
                        if name in ["prompt_prefix", "prompt_file_extension", "prompt_suffix"]: visible = True
                    elif source_val == "From Metadata":
                        if name in ["prompt_prefix", "prompt_suffix"]: visible = True
                    updates.append(gr.update(visible=visible))
                return updates

            ps_comp.change(
                fn=update_visibility_dynamic,
                inputs=[ps_comp],
                outputs=list(target_map.values())
            )

    # 6. Wire Up Prompt Presets -> Task Prompt
    if "prompt_presets" in component_map and "task_prompt" in component_map:
        it_comp = component_map["prompt_presets"]
        tp_comp = component_map["task_prompt"]
        
        current_version = current_values.get('model_version')
        presets = app.config_mgr.get_version_prompt_presets(model_id, current_version)
        
        def handle_preset_change(template_name, current_state):
            """
            Update the settings state with the selected prompt preset and produce the value to apply to the task prompt input.
            
            Parameters:
                template_name (str): Name of the selected prompt preset.
                current_state (dict | None): Current settings state; if None, a new dict will be created.
            
            Returns:
                tuple:
                    - `gr.update()` if no preset is selected or the preset name is not found, otherwise the preset prompt string to apply to the task prompt.
                    - The updated settings state dictionary with `prompt_presets` set to `template_name` and `task_prompt` set to the preset prompt when applicable.
            """
            if current_state is None:
                current_state = {}
            
            current_state['prompt_presets'] = template_name
            if not template_name or template_name not in presets:
                return gr.update(), current_state
            
            new_prompt = presets[template_name]
            current_state['task_prompt'] = new_prompt
            return new_prompt, current_state

        it_comp.change(
            fn=handle_preset_change,
            inputs=[it_comp, settings_state],
            outputs=[tp_comp, settings_state]
        )