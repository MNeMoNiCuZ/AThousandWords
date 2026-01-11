"""
Dynamic Component Generator for Gradio GUI

Utilities for creating Gradio components dynamically from feature configurations.

IMPORTANT: Feature layouts are now defined in each model's YAML config using `feature_rows`.


See: src/config/feature_layouts.yaml for shared presets.
"""

import gradio as gr
from typing import Any, Dict


def create_component_from_feature_config(feature_config: Dict[str, Any]):
    """
    Create a Gradio component based on feature configuration from get_gui_config().
    
    Args:
        feature_config: Dictionary from feature.get_gui_config() containing:
            - type: Component type (slider, dropdown, checkbox, textbox, number)
            - label: Display label
            - info: Tooltip/help text
            - value: Default value
            - min: Minimum value (for sliders/numbers)
            - max: Maximum value (for sliders/numbers)
            - step: Step size (for sliders)
            - choices: Options list (for dropdowns)
    
    Returns:
        Gradio component configured according to feature specs
    """
    gui_type = feature_config.get('type', 'textbox')
    
    # Base arguments for all components
    base_args = {
        'label': feature_config.get('label', ''),
        'info': feature_config.get('info', ''),
        'value': feature_config.get('value'),
        'visible': feature_config.get('visible', True),  # Respect config or default to True
        'interactive': feature_config.get('interactive', True)
    }
    
    # SAFETY: Ensure no 'choices' leak into base_args (though explicit dict above prevents it, this is for sanity)
    if 'choices' in base_args:
        del base_args['choices']
    
    # Create component based on type
    if gui_type == 'slider':
        return gr.Slider(
            minimum=feature_config.get('min', 0),
            maximum=feature_config.get('max', 1),
            step=feature_config.get('step', 0.1),
            **base_args
        )
    
    elif gui_type == 'dropdown':
        choices = feature_config.get('choices', [])
        value = base_args.get('value')
        
        # Auto-select first choice if no value provided
        if (value is None or value == '') and choices:
            value = choices[0]
        
        return gr.Dropdown(
            choices=choices,
            value=value,
            allow_custom_value=feature_config.get('allow_custom_value', True),
            label=base_args['label'],
            info=base_args['info'],
            visible=base_args['visible'],
            interactive=base_args['interactive']
        )
    
    elif gui_type == 'checkbox':
        return gr.Checkbox(**base_args)
    
    elif gui_type == 'number':
        return gr.Number(**base_args)
    
    elif gui_type == 'textbox':
        lines = feature_config.get('lines', 1)
        max_lines = feature_config.get('max_lines', 20)
        return gr.Textbox(lines=lines, max_lines=max_lines, **base_args)
    
    elif gui_type == 'code':
        # Remove info if present as Code might not support it in all versions
        if 'info' in base_args:
             del base_args['info']
        
        return gr.Code(
            language=feature_config.get('language', None),
            lines=feature_config.get('lines', 1),
            **base_args
        )
    
    else:
        # Fallback to textbox for unknown types
        return gr.Textbox(**base_args)



