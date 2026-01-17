"""
Tools tab factory.

Creates the Tools tab with auto-discovered tools.
"""

import gradio as gr
from src.tools import get_all_tools


def create_tools_tab(app) -> dict:
    """Create Tools tab with auto-discovered tools.
    
    Args:
        app: CaptioningApp instance
        
    Returns:
        dict of tool_name -> {tool, run_btn, inputs}
    """
    tool_components = {}
    
    with gr.Tabs():
        for tool_name, tool in get_all_tools().items():
            with gr.Tab(tool.config.display_name):
                run_btn, inputs = tool.create_gui(app)
                tool_components[tool_name] = {
                    "tool": tool,
                    "run_btn": run_btn,
                    "inputs": inputs
                }
    
    return tool_components


def wire_tool_events(app, tool_components: dict, gallery_output: gr.Gallery):
    """Wire tool events after gallery is created.
    
    Args:
        app: CaptioningApp instance
        tool_components: dict from create_tools_tab
        gallery_output: Gallery component for outputs
    """
    for tool_name, components in tool_components.items():
        tool = components["tool"]
        run_btn = components["run_btn"]
        inputs = components["inputs"]
        tool.wire_events(app, run_btn, inputs, gallery_output)
