"""
Tools tab factory.

Creates the Tools tab with auto-discovered tools.
"""

import gradio as gr
from src.tools import get_all_tools


def create_tools_tab(app) -> dict:
    """
    Builds a Tools tab container and instantiates GUIs for all discovered tools.
    
    Parameters:
        app: The application instance passed to each tool's GUI creation function.
    
    Returns:
        dict: Mapping from tool name (str) to a dict with keys:
            - "tool": the tool object
            - "run_btn": the tool's run button component
            - "inputs": the tool's input component(s)
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
    """
    Attach each discovered tool's event handlers to its run button and input components so tool outputs are sent to the provided gallery.
    
    Parameters:
        app: The application instance used by tools (e.g., CaptioningApp).
        tool_components (dict): Mapping from tool name to a dict with keys:
            - "tool": the tool object exposing `wire_events`.
            - "run_btn": the tool's run button component.
            - "inputs": the tool's input components.
        gallery_output (gr.Gallery): The gallery component that receives tool outputs.
    """
    for tool_name, components in tool_components.items():
        tool = components["tool"]
        run_btn = components["run_btn"]
        inputs = components["inputs"]
        tool.wire_events(app, run_btn, inputs, gallery_output)