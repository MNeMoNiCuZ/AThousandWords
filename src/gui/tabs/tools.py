"""
Tools tab factory.

Creates the Tools tab with auto-discovered tools.
"""

import gradio as gr
from src.tools import get_all_tools


def create_tools_tab(app, is_server_mode=False) -> dict:
    """Create Tools tab with auto-discovered tools.
    
    Args:
        app: CaptioningApp instance
        is_server_mode (bool): If True, restrict tool output paths
        
    Returns:
        dict of tool_name -> {tool, run_btn, inputs}
    """
    tool_components = {}
    
    # Use app.sorted_tools to ensure consistent order
    from src.tools import get_tool
    
    with gr.Tabs():
        for tool_name in app.sorted_tools:
            tool = get_tool(tool_name)
            if tool:
                is_visible = tool_name in app.enabled_tools
                with gr.Tab(tool.config.display_name, visible=is_visible) as tool_tab:
                    run_btn, inputs = tool.create_gui(app, is_server_mode=is_server_mode)
                    tool_components[tool_name] = {
                        "tool": tool,
                        "run_btn": run_btn,
                        "inputs": inputs,
                        "tab": tool_tab
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
