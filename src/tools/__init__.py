"""
Tool Registry with Auto-Discovery

Scans src/tools/ for .py files that implement BaseTool.
Any valid tool is automatically registered and available to the GUI.

Usage:
    from src.tools import get_all_tools, get_tool
    
    # Get all discovered tools
    tools = get_all_tools()
    for name, tool in tools.items():
        print(f"{tool.display_name}: {tool.config.description}")
    
    # Get a specific tool
    metadata_tool = get_tool("metadata")
"""

import os
import importlib
import inspect
import logging
from pathlib import Path

from .base import BaseTool, ToolConfig

logger = logging.getLogger(__name__)

# Files to exclude from auto-discovery
# These are utility modules, not tool implementations
_EXCLUDED_FILES = {
    '__init__.py',
    'base.py',
}


def _discover_tools() -> dict:
    """
    Discover and instantiate tool classes under the src.tools package.
    
    Searches .py files in this module's directory (skipping files listed in _EXCLUDED_FILES), imports each module, and for each class defined in that module that subclasses BaseTool (excluding BaseTool itself) attempts to instantiate it and register it by the instance's name. Import and instantiation failures are logged but do not stop discovery.
    
    Returns:
        dict: Mapping of tool name to the instantiated tool object (e.g., {tool_name: tool_instance}).
    """
    tools = {}
    tools_dir = Path(__file__).parent
    
    for py_file in tools_dir.glob("*.py"):
        if py_file.name in _EXCLUDED_FILES:
            continue
            
        module_name = py_file.stem
        
        try:
            # Import the module dynamically
            module = importlib.import_module(f".{module_name}", package="src.tools")
            
            # Find classes that inherit from BaseTool
            for name, obj in inspect.getmembers(module, inspect.isclass):
                # Check if it's a subclass of BaseTool (but not BaseTool itself)
                # and that it's defined in this module (not imported)
                if (issubclass(obj, BaseTool) and 
                    obj is not BaseTool and 
                    obj.__module__ == module.__name__):
                    try:
                        instance = obj()
                        tools[instance.name] = instance
                        logger.debug(f"Discovered tool: {instance.display_name}")
                    except Exception as e:
                        logger.error(f"Failed to instantiate tool {name}: {e}")
                        
        except Exception as e:
            logger.warning(f"Failed to load tool module {module_name}: {e}")
    
    return tools


# Auto-discover tools on module import
TOOL_REGISTRY = _discover_tools()


def get_tool(name: str) -> BaseTool:
    """
    Retrieve a registered tool by its name.
    
    Parameters:
        name (str): The registry key for the tool (e.g., "metadata", "resize").
    
    Returns:
        BaseTool or None: The tool instance matching `name`, or `None` if not found.
    """
    return TOOL_REGISTRY.get(name)


def get_all_tools() -> dict:
    """
    Get all registered tools.
    
    Returns:
        dict: Copy of {tool_name: tool_instance}
    """
    return TOOL_REGISTRY.copy()


def refresh_tools() -> None:
    """
    Refresh the module tool registry by rescanning the tools directory for tool implementations.
    
    Rebuilds TOOL_REGISTRY with newly discovered tool instances and logs the number of tools found.
    """
    global TOOL_REGISTRY
    TOOL_REGISTRY = _discover_tools()
    logger.info(f"Refreshed tools registry: {len(TOOL_REGISTRY)} tools found")