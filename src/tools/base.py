"""
Base Tool Module

Provides foundation for all tools in the application.
Any .py file in src/tools/ that implements BaseTool will be auto-discovered
and receive a tab in the Tools section of the GUI.

Each tool is SELF-CONTAINED with:
- Tool metadata (name, description, icon)  
- apply_to_dataset() method for processing
- create_gui() method that builds UI AND wires events
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any
import gradio as gr


@dataclass
class ToolConfig:
    """
    Configuration for a tool.
    
    Attributes:
        name: Internal identifier (e.g., "metadata")
        display_name: Tab label shown in GUI (e.g., "Metadata")
        description: Markdown description shown at top of tool tab
        icon: Emoji icon for the tool
    """
    name: str
    display_name: str
    description: str
    icon: str = ""


class BaseTool(ABC):
    """
    Abstract base class for all tools.
    
    To create a new tool:
    1. Create a .py file in src/tools/
    2. Define a class that inherits from BaseTool
    3. Implement config, apply_to_dataset(), and create_gui()
    4. The tool will be auto-discovered and added to the GUI
    
    Example:
        class CropTool(BaseTool):
            @property
            def config(self) -> ToolConfig:
                return ToolConfig(
                    name="crop",
                    display_name="Crop",
                    description="### ✂️ Crop Images",
                    icon="✂️"
                )
            
            def apply_to_dataset(self, dataset, **kwargs) -> str:
                # Process images
                return "Cropped X images"
            
            def create_gui(self, app) -> tuple:
                # Create UI components and return (run_button, inputs_list)
                run_btn = gr.Button("Crop", variant="primary")
                return (run_btn, [])
    """
    
    @property
    @abstractmethod
    def config(self) -> ToolConfig:
        """Return tool configuration. Must be implemented by subclasses."""
        pass
    
    @property
    def name(self) -> str:
        """Return tool name."""
        return self.config.name
    
    @property
    def display_name(self) -> str:
        """Return tool display name for UI."""
        return self.config.display_name
    
    @abstractmethod
    def apply_to_dataset(self, dataset, **kwargs) -> str:
        """
        Apply the tool to a dataset.
        
        Args:
            dataset: Dataset object containing images
            **kwargs: Tool-specific arguments from GUI
            
        Returns:
            str: Status message to display to user
        """
        pass
    
    @abstractmethod
    def create_gui(self, app) -> tuple:
        """
        Create Gradio UI components for this tool.
        
        This method is called inside the tool's tab context.
        It must create all UI elements and return them for later event wiring.
        
        Args:
            app: CaptioningApp instance for accessing dataset/config
            
        Returns:
            tuple: (run_button, list_of_input_components)
        """
        pass
    
    def wire_events(self, app, run_button, inputs: list, gallery_output: gr.Gallery,
                    limit_count=None) -> None:
        """
        Wire up events after all components are created.
        
        Called after the gallery component exists.
        Default implementation wires run_button to apply_to_dataset.
        Applies limit_count to dataset if provided.
        
        Args:
            app: CaptioningApp instance
            run_button: The button that triggers the tool
            inputs: List of input components
            gallery_output: The gallery to update after tool runs
            limit_count: Optional limit_count component from Input Source
        """
        import copy
        tool_name = self.config.display_name
        
        # ANSI color codes for console output
        CYAN = "\033[96m"
        GREEN = "\033[92m"
        YELLOW = "\033[93m"
        RESET = "\033[0m"
        BOLD = "\033[1m"
        
        # Build inputs list - append limit_count if provided
        all_inputs = inputs.copy() if inputs else []
        if limit_count is not None:
            all_inputs.append(limit_count)
        
        def run_handler(*args):
            if not app.dataset or not app.dataset.images:
                gr.Warning("No images loaded. Please load a folder in Input Source first.")
                return app._get_gallery_data()
            
            # Extract limit value from end of args if limit_count was added
            limit_val = None
            tool_args = args
            if limit_count is not None and len(args) > len(inputs):
                limit_val = args[-1]  # Last arg is limit_count value
                tool_args = args[:-1]  # Everything except last is tool inputs
            
            run_dataset = app.dataset
            total_count = len(app.dataset.images)
            
            # Apply limit if set
            if limit_val:
                try:
                    limit = int(limit_val)
                    if limit > 0 and total_count > limit:
                        run_dataset = copy.copy(app.dataset)
                        run_dataset.images = app.dataset.images[:limit]
                        print(f"{YELLOW}Limiting to first {limit} images (from {total_count} loaded).{RESET}")
                except (ValueError, TypeError):
                    pass
            
            image_count = len(run_dataset.images)
            print(f"")
            print(f"{BOLD}{CYAN}--- Running {tool_name} Tool on {image_count} images ---{RESET}")
            
            result = self.apply_to_dataset(run_dataset, *tool_args)
            
            print(f"{GREEN}Result: {result}{RESET}")
            print(f"{BOLD}{CYAN}--- {tool_name} Tool Complete ---{RESET}")
            print(f"")
            
            gr.Info(result)
            return app._get_gallery_data()
        
        run_button.click(
            run_handler,
            inputs=all_inputs,
            outputs=[gallery_output]
        )
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.config.name})"
