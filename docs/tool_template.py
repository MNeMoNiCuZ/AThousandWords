"""
Template Tool
Use this file as a starting point for creating new tools.
"""

import gradio as gr
from pathlib import Path
from colorama import Fore, Style

from src.tools.base import BaseTool, ToolConfig

class TemplateTool(BaseTool):
    """
    A template tool demonstrating best practices:
    1. UI Layout with Accordion
    2. Settings saving/loading (via BaseTool logic if following naming conventions)
    3. Server Mode compatibility:
       - Checks `is_server_mode` flag
       - Hides raw output paths to prevent unrestricted write access
       - Uses `GRADIO_TEMP_DIR` for temporary storage
       - Zips outputs for client download via `tool_finish_processing`
    4. Button state management
    5. Console logging with colors
    """
    
    @property
    def config(self) -> ToolConfig:
        return ToolConfig(
            name="template_tool",          # Internal ID (lowercase, snake_case)
            display_name="Template Tool",  # UI Tab Label
            description="### Template Tool\nDescription of what this tool does.",
            icon=""                        # Optional icon path
        )
    
    def apply_to_dataset(self, dataset, my_setting: str, count: int, output_dir: str = None) -> tuple:
        """
        Process the dataset.
        
        Args:
            dataset: The app's dataset object.
            my_setting: A string setting from the UI.
            count: A number setting from the UI.
            output_dir: Helper for saving files.
            
        Returns:
            tuple: (Status Message String, List of Generated File Paths)
        """
        import time
        from pathlib import Path
        
        processed_files = []
        
        # Determine output location
        # If output_dir is provided (e.g. from server mode helper), use it.
        # Otherwise use dataset path or default.
        save_dir = Path(output_dir) if output_dir else Path(dataset.images[0].path).parent / "output"
        save_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n{Fore.CYAN}--- Running Template Tool ---{Style.RESET_ALL}")
        print(f"Setting: {my_setting} | Count: {count}")
        print(f"Output: {save_dir}")
        
        # Simulate processing
        for i in range(min(len(dataset.images), int(count))):
            # Example: Create a dummy text file for each image
            img = dataset.images[i]
            out_file = save_dir / f"{img.path.stem}_template.txt"
            out_file.write_text(f"Processed with setting: {my_setting}", encoding="utf-8")
            processed_files.append(str(out_file))
            print(f"  {Fore.GREEN}✓{Style.RESET_ALL} Generated {out_file.name}")
            
        print(f"{Fore.CYAN}--- Tool Complete ---{Style.RESET_ALL}\n")
        
        return f"Successfully processed {len(processed_files)} items.", processed_files
    
    def create_gui(self, app, is_server_mode=False) -> tuple:
        """
        Create the Gradio UI.
        
        Args:
            app: Reference to the main CaptioningApp.
            is_server_mode (bool): Whether the app is running in server mode.
                                   If True, hide unrestricted paths and enable zip downloads.
        """
        self._is_server_mode = is_server_mode
        self._app = app
        
        gr.Markdown(self.config.description)
        
        with gr.Accordion("Settings", open=True):
            with gr.Row():
                my_setting = gr.Textbox(label="My Setting", value="default")
                count = gr.Number(label="Count", value=5, precision=0)
            
            with gr.Row():
                # Server Mode: Hide raw output directory inputs to prevent arbitrary writes
                output_dir = gr.Textbox(
                    label="Output Directory", 
                    placeholder="Optional", 
                    visible=not is_server_mode
                )
        
        with gr.Row():
            # Save Settings Button (handled by BaseTool logic automatically if added)
            save_btn = gr.Button("Save Settings", variant="secondary", scale=0)
            
            # Run Button
            run_btn = gr.Button("Run Template Tool", variant="primary", scale=1)
            
            # Download Button (Initially hidden, shown after processing in server mode)
            with gr.Column(visible=False, scale=0, min_width=80, elem_classes="download-btn-wrapper") as download_btn_group:
                download_btn = gr.DownloadButton(
                    label="", 
                    icon=str(Path(__file__).parent.parent / "core" / "download_white.svg"),
                    visible=True, variant="primary", scale=0, elem_classes="download-btn"
                )
        
        # Store components for event wiring
        self._save_btn = save_btn
        self._download_btn = download_btn
        self._download_btn_group = download_btn_group
        
        # Return Run Button and List of Inputs
        inputs = [my_setting, count, output_dir]
        return (run_btn, inputs)
    
    def wire_events(self, app, run_button, inputs: list, gallery_output: gr.Gallery, limit_count=None) -> None:
        """
        Connect UI events to logic.
        """
        # Import helpers for button state management
        from src.gui.inference import tool_start_processing, tool_finish_processing
        import tempfile
        import os
        
        download_btn = self._download_btn
        download_btn_group = self._download_btn_group
        save_btn = self._save_btn
        
        # Prepare inputs list (append limit_count if provided)
        all_inputs = inputs.copy() if inputs else []
        if limit_count is not None:
            all_inputs.append(limit_count)
            
        def run_handler(*args):
            # Parse arguments (last might be limit_count value)
            args_list = list(args)
            limit_val = None
            if limit_count is not None and len(args) > len(inputs):
                limit_val = args[-1]
                args_list = args_list[:-1]
                
            # Extract specific args based on inputs order
            # my_setting = args_list[0]
            # count = args_list[1]
            # output_dir = args_list[2]
            
            # 1. Validation
            if not app.dataset or not app.dataset.images:
                gr.Warning("No images loaded.")
                return tool_finish_processing("Run Template Tool", [])
            
            # 2. Server Mode Setup
            temp_dir_obj = None
            if self._is_server_mode:
                # Force output to a temporary directory
                temp_dir_obj = tempfile.TemporaryDirectory(dir=os.environ.get("GRADIO_TEMP_DIR"), prefix="template_")
                args_list[2] = temp_dir_obj.name # Override output_dir
                print(f"{Fore.CYAN}Server Mode: Output redirected to temp dir: {args_list[2]}{Style.RESET_ALL}")
            
            try:
                # 3. Execution
                # Pass args to main logic
                result_msg, generated_files = self.apply_to_dataset(app.dataset, *args_list)
                
                gr.Info(result_msg)
                
                # 4. Finish & Download
                # Pass zip_prefix to create a nice filename like "template_output_2024...zip"
                ui_updates = tool_finish_processing("Run Template Tool", generated_files, zip_prefix="template_output")
                
                # Cleanup logic: tool_finish_processing zips the files immediately.
                # If we used a temp dir, we can verify zip creation and then cleanup.
                if temp_dir_obj:
                    # Note: If generated_files is empty, zip isn't made, so we just cleanup.
                    # If zip is made, it's in a separate temp location, so cleaning this one is safe.
                    temp_dir_obj.cleanup()
                
                return ui_updates
                
            except Exception as e:
                import traceback
                traceback.print_exc()
                if temp_dir_obj:
                    temp_dir_obj.cleanup()
                raise e
        
        # Save Settings Handler (Optional if using BaseTool defaults)
        # See bucket/augment tools for full implementation pattern
        
        # Wire Logic
        # 1. Start: Disable button, show "Processing..."
        # 2. Run: Execute logic, return updates (including Download Button if enabled)
        run_button.click(
            tool_start_processing,
            inputs=[],
            outputs=[run_button, download_btn_group, download_btn]
        ).then(
            run_handler,
            inputs=all_inputs,
            outputs=[run_button, download_btn_group, download_btn],
            show_progress="hidden"
        )
