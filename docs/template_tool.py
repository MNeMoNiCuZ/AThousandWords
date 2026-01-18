"""
Template Tool (EXAMPLE)

This is a reference implementation demonstrating all core patterns for creating tools.
Copy this file as a starting point for new tools.

Key patterns demonstrated:
1. Settings save/load using existing patterns
2. Console logging with colorama
3. User popups (gr.Info, gr.Warning)
4. Button state management during processing
5. Dataset access without refreshing gallery during processing
6. Output file generation with download button
7. Accordion layout for settings grouping
"""

import time
import gradio as gr
from pathlib import Path

from .base import BaseTool, ToolConfig

# ============================================================================
# COLORAMA IMPORT - Always use this pattern for console colors
# ============================================================================
try:
    from colorama import Fore, Style
except ImportError:
    # Fallback if colorama not installed
    class Fore:
        CYAN = YELLOW = GREEN = RED = MAGENTA = ''
    class Style:
        RESET_ALL = ''


class TemplateTool(BaseTool):
    """
    Template tool demonstrating all core patterns.
    
    This tool is for reference only - it processes images but doesn't
    do anything meaningful. Use it as a starting point for real tools.
    """
    
    # ========================================================================
    # 1. CONFIG - Define tool metadata
    # ========================================================================
    
    @property
    def config(self) -> ToolConfig:
        """
        Return tool configuration.
        
        - name: Internal identifier (used in config keys)
        - display_name: Tab label in GUI  
        - description: Markdown shown at top of tool tab
        - icon: Leave empty (not currently used)
        """
        return ToolConfig(
            name="template_tool",
            display_name="Template Tool",
            description="### Template Tool (Example)\nThis is a reference implementation showing all core patterns. Copy this file to create new tools.",
            icon=""
        )
    
    # ========================================================================
    # 2. DEFAULTS - Define default values for all settings
    # ========================================================================
    
    def _get_defaults(self) -> dict:
        """
        Return default values for all settings.
        
        This is used by:
        - save_settings: To only save values different from defaults
        - get_loaded_values: To provide defaults when no saved value exists
        """
        return {
            "output_dir": "",
            "prefix": "",
            "suffix": "",
            "some_number": 100,
            "some_checkbox": True,
            "some_slider": 0.5,
            "some_dropdown": "option_a",
        }
    
    # ========================================================================
    # 3. LOAD VALUES - Load saved settings from user config
    # ========================================================================
    
    def get_loaded_values(self, app) -> list:
        """
        Load saved settings from user config.
        
        This is called by demo.load() in main.py to restore settings on page refresh.
        Return gr.update() for each input in the same order as create_gui inputs list.
        """
        defaults = self._get_defaults()
        saved = {}
        
        try:
            # Access saved settings from user_config
            tool_settings = app.config_mgr.user_config.get("tool_settings", {})
            saved = tool_settings.get("template_tool", {})
        except Exception:
            pass
        
        # Merge saved over defaults
        values = {**defaults, **saved}
        
        # Return gr.update for each input - ORDER MUST MATCH create_gui inputs list
        return [
            gr.update(value=values["output_dir"]),
            gr.update(value=values["prefix"]),
            gr.update(value=values["suffix"]),
            gr.update(value=values["some_number"]),
            gr.update(value=values["some_checkbox"]),
            gr.update(value=values["some_slider"]),
            gr.update(value=values["some_dropdown"]),
        ]
    
    # ========================================================================
    # 4. APPLY TO DATASET - Main processing logic
    # ========================================================================
    
    def apply_to_dataset(self, dataset, output_dir: str, prefix: str, suffix: str,
                         some_number: int, some_checkbox: bool, some_slider: float,
                         some_dropdown: str, app_input_path: str = None) -> tuple:
        """
        Process the dataset.
        
        Args:
            dataset: Dataset object containing images (access via dataset.images)
            output_dir: User-specified output directory
            prefix, suffix: Filename modifications
            some_number, some_checkbox, some_slider, some_dropdown: Example settings
            app_input_path: Fallback path when output_dir is empty (passed from wire_events)
            
        Returns:
            tuple: (result_message, list_of_generated_file_paths)
        """
        # ====================================================================
        # CONSOLE LOGGING - Use colorama Fore.COLOR and Style.RESET_ALL
        # ====================================================================
        
        if not dataset or not dataset.images:
            gr.Warning("No images loaded.")
            return "Error: No images in dataset.", []
        
        # Determine output directory with fallback logic
        if not output_dir or not output_dir.strip():
            if app_input_path and Path(app_input_path).exists():
                output_dir = app_input_path
                print(f"{Fore.YELLOW}Output dir not set, using input path: {output_dir}{Style.RESET_ALL}")
            else:
                output_dir = "output"
                print(f"{Fore.YELLOW}Output dir not set, using default: {output_dir}{Style.RESET_ALL}")
        
        # Print settings header
        print(f"\n{Fore.CYAN}--- Template Tool Settings ---{Style.RESET_ALL}")
        print(f"  {Fore.YELLOW}Output Dir:{Style.RESET_ALL} {output_dir}")
        print(f"  {Fore.YELLOW}Prefix:{Style.RESET_ALL} '{prefix}' | Suffix: '{suffix}'")
        print(f"  {Fore.YELLOW}Number:{Style.RESET_ALL} {some_number}")
        print(f"  {Fore.YELLOW}Checkbox:{Style.RESET_ALL} {some_checkbox}")
        print(f"  {Fore.YELLOW}Slider:{Style.RESET_ALL} {some_slider}")
        print(f"  {Fore.YELLOW}Dropdown:{Style.RESET_ALL} {some_dropdown}")
        print("")
        
        # Create output directory
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        
        # Process images
        generated_files = []
        processed = 0
        errors = 0
        t0 = time.time()
        
        print(f"{Fore.CYAN}Processing {len(dataset.images)} images...{Style.RESET_ALL}")
        
        for img_obj in dataset.images:
            try:
                # ============================================================
                # ACCESS IMAGE DATA
                # ============================================================
                # img_obj.path - Path to the image file
                # img_obj.caption - Current caption text
                # img_obj.load_image() - Load as PIL Image
                # img_obj.original_size - (width, height) tuple
                # img_obj.update_caption(text) - Update caption
                # img_obj.save_caption(ext, output_dir) - Save caption file
                
                # Example: Create an output file
                src_path = Path(img_obj.path)
                out_name = f"{prefix}{src_path.stem}{suffix}.txt"
                out_file = out_path / out_name
                
                # Write something to the file (this is just an example)
                with open(out_file, 'w', encoding='utf-8') as f:
                    f.write(f"Processed: {src_path.name}\n")
                    f.write(f"Original size: {img_obj.original_size}\n")
                    f.write(f"Caption: {img_obj.caption or '(none)'}\n")
                
                generated_files.append(str(out_file.absolute()))
                processed += 1
                
                # Progress logging - use [OK], [SKIP], [ERR] prefixes
                print(f"  {Fore.GREEN}[OK]{Style.RESET_ALL} {src_path.name} -> {out_name}")
                
            except Exception as e:
                errors += 1
                print(f"  {Fore.RED}[ERR]{Style.RESET_ALL} {src_path.name}: {e}")
        
        # Summary
        elapsed = time.time() - t0
        print("")
        print(f"{Fore.CYAN}Summary:{Style.RESET_ALL}")
        print(f"  {Fore.GREEN}Processed: {processed}{Style.RESET_ALL}")
        print(f"  {Fore.RED}Errors: {errors}{Style.RESET_ALL}")
        print(f"  {Fore.YELLOW}Time:{Style.RESET_ALL} {elapsed:.2f}s")
        print(f"{Fore.CYAN}--- Template Tool Complete ---{Style.RESET_ALL}\n")
        
        return f"Processed {processed} images, {errors} errors", generated_files
    
    # ========================================================================
    # 5. CREATE GUI - Build the Gradio UI
    # ========================================================================
    
    def create_gui(self, app) -> tuple:
        """
        Create the tool UI.
        
        Rules:
        - Show description at top with gr.Markdown(self.config.description)
        - Wrap settings in gr.Accordion("Settings Name", open=True)
        - Use gr.Row() to group related inputs horizontally
        - Add info= parameter to every input for tooltips
        - Return (run_button, inputs_list) where inputs_list order matches apply_to_dataset params
        """
        
        # Description at top
        gr.Markdown(self.config.description)
        
        # ====================================================================
        # ACCORDION - Group all settings in one accordion
        # ====================================================================
        with gr.Accordion("Tool Settings", open=True):
            
            # Output section
            gr.Markdown("**Output Settings**")
            with gr.Row():
                output_dir = gr.Textbox(
                    label="Output Directory",
                    placeholder="output",
                    info="Leave empty to use input path or 'output'"
                )
                prefix = gr.Textbox(
                    label="Prefix",
                    placeholder="processed_",
                    info="Added before filename"
                )
                suffix = gr.Textbox(
                    label="Suffix", 
                    placeholder="_v1",
                    info="Added after filename"
                )
            
            # Example settings section
            gr.Markdown("**Example Settings**")
            with gr.Row():
                some_number = gr.Number(
                    label="Some Number",
                    value=100,
                    precision=0,
                    info="Example number input"
                )
                some_checkbox = gr.Checkbox(
                    label="Some Checkbox",
                    value=True,
                    info="Example checkbox"
                )
                some_slider = gr.Slider(
                    minimum=0,
                    maximum=1,
                    value=0.5,
                    step=0.05,
                    label="Some Slider",
                    info="Example slider"
                )
                some_dropdown = gr.Dropdown(
                    choices=["option_a", "option_b", "option_c"],
                    value="option_a",
                    label="Some Dropdown",
                    info="Example dropdown"
                )
        
        # ====================================================================
        # CONTROL BUTTONS - Run button and optional save/download
        # ====================================================================
        with gr.Row():
            save_btn = gr.Button("Save Settings", variant="secondary", scale=0)
            run_btn = gr.Button("Run Template Tool", variant="primary", scale=1)
            
            # Download button container (hidden until files are generated)
            with gr.Column(visible=False, scale=0, min_width=80) as download_btn_group:
                download_btn = gr.DownloadButton(
                    label="",
                    icon=str(Path(__file__).parent.parent / "core" / "download_white.svg"),
                    visible=True,
                    variant="primary",
                    scale=0
                )
        
        # Store references for wire_events
        self._download_btn = download_btn
        self._download_btn_group = download_btn_group
        self._save_btn = save_btn
        
        # Return inputs IN ORDER matching apply_to_dataset parameters (excluding dataset and app_input_path)
        inputs = [
            output_dir, prefix, suffix,
            some_number, some_checkbox, some_slider, some_dropdown
        ]
        
        return (run_btn, inputs)
    
    # ========================================================================
    # 6. WIRE EVENTS - Custom event handling
    # ========================================================================
    
    def wire_events(self, app, run_button, inputs: list, gallery_output: gr.Gallery,
                    limit_count=None) -> None:
        """
        Wire button events with state management.
        
        Key patterns:
        - Change button text to "Processing..." and disable during operation
        - Don't refresh gallery during processing (only after completion)
        - Show download button with zip file when files are generated
        - Pass app_input_path for output directory fallback
        """
        import copy as copy_module
        from src.gui.constants import filter_user_overrides
        # ====================================================================
        # REUSABLE BUTTON STATE FUNCTIONS - Import from inference.py
        # ====================================================================
        from src.gui.inference import tool_start_processing, tool_finish_processing
        
        tool_name = self.config.display_name
        download_btn = self._download_btn
        download_btn_group = self._download_btn_group
        save_btn = self._save_btn
        
        # Build inputs list with optional limit_count
        all_inputs = inputs.copy() if inputs else []
        if limit_count is not None:
            all_inputs.append(limit_count)
        
        # ====================================================================
        # RUN HANDLER - Main processing with proper state management
        # ====================================================================
        def run_handler(*args):
            """Execute tool with proper state management."""
            # Extract limit if provided
            if limit_count is not None:
                tool_args = list(args[:-1])
                limit_val = args[-1]
            else:
                tool_args = list(args)
                limit_val = None
            
            # Apply limit to dataset
            run_dataset = app.dataset
            total_count = len(app.dataset.images)
            
            if limit_val:
                try:
                    limit = int(limit_val)
                    if limit > 0 and total_count > limit:
                        run_dataset = copy_module.copy(app.dataset)
                        run_dataset.images = app.dataset.images[:limit]
                        print(f"{Fore.YELLOW}Limiting to first {limit} images.{Style.RESET_ALL}")
                except (ValueError, TypeError):
                    pass
            
            image_count = len(run_dataset.images)
            print(f"\n{Fore.CYAN}--- Running {tool_name} on {image_count} images ---{Style.RESET_ALL}")
            
            # Pass app_input_path for output directory fallback
            app_input_path = app.current_input_path if not app.is_drag_and_drop else None
            tool_args.append(app_input_path)
            
            # Execute tool
            result, generated_files = self.apply_to_dataset(run_dataset, *tool_args)
            
            print(f"{Fore.GREEN}Result: {result}{Style.RESET_ALL}")
            
            # Show popup
            gr.Info(result)
            
            # Use REUSABLE finish function - handles download button and button state
            return tool_finish_processing("Run Template Tool", generated_files)
        
        # ====================================================================
        # SAVE SETTINGS - Uses existing save pattern (only save diffs)
        # ====================================================================
        def save_settings(*args):
            """Save tool settings to user config using existing pattern."""
            settings = {
                "output_dir": args[0],
                "prefix": args[1],
                "suffix": args[2],
                "some_number": args[3],
                "some_checkbox": args[4],
                "some_slider": args[5],
                "some_dropdown": args[6],
            }
            
            try:
                # Initialize tool_settings if needed
                if "tool_settings" not in app.config_mgr.user_config:
                    app.config_mgr.user_config["tool_settings"] = {}
                if "template_tool" not in app.config_mgr.user_config["tool_settings"]:
                    app.config_mgr.user_config["tool_settings"]["template_tool"] = {}
                
                # Get defaults for diff comparison
                defaults = self._get_defaults()
                
                # Only save values that differ from defaults
                for key, value in settings.items():
                    default_val = defaults.get(key)
                    if value != default_val:
                        app.config_mgr.user_config["tool_settings"]["template_tool"][key] = value
                    elif key in app.config_mgr.user_config["tool_settings"]["template_tool"]:
                        del app.config_mgr.user_config["tool_settings"]["template_tool"][key]
                
                # Clean up empty dicts
                if not app.config_mgr.user_config["tool_settings"]["template_tool"]:
                    del app.config_mgr.user_config["tool_settings"]["template_tool"]
                if not app.config_mgr.user_config["tool_settings"]:
                    del app.config_mgr.user_config["tool_settings"]
                
                # Save using filter_user_overrides
                filtered = filter_user_overrides(app.config_mgr.user_config)
                app.config_mgr._save_yaml(app.config_mgr.user_config_path, filtered)
                
                gr.Info("Template Tool settings saved!")
            except Exception as e:
                gr.Warning(f"Failed to save settings: {e}")
        
        # ====================================================================
        # WIRE BUTTON EVENTS
        # ====================================================================
        
        # Save button
        save_btn.click(
            save_settings,
            inputs=inputs,
            outputs=[]
        )
        
        # Run button with state management
        # Uses REUSABLE start function for consistent button behavior
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
