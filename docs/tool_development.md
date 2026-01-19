# Tool Development Guide

This guide explains how to create new tools for the A Thousand Words application. Tools appear in the **Tools** tab and provide batch operations on loaded datasets.

---

## Architecture Overview

### Auto-Discovery System

The tool system uses **automatic discovery**. Any `.py` file in `src/tools/` that contains a class inheriting from `BaseTool` is automatically:
1. Imported at application startup
2. Instantiated and registered
3. Given its own tab in the Tools section

**No registration required** - just create the file and it appears.

### Key Files

| File | Purpose |
|------|---------|
| `src/tools/base.py` | `BaseTool` abstract class and `ToolConfig` dataclass |
| `src/tools/__init__.py` | Auto-discovery and registry logic |
| `docs/template_tool.py` | **REFERENCE IMPLEMENTATION** - Copy this for new tools |
| `src/gui/tabs/tools.py` | Creates Tool tabs and wires events |

> [!TIP]
> **Start with the template!** Copy `src/tools/template_tool.py` as your starting point. It demonstrates all core patterns with detailed comments.

---

## Creating a New Tool

### Step 1: Create the Tool File

Create a new file: `src/tools/your_tool.py`

```python
"""
Your Tool Name

Brief description of what the tool does.
"""

import gradio as gr
from pathlib import Path

from .base import BaseTool, ToolConfig


class YourTool(BaseTool):
    """Tool for doing something useful."""
    
    @property
    def config(self) -> ToolConfig:
        return ToolConfig(
            name="your_tool",           # Internal identifier
            display_name="Your Tool",   # Tab label in GUI
            description="### Your Tool\nMarkdown description shown at top of tool tab.",
            icon=""                     # Leave empty (icons not currently used)
        )
    
    def apply_to_dataset(self, dataset, param1: str, param2: int) -> str:
        """
        Process the dataset.
        
        Args:
            dataset: Dataset object containing images
            param1, param2: Values from GUI inputs (in order)
            
        Returns:
            str: Status message shown to user
        """
        # Your processing logic here
        count = 0
        for img_obj in dataset.images:
            # Process each image...
            count += 1
        
        return f"Processed {count} images"
    
    def create_gui(self, app) -> tuple:
        """
        Create Gradio UI components.
        
        Args:
            app: CaptioningApp instance (access dataset, config, etc.)
            
        Returns:
            tuple: (run_button, list_of_input_components)
        """
        gr.Markdown(self.config.description)
        
        with gr.Row():
            param1 = gr.Textbox(label="Parameter 1", value="default")
            param2 = gr.Number(label="Parameter 2", value=10, precision=0)
        
        run_btn = gr.Button("Run Tool", variant="primary")
        
        # Return button and inputs IN ORDER matching apply_to_dataset params
        return (run_btn, [param1, param2])
```

### Step 2: That's It

The tool will automatically appear in the GUI on next launch. No registration needed.

---

## Accessing the Dataset

### Dataset Object

The `dataset` parameter in `apply_to_dataset()` is a `Dataset` object from `src/core/dataset.py`:

```python
# Dataset properties
dataset.images      # List[MediaObject] - all loaded media
len(dataset)        # Number of images
dataset.get_paths() # List[Path] - all file paths

# Iterate
for img_obj in dataset.images:
    ...
```

### MediaObject Properties

Each item in `dataset.images` is a `MediaObject`:

```python
img_obj.path           # Path - absolute file path
img_obj.caption        # str - current caption text
img_obj.original_size  # tuple - (width, height)
img_obj.metadata       # dict - extracted metadata
img_obj.error          # Optional[str] - error message if any
img_obj.media_type     # str - "image" or "video"
img_obj._modified      # bool - caption changed flag

# Methods
img_obj.is_video()              # bool - True if video file
img_obj.load_image()            # PIL.Image - load the image
img_obj.update_caption(text)    # Update caption (sets _modified)
img_obj.save_caption(ext, dir)  # Save caption to file
```

### Loading Images

```python
from PIL import Image

for img_obj in dataset.images:
    # Load as PIL Image
    pil_image = img_obj.load_image()
    if pil_image is None:
        continue  # Failed to load
    
    # Or open directly from path
    pil_image = Image.open(img_obj.path)
```

---

## GUI Components

### Common Patterns

**Row Layout (4 items per row recommended):**
```python
with gr.Row():
    input1 = gr.Textbox(label="Input 1")
    input2 = gr.Number(label="Input 2", value=0, precision=0)
    input3 = gr.Checkbox(label="Option", value=True)
    input4 = gr.Dropdown(["A", "B", "C"], value="A", label="Choice")
```

**Sections with Markdown:**
```python
gr.Markdown("**Section Title**")
with gr.Row():
    ...
```

**Sliders:**
```python
slider = gr.Slider(
    minimum=0, maximum=100, value=50, step=1,
    label="Value", info="Description shown below"
)
```

### Accordion Grouping

**ALWAYS wrap settings in an accordion** to reduce visual clutter:

```python
gr.Markdown(self.config.description)

with gr.Accordion("Settings", open=True):
    gr.Markdown("**Output Settings**")
    with gr.Row():
        output_dir = gr.Textbox(label="Output Directory", info="...")
        prefix = gr.Textbox(label="Prefix", info="...")
    
    gr.Markdown("**Processing Settings**")
    with gr.Row():
        some_option = gr.Checkbox(label="Option", info="...")
        some_slider = gr.Slider(0, 1, value=0.5, info="...")

run_btn = gr.Button("Run Tool", variant="primary")  # Button OUTSIDE accordion
```

**Rules:**
- Wrap all settings in ONE accordion
- Use `gr.Markdown("**Section Name**")` for sub-sections inside the accordion
- Keep the run button OUTSIDE the accordion
- Use `open=True` so settings are visible by default

### Component Types

| Type | Gradio Component | Use Case |
|------|------------------|----------|
| Text | `gr.Textbox` | Paths, strings |
| Number | `gr.Number` | Integers, counts |
| Slider | `gr.Slider` | Ranges, probabilities |
| Checkbox | `gr.Checkbox` | Boolean options |
| Dropdown | `gr.Dropdown` | Fixed choices |
| File | `gr.File` | File uploads |

---

## Notifications & Popups

### User Feedback

```python
import gradio as gr

# Info popup (blue, temporary)
gr.Info("Operation completed successfully.")

# Warning popup (yellow, stays visible)
gr.Warning("Some files were skipped.")

# Error (red, raises exception)
raise gr.Error("Critical failure occurred.")
```

### Console Logging

**Use colorama for all console colors.** See `GEMINI.md` for the full color standard.

```python
from colorama import Fore, Style

# Tool header
print(f"\n{Fore.CYAN}--- Running Tool ---{Style.RESET_ALL}")

# Settings summary
print(f"{Fore.CYAN}Settings:{Style.RESET_ALL}")
print(f"  {Fore.YELLOW}Option:{Style.RESET_ALL} {value}")

# Progress
for item in items:
    if success:
        print(f"  {Fore.GREEN}✓{Style.RESET_ALL} {item.name}")
    else:
        print(f"  {Fore.RED}✗{Style.RESET_ALL} {item.name}")

# Summary
print(f"\n{Fore.CYAN}Summary:{Style.RESET_ALL}")
print(f"  {Fore.GREEN}Success: {count}{Style.RESET_ALL}")
print(f"{Fore.CYAN}--- Tool Complete ---{Style.RESET_ALL}\n")
```

**Color meanings:**
- `Fore.CYAN` - Headers, section titles
- `Fore.GREEN` - Success, progress
- `Fore.YELLOW` - Labels, warnings
- `Fore.RED` - Errors

---

## Custom Event Wiring

### Default Behavior

The base `wire_events()` method automatically:
1. Handles the limit_count from Input Source
2. Calls `apply_to_dataset()` with GUI inputs
3. Updates the gallery after completion
4. Shows the result message

### Custom Events

Override `wire_events()` for complex tools (like Bucketing):

```python
def wire_events(self, app, run_button, inputs: list, gallery_output: gr.Gallery,
                limit_count=None) -> None:
    """Custom event wiring."""
    
    # Extract specific inputs
    my_input = inputs[0]
    other_btn = inputs[5]  # If you added extra buttons to inputs
    
    def custom_handler(*args):
        # Your logic
        result = self.apply_to_dataset(app.dataset, *args)
        gr.Info(result)
        return app._get_gallery_data()
    
    run_button.click(
        custom_handler,
        inputs=[my_input],
        outputs=[gallery_output]
    )
```

### Reusable Button State Functions

**USE THESE for consistent button behavior.** Import from `src/gui/inference.py`:

```python
from src.gui.inference import tool_start_processing, tool_finish_processing

def wire_events(self, app, run_button, inputs, gallery_output, limit_count=None):
    download_btn = self._download_btn
    download_btn_group = self._download_btn_group
    
    def run_handler(*args):
        # Your processing logic...
        result, generated_files = self.apply_to_dataset(app.dataset, *args)
        gr.Info(result)
        
        # Use reusable finish function - handles button state and download
        return tool_finish_processing("Run My Tool", generated_files)
    
    # Use reusable start function - shows "Processing..." and spinner
    run_button.click(
        tool_start_processing,
        inputs=[],
        outputs=[run_button, download_btn_group, download_btn]
    ).then(
        run_handler,
        inputs=inputs,
        outputs=[run_button, download_btn_group, download_btn],
        show_progress="hidden"
    )
```

**Functions:**
- `tool_start_processing()` - Returns button updates for "Processing..." state
- `tool_finish_processing(button_text, generated_files)` - Returns button updates with optional download zip

---

## Working with Files

### Path Handling

**Always use forward slashes** (works on all platforms):

```python
from pathlib import Path

# Good
output_path = Path(output_dir) / "subfolder" / filename

# Also good
output_path = Path("data/output/file.txt")
```

### Creating Output Directories

```python
out_path = Path(output_dir)
out_path.mkdir(parents=True, exist_ok=True)
```

### Saving Files

```python
# Save PIL Image
img.save(output_path, format="JPEG", quality=95)

# Save text
Path(output_path).write_text(content, encoding="utf-8")
```

---

## Accessing App State

The `app` parameter in `create_gui()` provides access to:

```python
# Dataset
app.dataset              # Current Dataset object
app.is_drag_and_drop     # True if files were dragged (not folder path)
app.current_input_path   # Path string of loaded source

# Configuration
app.config_mgr           # ConfigManager for settings
app.config_mgr.get_global_settings()  # Global user settings

# File utilities
app.analyze_input_paths()  # Returns (common_root, mixed_sources, collisions)
app.create_zip(file_list)  # Create zip, returns path

# Gallery
app._get_gallery_data()  # Get current gallery display data
```

---

## Handling Server Mode & Downloads

When the app runs in **Server Mode** (`--server`), users cannot access the server's filesystem directly. Tools must:
1. Hide unrestricted directory inputs.
2. Generate outputs to a temporary directory.
3. Zip the results and provide a download button.

### 1. Hide Directory Inputs

Pass `is_server_mode` to `create_gui`:

```python
def create_gui(self, app, is_server_mode=False) -> tuple:
    self._is_server_mode = is_server_mode
    
    # ...
    
    output_dir = gr.Textbox(
        label="Output Directory", 
        visible=not is_server_mode  # Hide in server mode
    )
    
    # Add Download Button (initially hidden)
    with gr.Column(visible=False, scale=0, min_width=80, elem_classes="download-btn-wrapper") as download_btn_group:
        download_btn = gr.DownloadButton(
            label="", 
            icon=str(Path(__file__).parent.parent / "core" / "download_white.svg"),
            visible=True, variant="primary", scale=0, elem_classes="download-btn"
        )
        
    self._download_btn = download_btn
    self._download_btn_group = download_btn_group
```

### 2. Wire Events for Server Support

Use the reusable `tool_start_processing` and `tool_finish_processing` helpers.

```python
def wire_events(self, app, run_button, inputs, gallery_output, limit_count=None):
    from src.gui.inference import tool_start_processing, tool_finish_processing
    import tempfile
    import os
    
    def run_handler(*args):
        args_list = list(args)
        
        # Handle Server Mode: Redirect output to temp dir
        temp_dir_obj = None
        if self._is_server_mode:
            temp_dir_obj = tempfile.TemporaryDirectory(dir=os.environ.get("GRADIO_TEMP_DIR"), prefix="tool_")
            args_list[2] = temp_dir_obj.name # Assuming args[2] is output_dir
        
        # Run Tool
        msg, generated_files = self.apply_to_dataset(app.dataset, *args_list)
        gr.Info(msg)
        
        # Finish & Update UI
        # zip_prefix creates "my_tool_YYYYMMDD_HHMMSS.zip"
        ui_updates = tool_finish_processing(
            "Run Tool", 
            generated_files, 
            zip_prefix="my_tool"
        )
        
        if temp_dir_obj:
            temp_dir_obj.cleanup()
            
        return ui_updates

    run_button.click(
        tool_start_processing, 
        inputs=[], 
        outputs=[run_button, self._download_btn_group, self._download_btn]
    ).then(
        run_handler,
        inputs=inputs, 
        outputs=[run_button, self._download_btn_group, self._download_btn]
    )
```

See `src/tools/template_tool.py` for a complete, copy-pasteable implementation.

## Examples

### Simple Tool: Rename Files

```python
class RenameTool(BaseTool):
    @property
    def config(self) -> ToolConfig:
        return ToolConfig(
            name="rename",
            display_name="Rename",
            description="### Rename Files\nAdd prefix/suffix to filenames."
        )
    
    def apply_to_dataset(self, dataset, prefix: str, suffix: str) -> str:
        import shutil
        count = 0
        for img_obj in dataset.images:
            old_path = img_obj.path
            new_name = f"{prefix}{old_path.stem}{suffix}{old_path.suffix}"
            new_path = old_path.parent / new_name
            shutil.move(old_path, new_path)
            img_obj.path = new_path
            count += 1
        return f"Renamed {count} files"
    
    def create_gui(self, app) -> tuple:
        gr.Markdown(self.config.description)
        with gr.Row():
            prefix = gr.Textbox(label="Prefix", placeholder="Added before name")
            suffix = gr.Textbox(label="Suffix", placeholder="Added after name")
        run_btn = gr.Button("Rename", variant="primary")
        return (run_btn, [prefix, suffix])
```

### Complex Tool: See `bucketing.py`

For advanced examples with custom event wiring, multiple buttons, and HTML output, see `src/tools/bucketing.py`.

---

## Checklist

Before submitting your tool:

- [ ] Inherits from `BaseTool`
- [ ] Implements `config` property with `ToolConfig`
- [ ] Implements `apply_to_dataset()` returning status string
- [ ] Implements `create_gui()` returning `(button, inputs)`
- [ ] Inputs list matches `apply_to_dataset()` parameter order
- [ ] Uses forward slashes in all file paths
- [ ] Includes console logging with ANSI colors
- [ ] Uses `gr.Info()` / `gr.Warning()` for user feedback
- [ ] Contains docstrings explaining functionality
