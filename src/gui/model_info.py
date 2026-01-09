"""
Model Information Tab

Displays comprehensive model statistics, VRAM requirements,
batch recommendations, and feature support grids.
"""

import gradio as gr
from typing import Dict, Any, List, Tuple


def get_vram_stats(vram_table: Any) -> Tuple[int, int]:
    """
    Extract min and max VRAM from a vram_table.
    
    Args:
        vram_table: Either a flat dict {vram: batch} or nested {version: {vram: batch}}
        
    Returns:
        Tuple of (min_vram, max_vram) in GB, or (0, 0) if no data
    """
    if not vram_table:
        return (0, 0)
    
    all_vram_values = []
    
    # Check if nested (version-specific) or flat
    first_key = next(iter(vram_table), None)
    if first_key is None:
        return (0, 0)
    
    first_value = vram_table.get(first_key)
    
    if isinstance(first_value, dict):
        # Nested structure: {version: {vram: batch}}
        for version_data in vram_table.values():
            if isinstance(version_data, dict):
                for key in version_data.keys():
                    if isinstance(key, int):
                        all_vram_values.append(key)
    else:
        # Flat structure: {vram: batch}
        for key in vram_table.keys():
            if isinstance(key, int):
                all_vram_values.append(key)
    
    if not all_vram_values:
        return (0, 0)
    
    return (min(all_vram_values), max(all_vram_values))


def get_batch_range(vram_table: Any) -> Tuple[int, int]:
    """
    Extract min and max batch sizes from a vram_table.
    
    Args:
        vram_table: Either a flat dict {vram: batch} or nested {version: {vram: batch}}
        
    Returns:
        Tuple of (min_batch, max_batch), or (0, 0) if no data
    """
    if not vram_table:
        return (0, 0)
    
    all_batch_values = []
    
    # Check if nested or flat
    first_key = next(iter(vram_table), None)
    if first_key is None:
        return (0, 0)
    
    first_value = vram_table.get(first_key)
    
    if isinstance(first_value, dict):
        # Nested structure
        for version_data in vram_table.values():
            if isinstance(version_data, dict):
                for batch in version_data.values():
                    if isinstance(batch, int):
                        all_batch_values.append(batch)
    else:
        # Flat structure
        for batch in vram_table.values():
            if isinstance(batch, int):
                all_batch_values.append(batch)
    
    if not all_batch_values:
        return (0, 0)
    
    return (min(all_batch_values), max(all_batch_values))


def build_unified_model_table_markdown(config_mgr) -> str:
    """
    Build a single unified markdown table with VRAM stats, feature support, and descriptions.
    """
    # Key features to display
    key_features = [
        ('model_version', 'Versions'),
        ('flash_attention', 'Flash Attn'),
        ('system_prompt', 'Sys Prompt'),
        ('task_prompt', 'Task Prompt'),
        ('instruction_template', 'Instr Template'),
        ('model_mode', 'Modes'),
    ]
    
    # Build header
    feature_headers = [label for _, label in key_features]
    all_headers = ['Model', 'Min VRAM', 'Speed'] + feature_headers + ['Video']
    
    # Start HTML Table property
    # Using slightly transparent backgrounds for compatibility with light/dark themes
    html = "<table style='width: 100%; border-collapse: collapse;'>"
    
    # Table Head
    html += "<thead><tr style='border-bottom: 2px solid #555;'>"
    for h in all_headers:
        align = "left" if h == "Model" else "center"
        html += f"<th style='padding: 10px; text-align: {align}; font-size: 1.1em;'>{h}</th>"
    html += "</tr></thead>"
    
    html += "<tbody>"
    
    # config_mgr.list_models() returns models sorted by model_order
    for model_id in config_mgr.list_models():
        config = config_mgr.get_model_config(model_id)
        if not config:
            continue
        
        # Formatted Name & Description
        description = config.get('description', 'No description available.')
        model_path = config.get('model_path', '')
        
        description = description.replace('<br>', ' ').replace('<br/>', ' ').replace('\n', ' ')
        
        # Path display (Linked)
        if model_path:
            hf_url = f"https://huggingface.co/{model_path}"
            # Slightly smaller link as requested (0.85em)
            path_html = f"<div style='margin-bottom: 6px;'><a href='{hf_url}' target='_blank' style='text-decoration: none;'><code style='font-size: 0.85em; background: none; color: inherit; cursor: pointer;'>{model_path}</code></a></div>"
        else:
            path_html = ""
        
        # H2 Header for Model Name
        name_cell = (
            f"<div style='width: 400px; white-space: normal; padding: 5px;'>"
            f"<h2 style='margin: 0 0 5px 0; padding: 0; font-size: 1.4em;'>{model_id}</h2>"
            f"{path_html}"
            f"<span style='font-size: 0.9em; font-weight: normal; display: block; opacity: 0.9;'>{description}</span>"
            f"</div>"
        )
        
        # VRAM Stats (Min Only)
        vram_table = config.get('vram_table', {})
        min_vram, _ = get_vram_stats(vram_table)
        
        if min_vram == 0:
            vram_str = "N/A"
        else:
            vram_str = f"{min_vram} GB"
            
        # Speed Stats (New)
        speed_val = config.get('caption_speed', 0)
        if speed_val and int(speed_val) > 0:
            speed_str = f"{speed_val} it/s"
        else:
            speed_str = "-"
        
        # Determine Feature status (True/False)
        features = config.get('features', [])
        media_types = config.get('media_type', [])
        if isinstance(media_types, str): media_types = [media_types]
        has_video = 'Video' in media_types
        
        # Row Content
        html += "<tr style='border-bottom: 1px solid #444;'>"
        
        # Model Name Cell
        html += f"<td style='padding: 10px; vertical-align: top;'>{name_cell}</td>"
        
        # VRAM Cell
        html += f"<td style='padding: 10px; text-align: center; vertical-align: middle; font-size: 1.1em;'>{vram_str}</td>"
        
        # Speed Cell
        html += f"<td style='padding: 10px; text-align: center; vertical-align: middle; font-size: 1.1em;'>{speed_str}</td>"
        
        # Helper for Feature Cells
        def get_feat_cell(is_active):
            if is_active:
                # Green bg, Large Check
                style = "background-color: rgba(0, 128, 0, 0.15); color: #4ade80; font-size: 1.8em; font-weight: bold;"
                content = "✓"
            else:
                # Red bg, Empty (No X)
                style = "background-color: rgba(128, 0, 0, 0.1);"
                content = ""
            return f"<td style='padding: 5px; text-align: center; vertical-align: middle; {style}'>{content}</td>"
            
        # Add Feature Cells
        for feature_key, _ in key_features:
            html += get_feat_cell(feature_key in features)
            
        # Add Video Cell
        html += get_feat_cell(has_video)
        
        html += "</tr>"
    
    html += "</tbody></table>"
    
    return html


def create_model_info_tab(config_mgr):
    """
    Create the Model Information tab content.
    
    Args:
        config_mgr: ConfigManager instance with loaded model configs
    """
    gr.Markdown("### 📊 Model Information")
    
    with gr.Accordion(label="", open=True):
        grid_md = build_unified_model_table_markdown(config_mgr)
        gr.HTML(grid_md)
        
        # Legend (Vertical layout as requested)
        gr.HTML("""
        <div style="margin-top: 15px; padding: 10px; background-color: rgba(128, 128, 128, 0.05); border-radius: 5px;">
            <ul style="list-style-type: none; padding-left: 0; line-height: 1.6;">
                <li><strong>Min VRAM:</strong> Minimum GPU memory required to run the model.</li>
                <li><strong>Speed:</strong> Average iterations per second (it/s) measured on RTX 5090 (32GB VRAM) with batch size 64.</li>
                <li><strong>Versions:</strong> Model supports multiple variants (e.g., Quantized vs Full).</li>
                <li><strong>Flash Attn:</strong> Supports Flash Attention optimization for faster inference.</li>
                <li><strong>Sys Prompt:</strong> Supports custom System Prompts.</li>
                <li><strong>Modes:</strong> Supports special modes like Object Detection, Pointing, or Visual Query.</li>
            </ul>
        </div>
        """)
