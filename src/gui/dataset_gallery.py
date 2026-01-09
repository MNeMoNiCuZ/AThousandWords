"""
Reusable Dataset Gallery Component
Provides the input controls and gallery for loading/viewing datasets.
"""

import gradio as gr


def create_dataset_gallery(app, gallery_id_suffix="", show_headers=True):
    """
    Create a reusable dataset gallery component with input controls.
    
    Args:
        app: CaptioningApp instance
        gallery_id_suffix: Suffix to make component IDs unique (optional)
        show_headers: Whether to show section headers (False when wrapped in Accordion)
        
    Returns:
        tuple: (input_files, load_folder_btn, clear_gallery_btn, gallery)
    """
    with gr.Column(elem_classes="input-section"):
        if show_headers:
            gr.Markdown("### 📂 Input Source")
        with gr.Row():
            input_files = gr.File(
                label="Drop Images or Folders",
                file_count="directory", 
                height=130,
                elem_id=f"input_files{gallery_id_suffix}" if gallery_id_suffix else None
            )
            with gr.Column(scale=0, min_width=200):
                load_folder_btn = gr.Button(
                    "Load Input Folder", 
                    variant="secondary", 
                    size="lg",
                    elem_id=f"load_folder_btn{gallery_id_suffix}" if gallery_id_suffix else None
                )
                clear_gallery_btn = gr.Button(
                    "Clear Dataset Gallery", 
                    variant="secondary", 
                    size="lg",
                    elem_id=f"clear_gallery_btn{gallery_id_suffix}" if gallery_id_suffix else None
                )
        
        if show_headers:
            gr.Markdown("### 🖼️ Dataset Gallery")
        gallery = gr.Gallery(
            label=None,
            columns=app.gallery_columns,
            height=600,
            object_fit="contain",
            allow_preview=False,
            show_label=True,
            elem_classes="gallery-section",
            elem_id=f"gallery{gallery_id_suffix}" if gallery_id_suffix else None
        )
    
    return input_files, load_folder_btn, clear_gallery_btn, gallery
