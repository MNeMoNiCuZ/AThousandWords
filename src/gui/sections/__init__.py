"""Input Source section UI factory."""

import gradio as gr


def create_input_source(app):
    """
    Create the Input Source accordion with drag-drop, path input, and controls.
    
    Args:
        app: CaptioningApp instance
        
    Returns:
        dict: Component references
    """
    with gr.Accordion("Input Source", open=True):
        with gr.Row():
            with gr.Column(scale=1):
                input_files = gr.File(
                    label="Drop Images or Folders",
                    file_count="multiple",
                    type="filepath",
                    height=130
                )

            with gr.Column(scale=1):
                input_path_text = gr.Textbox(
                    label="Input Folder Path", 
                    placeholder="C:/Path/To/Images", 
                    value="",
                    lines=1
                )
                image_count = gr.Markdown(f"<center><b style='font-size: 1.2em'>{len(app.dataset.images)} images</b></center>")

            with gr.Column(scale=1, min_width=200):
                load_source_btn = gr.Button(
                    "Load Images From Input", 
                    variant="primary", 
                    size="lg"
                )
                with gr.Row():
                    limit_count = gr.Textbox(
                        placeholder="Limit", 
                        show_label=False, 
                        container=False, 
                        min_width=80,
                        scale=1
                    )
                    
                    clear_gallery_btn = gr.Button(
                        "Clear Dataset Gallery", 
                        variant="secondary", 
                        size="lg",
                        min_width=200,
                        scale=3
                    )
        
        with gr.Row():
            gr.Markdown(
                "<div style='font-size: 0.8em; color: gray; margin-left: 10px;'>"
                "⚠️ <b>Note:</b> Dragged files are saved to a temporary location and outputs will be placed in the configured output folder (default: /output)."
                "</div>"
            )
    
    return {
        "input_files": input_files,
        "input_path_text": input_path_text,
        "image_count": image_count,
        "load_source_btn": load_source_btn,
        "limit_count": limit_count,
        "clear_gallery_btn": clear_gallery_btn
    }


def create_gallery_section(app):
    """
    Create the Dataset Gallery section with pagination.
    
    Args:
        app: CaptioningApp instance
        
    Returns:
        dict: Component references including gallery_group wrapper
    """
    with gr.Group(visible=app.gallery_rows > 0, elem_id="gallery_group") as gallery_group:
        with gr.Accordion("🖼️ Dataset Gallery", open=True) as gallery_accordion:
            with gr.Row(elem_classes="pagination-row compact-pagination", visible=False) as pagination_row:
                prev_btn = gr.Button("◀", variant="secondary", size="sm", elem_classes="pagination-btn")
                
                page_number_input = gr.Number(
                    value=app.current_page,
                    label=None,
                    show_label=False,
                    precision=0,
                    minimum=1,
                    container=False,
                    elem_classes="pagination-input",
                    scale=0,
                    min_width=60
                )
                
                total_pages_label = gr.Markdown(value=f"/ {app.get_total_pages()}", elem_classes="pagination-label")
                
                next_btn = gr.Button("▶", variant="secondary", size="sm", elem_classes="pagination-btn")

            gal = gr.Gallery(
                label=None,
                columns=app.gallery_columns,
                height=app.calc_gallery_height(),
                object_fit="contain",
                allow_preview=False,
                show_label=True,
                elem_classes="gallery-section",
                elem_id="main_gallery"
            )
    
    return {
        "gallery_group": gallery_group,
        "gallery_accordion": gallery_accordion,
        "pagination_row": pagination_row,
        "prev_btn": prev_btn,
        "page_number_input": page_number_input,
        "total_pages_label": total_pages_label,
        "next_btn": next_btn,
        "gal": gal
    }


def create_viewer_section():
    """
    Create the Inspector/Viewer section for viewing and editing captions.
    
    Returns:
        dict: Component references including inspector_group wrapper
    """
    with gr.Group(visible=False) as inspector_group:
        gr.Markdown("### 🔍 Viewer")
        with gr.Column(elem_classes="input-section"):
            with gr.Row():
                with gr.Tabs() as insp_tabs:
                    with gr.Tab("Image", id="img_tab") as img_tab:
                        insp_img = gr.Image(label=None, interactive=False, height=600, show_label=False)
                    with gr.Tab("Video", id="vid_tab") as vid_tab:
                        insp_video = gr.Video(label=None, interactive=False, height=600, show_label=False, autoplay=False)
                with gr.Column():
                    insp_cap = gr.TextArea(label="Caption", lines=15)
                    with gr.Row():
                        save_cap_btn = gr.Button("💾 Save Caption", variant="primary", scale=1)
                        insp_remove_btn = gr.Button("🗑️ Remove", variant="stop", scale=0, min_width=120)
                    close_insp_btn = gr.Button("Close Viewer", variant="secondary")
    
    return {
        "inspector_group": inspector_group,
        "insp_tabs": insp_tabs,
        "insp_img": insp_img,
        "insp_video": insp_video,
        "insp_cap": insp_cap,
        "save_cap_btn": save_cap_btn,
        "insp_remove_btn": insp_remove_btn,
        "close_insp_btn": close_insp_btn
    }


def wire_dataset_events(app, input_components, gallery_components, viewer_components, 
                         recursive_checkbox=None, items_per_page_input=None):
    """
    Wire ALL event handlers for dataset, gallery, pagination, and viewer sections.
    
    Args:
        app: CaptioningApp instance
        input_components: Dict from create_input_source
        gallery_components: Dict from create_gallery_section
        viewer_components: Dict from create_viewer_section
        recursive_checkbox: Optional recursive checkbox component
        items_per_page_input: Optional items per page setting component
    """
    import gradio as gr
    
    inp = input_components
    gal_c = gallery_components
    view = viewer_components
    
    gal = gal_c["gal"]
    pagination_outputs = [gal, gal_c["page_number_input"], gal_c["total_pages_label"], gal_c["pagination_row"]]
    
    # Load/Clear events
    load_inputs = [inp["input_path_text"]]
    if recursive_checkbox:
        load_inputs.append(recursive_checkbox)
    load_inputs.append(inp["limit_count"])
    
    inp["load_source_btn"].click(
        app.load_input_source, 
        inputs=load_inputs, 
        outputs=[gal, inp["input_files"]] + pagination_outputs[1:]
    )
    inp["clear_gallery_btn"].click(
        app.clear_gallery, 
        outputs=[gal, view["inspector_group"]] + pagination_outputs[1:]
    )
    inp["input_files"].change(
        app.load_files, 
        inputs=[inp["input_files"]], 
        outputs=[gal, inp["input_files"]] + pagination_outputs[1:]
    )
    
    # Image count updates
    def get_image_count():
        count = len(app.dataset.images) if app.dataset else 0
        return f"<center><b style='font-size: 1.2em'>{count} images</b></center>"
    
    inp["load_source_btn"].click(get_image_count, outputs=[inp["image_count"]])
    inp["clear_gallery_btn"].click(get_image_count, outputs=[inp["image_count"]])
    inp["input_files"].change(get_image_count, outputs=[inp["image_count"]])
    
    # Pagination events
    gal_c["prev_btn"].click(app.prev_page, inputs=None, outputs=pagination_outputs)
    gal_c["next_btn"].click(app.next_page, inputs=None, outputs=pagination_outputs)
    gal_c["page_number_input"].submit(app.jump_to_page, inputs=[gal_c["page_number_input"]], outputs=pagination_outputs)
    
    # Items per page change refreshes gallery
    if items_per_page_input:
        items_per_page_input.change(app.update_items_per_page, inputs=[items_per_page_input], outputs=pagination_outputs)
    
    # Viewer/Inspector events
    gal.select(app.open_inspector, outputs=[view["inspector_group"], view["insp_tabs"], view["insp_img"], view["insp_video"], view["insp_cap"]])
    view["save_cap_btn"].click(app.save_and_close, inputs=[view["insp_cap"]], outputs=[gal, view["inspector_group"]])
    view["close_insp_btn"].click(app.close_inspector, outputs=[view["inspector_group"]])
    view["insp_remove_btn"].click(
        app.remove_from_gallery,
        outputs=[gal, view["inspector_group"]] + pagination_outputs[1:]
    ).then(
        get_image_count,
        outputs=[inp["image_count"]]
    )
    
    return get_image_count  # Return for external use if needed


def wire_gallery_settings(app, gal_cols, gal_rows_slider, gallery_group, gal):
    """Wire gallery column/row settings changes."""
    import gradio as gr
    
    def refresh_cols(val):
        app.gallery_columns = val
        return gr.update(columns=val, height=app.calc_gallery_height(), value=app._get_gallery_data())
    
    gal_cols.change(refresh_cols, inputs=[gal_cols], outputs=[gal])
    
    def refresh_vis_rows(val):
        app.gallery_rows = val
        is_visible = val > 0
        pixel_height = app.calc_gallery_height()
        return (
            gr.update(visible=is_visible),
            gr.update(height=pixel_height, value=app._get_gallery_data()),
        )
    
    gal_rows_slider.change(refresh_vis_rows, inputs=[gal_rows_slider], outputs=[gallery_group, gal])

