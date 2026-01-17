"""Inspector and gallery interaction logic."""

import gradio as gr


def open_inspector(app, evt: gr.SelectData):
    """Open the inspector panel for a selected media item."""
    if not evt:
        return (
            gr.update(visible=False),
            gr.update(selected="img_tab"),
            None,
            None,
            ""
        )
    
    page_offset = (app.current_page - 1) * app.gallery_items_per_page
    index = evt.index + page_offset
    
    app.selected_index = index
    
    if index < len(app.dataset.images):
        media_obj = app.dataset.images[index]
        app.selected_path = media_obj.path
        
        if media_obj.is_video():
            return (
                gr.update(visible=True),
                gr.update(selected="vid_tab"),
                gr.update(value=None, visible=False),
                gr.update(value=str(media_obj.path), visible=True),
                media_obj.caption
            )
        else:
            return (
                gr.update(visible=True),
                gr.update(selected="img_tab"),
                gr.update(value=str(media_obj.path), visible=True),
                gr.update(value=None, visible=False),
                media_obj.caption
            )
    
    return (
        gr.update(visible=False),
        gr.update(selected="img_tab"),
        None,
        None,
        ""
    )


def remove_from_gallery(app):
    """Remove currently selected image from dataset (not from disk)."""
    if app.selected_index is not None and 0 <= app.selected_index < len(app.dataset.images):
        app.dataset.images.pop(app.selected_index)
        app.selected_index = None
        app.selected_path = None
        return app._get_gallery_data(), gr.update(visible=False), app.current_page, app.get_total_label(), app._get_pagination_vis()
    else:
        gr.Warning("No image selected")
        return app._get_gallery_data(), gr.update(visible=True), app.current_page, app.get_total_label(), app._get_pagination_vis()


def save_and_close(app, caption):
    """Save caption and close inspector."""
    if app.selected_path:
        for img in app.dataset.images:
            if img.path == app.selected_path:
                img.update_caption(caption)
                img.save_caption()
                break
    app._save_dataset_list()
    return app._get_gallery_data(), gr.update(visible=False)


def close_inspector():
    """Close the inspector panel."""
    return gr.update(visible=False)


def clear_gallery(app):
    """Clear the dataset, gallery, and close inspector."""
    from src.core.dataset import Dataset
    app.dataset = Dataset()
    app.selected_path = None
    app.current_page = 1
    return [], gr.update(visible=False), 1, app.get_total_label(), app._get_pagination_vis()
