"""Inspector and gallery interaction logic."""

import gradio as gr


def open_inspector(app, evt: gr.SelectData):
    """
    Open the inspector panel for the media item represented by the given gallery selection.
    
    Sets app.selected_index and, when the index is valid, app.selected_path to the selected item's path. If no event is provided or the computed index is out of range, the inspector is hidden and fields are cleared.
    
    Parameters:
        app: Application state object containing dataset, pagination, and selection attributes.
        evt (gr.SelectData | None): Selection event from the gallery; its `index` is interpreted relative to the current page. If falsy, the inspector will be closed.
    
    Returns:
        tuple: (inspector_visibility, selected_tab, image_input_update, video_input_update, caption)
            - inspector_visibility: gr.update controlling inspector panel visibility.
            - selected_tab: gr.update selecting either the image or video tab.
            - image_input_update: gr.update for the image path input when an image is selected, or None/hidden otherwise.
            - video_input_update: gr.update for the video path input when a video is selected, or None/hidden otherwise.
            - caption: caption string for the selected media, or empty string when no valid selection exists.
    """
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
    """
    Remove the currently selected image from the in-memory dataset and update inspector state.
    
    If an image was removed, clears the selection (selected_index and selected_path) and hides the inspector.
    If no image is selected, emits a warning and leaves the dataset and selection unchanged.
    
    Returns:
        tuple: (
            gallery_data — list of gallery items to display (updated),
            inspector_visibility_update — gr.update() controlling inspector visibility,
            current_page — current page number (int),
            total_label — text for the total/count label,
            pagination_visibility — pagination visibility state
        )
    """
    if app.selected_index is not None and 0 <= app.selected_index < len(app.dataset.images):
        app.dataset.images.pop(app.selected_index)
        app.selected_index = None
        app.selected_path = None
        return app._get_gallery_data(), gr.update(visible=False), app.current_page, app.get_total_label(), app._get_pagination_vis()
    else:
        gr.Warning("No image selected")
        return app._get_gallery_data(), gr.update(visible=True), app.current_page, app.get_total_label(), app._get_pagination_vis()


def save_and_close(app, caption):
    """
    Save the provided caption to the currently selected media item and close the inspector.
    
    Parameters:
        app: Application state object containing `dataset.images`, `selected_path`, and persistence helpers.
        caption (str): Caption text to assign to the selected media item.
    
    Returns:
        tuple: A two-element tuple where the first element is the updated gallery data list and the second is a Gradio update object that hides the inspector (`gr.update(visible=False)`).
    """
    if app.selected_path:
        for img in app.dataset.images:
            if img.path == app.selected_path:
                img.update_caption(caption)
                img.save_caption()
                break
    app._save_dataset_list()
    return app._get_gallery_data(), gr.update(visible=False)


def close_inspector():
    """
    Close the inspector panel.
    
    Returns:
        gr_update: A Gradio update object that sets the component's visibility to False.
    """
    return gr.update(visible=False)


def clear_gallery(app):
    """
    Reset the gallery dataset, clear the current selection, reset pagination to page 1, and close the inspector.
    
    Parameters:
        app: Application state object whose `dataset`, `selected_path`, and `current_page` will be reset.
    
    Returns:
        A 5-tuple containing:
        - `gallery_items` (list): Empty list representing cleared gallery contents.
        - `inspector_update` (gr.update): Gradio update to hide the inspector panel.
        - `current_page` (int): The reset current page number (`1`).
        - `total_label` (str): Text for the total-count label after clearing the dataset.
        - `pagination_visible` (bool): Whether pagination should be visible after the reset.
    """
    from src.core.dataset import Dataset
    app.dataset = Dataset()
    app.selected_path = None
    app.current_page = 1
    return [], gr.update(visible=False), 1, app.get_total_label(), app._get_pagination_vis()