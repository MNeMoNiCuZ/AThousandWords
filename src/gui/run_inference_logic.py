"""Run inference logic - core inference execution."""

import copy
import gradio as gr
import logging

from src.core.registry import ModelRegistry

logger = logging.getLogger("GUI")


def run_inference(app, mod, args):
    """Run inference on the dataset."""
    if "gpu_vram" not in args:
        settings = app.config_mgr.get_global_settings()
        args["gpu_vram"] = settings['gpu_vram']

    if app.is_drag_and_drop and not args.get("output_dir"):
        print("Detected Drag & Drop input with no output directory. Defaulting to 'output'.")
        args["output_dir"] = "output"

    cli_command = app.generate_cli_command(mod, args, skip_defaults=False)

    total_images = len(app.dataset.images) if app.dataset and app.dataset.images else 0
    common_root, mixed_sources, collisions = app.analyze_input_paths()
    
    if collisions and not common_root:
        gr.Warning(f"Filename collisions detected: {collisions[:3]}. Cannot save safely to single output.")
        return app._get_gallery_data(), gr.update(visible=False), {}
    
    if mixed_sources:
        msg = "⚠️ Warning: Inputs are from different drives/locations. Output files will be flattened into the output folder."
        gr.Warning(msg)
        print(msg)

    if common_root:
        args['input_root'] = common_root

    print(f"--- Starting Inference Run for {mod} ---")
    print("")
    print(cli_command)
    print("")
    
    if not app.dataset or not app.dataset.images:
        gr.Warning("No images found. Please load a folder or add images to the 'Input Source' before running.")
        return app._get_gallery_data(), gr.update(visible=False), {}

    try:
        limit_count = args.get('limit_count', 0)
        run_dataset = app.dataset
        
        try:
            limit = int(limit_count)
            if limit > 0 and len(app.dataset.images) > limit:
                print(f"Limiting to first {limit} files (from {len(app.dataset.images)} total).")
                run_dataset = copy.copy(app.dataset)
                run_dataset.images = app.dataset.images[:limit]
        except (ValueError, TypeError):
            pass
    
        generated_files, stats = ModelRegistry.load_wrapper(mod).run(run_dataset, args)
        
        if isinstance(generated_files, list) and generated_files:
            # unique files only to prevent zip duplicates
            unique_files = sorted(list(set(generated_files)))
            zip_path = app.create_zip(unique_files)
            if zip_path:
                return app._get_gallery_data(), gr.update(visible=True, value=zip_path), stats
        
        return app._get_gallery_data(), gr.update(visible=False), stats
        
    except (RuntimeError, OSError) as e:
        error_msg = str(e).lower()
        if "out of memory" in error_msg or "vram" in error_msg:
            gr.Warning("🔴 CUDA OUT OF MEMORY - Reduce batch size, resize images, or use a smaller model.")
            return app._get_gallery_data(), gr.update(visible=False), {}
            
        if "permission denied" in error_msg or "permissionerror" in error_msg:
            # Common Docker/HF cache issue
            msg = (
                "🔴 FILE PERMISSION ERROR: A model file is locked or corrupted.\n"
                "Try restarting the container. If it persists, you may need to clear the 'models' folder manually."
            )
            gr.Warning(msg)
            print(f"Error details: {e}")
            return app._get_gallery_data(), gr.update(visible=False), {}
            
        import traceback
        traceback.print_exc()
        raise gr.Error(f"Processing failed: {str(e)}")
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        raise gr.Error(f"Processing failed: {str(e)}")
