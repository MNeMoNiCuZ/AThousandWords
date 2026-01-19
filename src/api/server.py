"""
A Thousand Words - API Server

This module provides a REST API wrapper around the captioning functionality.
It allows remote clients to submit images for captioning via HTTP.
"""

import os
import sys
import logging
import uuid
import shutil
import tempfile
import asyncio
from pathlib import Path
from typing import List, Optional, Dict, Any

# Ensure imports work from root directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from src.core.config import ConfigManager
from src.core.registry import ModelRegistry
from src.core.loader import DataLoader
import src.features as feature_registry
from src.core.console_kit import console

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("API")

app = FastAPI(title="A Thousand Words API", version="1.0.0")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Utilities ---

def cleanup_job_dir(job_dir: Path):
    """Background task to clean up job directory properly."""
    try:
        if job_dir.exists():
            shutil.rmtree(job_dir)
            logger.info(f"Cleaned up job directory: {job_dir}")
    except Exception as e:
        logger.error(f"Failed to cleanup {job_dir}: {e}")

# --- Endpoints ---

@app.get("/api/health")
async def health_check():
    return {"status": "ok"}

@app.get("/api/models")
async def list_models():
    """List available model IDs."""
    config_mgr = ConfigManager()
    models = config_mgr.list_models()
    return {"models": models}

@app.post("/api/caption")
async def caption_images(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...),
    model: str = Form(...),
    # Common overrideable parameters
    batch_size: Optional[int] = Form(None),
    task_prompt: Optional[str] = Form(None),
    max_tokens: Optional[int] = Form(None),
    temperature: Optional[float] = Form(None),
    recursive: bool = Form(False), # Ignored for direct upload, but kept for compatibility
):
    """
    Submit images for captioning.
    
    Args:
        files: List of image files to caption
        model: Model ID to use (must be valid)
        ...other params as form fields
    
    Returns:
        JSON with caption results
    """
    job_id = uuid.uuid4().hex
    # Use absolute path relative to project root (2 levels up from src/api)
    project_root = Path(__file__).parent.parent.parent.absolute()
    temp_root = project_root / "temp" / "api"
    job_dir = temp_root / job_id
    input_dir = job_dir / "input"
    output_dir = job_dir / "output" # For any saved files if needed
    
    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # 1. Save uploaded files
        saved_files = []
        for file in files:
            safe_name = Path(file.filename).name
            file_path = input_dir / safe_name
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            saved_files.append(file_path)
            
        logger.info(f"Job {job_id}: Received {len(saved_files)} files. Model: {model}")
        
        # 2. Validate Model
        config_mgr = ConfigManager()
        if model not in config_mgr.list_models():
            raise HTTPException(status_code=400, detail=f"Unknown model: {model}")

        # 3. Build Arguments (similar to captioner.py)
        # Load defaults
        global_defaults = config_mgr.get_global_settings()
        model_config = config_mgr.get_model_config(model)
        
        # Determine version (default to config if not provided)
        # simplified for API: use default version
        raw_defaults = model_config.get('defaults', {})
        version = raw_defaults.get('model_version') if isinstance(raw_defaults, dict) else None
        
        model_defaults = config_mgr.get_version_defaults(model, version)
        
        args = {}
        all_features = feature_registry.get_all_features()
        
        for name, feature in all_features.items():
            # Check if provided in Form data (simplification: strict param names)
            # In a real app, we might want to support --flag-style names too
            # Here we just use the python variable names as form fields
            
            # NOTE: FastAPI Form(...) doesn't automatically map all possibilities.
            # We implemented a few common ones as explicit args (batch_size, etc).
            # For full support, one might parse `request` directly or add all args.
            # For this implementation, we'll stick to defaults + explicit overrides.
            
            # Use explicit local var if it exists (e.g. batch_size param)
            local_val = locals().get(name) 
            
            # Prioritize local keys (like model specific params) that might be in the args dict
            if name in args:
                pass # Already set via cli_args logic equivalent
            elif local_val is not None:
                args[name] = local_val
            elif name in model_defaults:
                args[name] = model_defaults[name]
            elif name in global_defaults:
                args[name] = global_defaults[name]
            else:
                args[name] = feature.get_default()
        
        # Force some args for API context
        args["output_dir"] = str(output_dir) # Use temp output dir
        args["gpu_vram"] = config_mgr.user_config.get('gpu_vram', 24)
        args["overwrite"] = True
        args["print_console"] = False # Don't spam server console
        
        # 4. Load Dataset
        dataset = DataLoader.scan_directory(str(input_dir))
        
        if len(dataset) == 0:
            raise HTTPException(status_code=400, detail="No valid images found in upload")

        # 5. Run Inference
        # We need to capture the results. The wrapper usually writes to file or updates dataset objects.
        # Most wrappers in this codebase write files (caption file).
        # We can extract the captions from the dataset objects after run() if the wrapper updates them.
        # Let's check if wrappers update dataset.images[i].caption
        
        wrapper = ModelRegistry.load_wrapper(model)
        
        # Run synchronous inference
        wrapper.run(dataset, args)
        
        # 6. Collect Results
        results = []
        for img in dataset.images:
            # Read caption from file if it exists (since wrappers save to file)
            # Or use img.caption if updated
            
            # Try to read the expected output file
            # Assuming default extension .txt
            ext = args.get("output_format", ".txt")
            if not ext.startswith("."): ext = "." + ext
            
            # The output would be in output_dir (if wrapper respects it) or input_path parent
            # Standard wrappers use dataset paths.
            # args["output_dir"] was set above. The wrappers should use it.
            
            # Check output dir for the file
            out_file = output_dir / img.path.with_suffix(ext).name
            content = ""
            
            if out_file.exists():
                with open(out_file, "r", encoding="utf-8") as f:
                    content = f.read()
            elif img.caption:
                content = img.caption
                
            results.append({
                "filename": img.path.name,
                "caption": content
            })

        return {"status": "success", "results": results}
        
    except Exception as e:
        logger.error(f"Error processing job {job_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
        
    finally:
        # Schedule cleanup
        background_tasks.add_task(cleanup_job_dir, job_dir)

def create_app():
    return app

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
