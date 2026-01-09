"""
Model Utilities

Shared utilities for model loading and unloading.
Handles different model types: device_map, standard, ONNX.
"""

import gc
from enum import Enum
from typing import Any, Optional


class UnloadMode(Enum):
    """Unload modes for different model types."""
    DEVICE_MAP = "device_map"  # For accelerate/quantized models with device_map
    STANDARD = "standard"      # For models loaded with .to(device)
    ONNX = "onnx"             # For ONNX Runtime models


def unload_model(
    model: Any,
    processor: Any = None,
    mode: UnloadMode = UnloadMode.STANDARD,
    extra_refs: Optional[list] = None
) -> None:
    """
    Properly unload a model and free GPU memory.
    
    Args:
        model: The model object to unload
        processor: Optional processor/tokenizer to unload
        mode: UnloadMode indicating how the model was loaded
        extra_refs: Optional list of additional objects to delete
    """
    import torch
    
    if model is not None:
        if mode == UnloadMode.DEVICE_MAP:
            # For device_map models, remove accelerate hooks first
            try:
                from accelerate.hooks import remove_hook_from_submodules
                remove_hook_from_submodules(model)
            except Exception:
                pass
            
            # Move to CPU before deletion
            try:
                model.to('cpu')
            except Exception:
                # Quantized models often fail to move to CPU, ignore
                pass
        
        elif mode == UnloadMode.STANDARD:
            # Standard models - just move to CPU
            try:
                model.to('cpu')
            except Exception:
                pass
        
        elif mode == UnloadMode.ONNX:
            # ONNX models - no GPU movement needed, just delete
            pass
        
        # NUCLEAR OPTION: Shred tensors to force visual memory release
        # This clears the data underlying the parameters even if the object hangs around
        try:
            for param in model.parameters():
                if hasattr(param, 'data'):
                    param.data = torch.empty(0)
                if hasattr(param, 'grad') and param.grad is not None:
                    param.grad = None
        except Exception:
            pass
            
        # Delete the model
        del model
    
    # Delete processor if provided
    if processor is not None:
        del processor
    
    # Delete any extra references
    if extra_refs:
        for ref in extra_refs:
            if ref is not None:
                del ref
    
    # Force garbage collection (Multiple cycles)
    gc.collect()
    gc.collect()
    
    # Clear CUDA cache with sync
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass


def get_unload_mode_for_variant(model_id: str, variant: str = None) -> UnloadMode:
    """
    Determine the appropriate unload mode based on model/variant.
    
    Args:
        model_id: The model identifier
        variant: Optional variant name (e.g., "Quantized (nf4)")
    
    Returns:
        UnloadMode appropriate for this model/variant
    """
    # Check for quantized/device_map indicators
    quantized_indicators = ['nf4', 'int4', 'int8', 'bnb', 'gptq', 'awq']
    
    model_lower = model_id.lower() if model_id else ""
    variant_lower = variant.lower() if variant else ""
    
    for indicator in quantized_indicators:
        if indicator in model_lower or indicator in variant_lower:
            return UnloadMode.DEVICE_MAP
    
    # Check for ONNX
    if 'onnx' in model_lower:
        return UnloadMode.ONNX
    
    # Default to standard
    return UnloadMode.STANDARD
