"""
Hardware detection utilities.
"""
import math
import logging

logger = logging.getLogger(__name__)

def get_vram_gb() -> int:
    """
    Attempts to detect the total VRAM of the primary GPU in GB.
    Returns None if detection fails or is unavailable.
    """
    try:
        import torch
        if torch.cuda.is_available():
            # Get properties for device 0 (primary)
            props = torch.cuda.get_device_properties(0)
            total_memory_bytes = props.total_memory
            # Convert to GB and round to nearest integer
            total_memory_gb = round(total_memory_bytes / (1024**3))
            # logger.info(f"Detected {total_memory_gb} GB VRAM on {props.name}")
            return total_memory_gb
        else:
            logger.warning("CUDA not available, cannot detect VRAM.")
            return None
    except ImportError:
        logger.warning("Torch not found, cannot detect VRAM.")
        return None
    except Exception as e:
        logger.error(f"Error detecting VRAM: {e}")
        return None
