# JoyTag Model Classes
# Source: https://huggingface.co/fancyfeast/joytag
# Required for loading the custom VisionModel/ViT architecture

# The Models.py file is copied from the JoyTag repository and contains
# the VisionModel base class and ViT implementation needed for inference.

from .Models import VisionModel

__all__ = ["VisionModel"]
