"""
JoyTag Wrapper

Model: fancyfeast/joytag (HuggingFace Hub)

This wrapper implements JoyTag tagging with automatic model downloading from HuggingFace Hub.
Uses the VisionModel class from joytag_models/ submodule.
"""

from .base import BaseCaptionModel
from typing import List, Dict, Any
from PIL import Image
from pathlib import Path
import torch
import torchvision.transforms.functional as TVF


class JoyTagWrapper(BaseCaptionModel):
    """
    Wrapper for JoyTag tagging model (fancyfeast/joytag).
    
    Model-specific behavior:
    - Threshold-based tagging (save_threshold=0.2)
    - Outputs comma-separated tags
    - Uses custom image preprocessing (pad to square, BICUBIC resize)
    
    NOTE: This is a TAGGING model, not a language model
    """
    
    MODEL_ID = "fancyfeast/joytag"
    SAVE_THRESHOLD = 0.2
    TOKEN_LENGTH = 100
    
    def __init__(self, config):
        super().__init__(config)
        self.top_tags = []
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    def _load_model(self):
        """Load JoyTag model from HuggingFace Hub with automatic caching."""
        if self.model is not None:
            return
        
        # Import VisionModel from local joytag_models submodule
        from .joytag_models import VisionModel
        
        print("Loading JoyTag model from HuggingFace Hub...")
        
        # Download model files from HuggingFace Hub (automatically cached)
        # Download model files from HuggingFace Hub (automatically cached)
        from huggingface_hub import snapshot_download
        
        # Use config model_id
        model_id = self.config.get('model_id', self.MODEL_ID)
        
        model_path = snapshot_download(
            repo_id=model_id,
            allow_patterns=["*.safetensors", "*.json", "*.txt"]
        )
        print(f"  Model cached at: {model_path}")
        
        # Load model using VisionModel.load_model
        self.model = VisionModel.load_model(model_path)
        self.model.eval()
        self.model = self.model.to(self.device)
        
        # Load tags from the cached model directory
        tags_file = Path(model_path) / "top_tags.txt"
        with open(tags_file, 'r') as f:
            self.top_tags = [line.strip() for line in f.readlines() if line.strip()]
        
        print(f"JoyTag loaded on {self.device} with {len(self.top_tags)} tags")
        
        # DEBUG: Check model device
        try:
            sample_param = next(self.model.parameters())
            print(f"DEBUG: JoyTag model parameter device: {sample_param.device}")
        except Exception as e:
            print(f"DEBUG: Could not check model device: {e}")
    
    def _prepare_image(self, image: Image.Image, target_size: int) -> torch.Tensor:
        """Prepare image for JoyTag (pad to square, resize, normalize)."""
        # Pad image to square
        image_shape = image.size
        max_dim = max(image_shape)
        pad_left = (max_dim - image_shape[0]) // 2
        pad_top = (max_dim - image_shape[1]) // 2
        
        padded_image = Image.new('RGB', (max_dim, max_dim), (255, 255, 255))
        padded_image.paste(image, (pad_left, pad_top))
        
        # Resize image
        if max_dim != target_size:
            padded_image = padded_image.resize((target_size, target_size), Image.BICUBIC)
        
        # Convert to tensor and normalize
        image_tensor = TVF.pil_to_tensor(padded_image) / 255.0
        image_tensor = TVF.normalize(
            image_tensor,
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711]
        )
        
        return image_tensor
    
    
    def _run_inference(self, images: List[Image.Image], prompt: List[str], args: Dict[str, Any]) -> List[str]:
        """Run JoyTag tagging on images. Prompt is ignored."""
        results = []
        batch_size = int(args.get('batch_size', 1))
        
        # Process in batches
        for i in range(0, len(images), batch_size):
            chunk = images[i : i + batch_size]
            
            # Prepare batch tensors
            tensors = []
            for img in chunk:
                tensors.append(self._prepare_image(img, self.model.image_size))
            
            # Stack into [B, C, H, W]
            batch_tensor = torch.stack(tensors).to(self.device)
            batch_input = {'image': batch_tensor}
            
            with torch.no_grad():
                with torch.amp.autocast_mode.autocast(self.device, enabled=True):
                    preds = self.model(batch_input)
                    # preds['tags'] is [B, NumTags]
                    tag_preds = preds['tags'].sigmoid().cpu()
            
            # Process results for each image in the batch
            for j in range(len(chunk)):
                # Get scores for this image
                # tag_preds[j] corresponds to chunk[j]
                current_preds = tag_preds[j]
                
                scores = {self.top_tags[k]: current_preds[k] for k in range(len(self.top_tags))}
                
                # Use config threshold
                threshold = args.get('threshold', self.config.get('defaults', {}).get('save_threshold', self.SAVE_THRESHOLD))
                save_tags = [tag for tag, score in scores.items() if score > threshold]
                tag_string = ', '.join(save_tags)
                
                # Apply token length constraint
                token_length = self.config.get('defaults', {}).get('token_length', self.TOKEN_LENGTH)
                if token_length:
                    words = tag_string.split(', ')
                    if len(words) > token_length:
                        tag_string = ', '.join(words[:token_length])
                
                results.append(tag_string)
            
            # Optional: Explicitly clean up batch tensor
            del batch_tensor
            del batch_input
            del preds
            del tag_preds
        
        return results
    
    def unload(self):
        """Free model resources using shared utility."""
        from src.core.model_utils import unload_model, UnloadMode
        unload_model(self.model, None, UnloadMode.STANDARD)
        self.model = None
        self.top_tags = []

