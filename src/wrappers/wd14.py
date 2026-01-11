"""
WD14 Tagger Wrapper

Model: SmilingWolf/wd-*-tagger-v3 (HuggingFace Hub)

This wrapper implements WD14 tagging using ONNX Runtime.
It supports multiple model variants selected via config.
"""

from .base import BaseCaptionModel
from typing import List, Dict, Any, Tuple
from PIL import Image
import numpy as np
import csv
import os
import cv2  # OpenCV for image preprocessing (standard for WD14)
from huggingface_hub import snapshot_download

# Check for onnxruntime availability
try:
    import onnxruntime as ort
    ORT_AVAILABLE = True
except ImportError:
    ORT_AVAILABLE = False


class WD14Wrapper(BaseCaptionModel):
    """
    Wrapper for WD14 Tagger models (ONNX based).
    
    Model-specific behavior:
    - Uses ONNX Runtime for inference
    - Resizes images to 448x448 (standard for v3 models)
    - Outputs comma-separated tags based on threshold (default 0.35)
    - Supports model variants (ViT, SwinV2, ConvNext, etc.)
    """
    
    # Tag indices for exclusion (0-3 are ratings: general, sensitive, questionable, explicit)
    RATING_INDICES = [0, 1, 2, 3]
    
    def __init__(self, config):
        if not ORT_AVAILABLE:
            raise ImportError(
                "onnxruntime is required for WD14 Tagger. "
                "Please install it using: pip install onnxruntime onnxruntime-gpu"
            )
        super().__init__(config)
        self.tags = []
        self.character_tags = []
        self.general_tags = []
        self.model_version = None
        self.current_version = None
    
    def _load_model(self):
        """Load WD14 ONNX model and tags from HuggingFace Hub."""
        if self.model is not None:
            return
        
        # Get model ID from version selection or default
        # Access args set by BaseCaptionModel before _load_model is called
        current_args = getattr(self, 'current_args', {})
        version_name = current_args.get('model_version') or self.config.get('defaults', {}).get('model_version', 'ViT v3')
        
        # Look up repo ID from config model_versions
        versions = self.config.get('model_versions', {})
        repo_id = versions.get(version_name, version_name) # Fallback to name if not found
            
        print(f"Loading WD14 model: {repo_id}...")
        
        # Download model files
        model_path = snapshot_download(
            repo_id=repo_id,
            allow_patterns=["model.onnx", "selected_tags.csv"]
        )
        
        # Load Tags
        tags_file = os.path.join(model_path, "selected_tags.csv")
        self.tags = []
        with open(tags_file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader) # Skip header
            for row in reader:
                # Format: tag_id, name, category, count
                self.tags.append(row[1])
        
        print(f"  Loaded {len(self.tags)} tags from {tags_file}")
        
        # Load ONNX Model
        onnx_file = os.path.join(model_path, "model.onnx")
        
        # Set providers (CUDA if available, else CPU)
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        if 'CUDAExecutionProvider' not in ort.get_available_providers():
            providers = ['CPUExecutionProvider']
            print("  Warning: CUDA provider not found for ONNX Runtime. Running on CPU.")
        
        # Configure session options to suppress verbose logging
        sess_options = ort.SessionOptions()
        sess_options.log_severity_level = 3  # 0=Verbose, 1=Info, 2=Warning, 3=Error, 4=Fatal
        
        self.model = ort.InferenceSession(onnx_file, sess_options=sess_options, providers=providers)
        
        # Get input name
        self.input_name = self.model.get_inputs()[0].name
        self.label_name = self.model.get_outputs()[0].name
        
        print(f"WD14 Tagger loaded on {providers[0]}")
        
    def _preprocess_image(self, image: Image.Image, args: Dict[str, Any] = None) -> np.ndarray:
        """
        Preprocess image for WD14:
        1. Maintain aspect ratio, pad to square with white background
        2. Resize to 448x448
        3. Convert from RGB to BGR (OpenCV format)
        4. Normalize
        """
        if args is None:
            args = {}
            
        # Convert PIL to BGR numpy array
        # Note: WD14 models trained with cv2 which reads as BGR
        img = np.array(image)
        if img.shape[2] == 4:
            # Handle RGBA - convert to RGB white background
            alpha = img[:, :, 3]
            img_rgb = img[:, :, :3]
            bg = np.ones_like(img_rgb) * 255
            # Alpha blending
            alpha_factor = alpha[:, :, np.newaxis] / 255.0
            img = (img_rgb * alpha_factor + bg * (1 - alpha_factor)).astype(np.uint8)
        
        # Ensure RGB
        if len(img.shape) == 2: # Gray
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) # Convert to BGR for standard processing checks
            # Wait,SmilingWolf models expect BGR?
            # Standard preprocessing for these models usually involves:
            # 1. Resize so max dim is 448
            # 2. Pad to 448x448
            # 3. BGR format
        
        # Resize logic matching typical WD14 inference scripts
        h, w = img.shape[:2]
        size = max(h, w)
        pad_h = (size - h) // 2
        pad_w = (size - w) // 2
        
        # Pad to square
        padded = np.ones((size, size, 3), dtype=np.uint8) * 255
        padded[pad_h:pad_h+h, pad_w:pad_w+w] = img
        
        # Target size is fixed for WD14 v3 models (ONNX export is fixed resolution)
        target_size = 448
        
        # Resize to target
        if size != target_size:
            resized = cv2.resize(padded, (target_size, target_size), interpolation=cv2.INTER_CUBIC)
        else:
            resized = padded
            
        # Normalize
        resized = resized.astype(np.float32)
        # Expected shape (1, 448, 448, 3) or (1, 3, 448, 448) ? 
        # Standard WD14 ONNX models take (Batch, Height, Width, Channel) -> (B, 448, 448, 3)
        # using BGR channel order usually.
        
        return resized
    
    def _run_inference(self, images: List[Image.Image], prompt: List[str], args: Dict[str, Any]) -> List[str]:
        """Run WD14 tagging on images."""
        batch_input = []
        
        for image in images:
            processed = self._preprocess_image(image, args)
            batch_input.append(processed)
            
        # Stack batch: (B, 448, 448, 3)
        input_tensor = np.stack(batch_input, axis=0)
        
        # Run inference
        probs = self.model.run([self.label_name], {self.input_name: input_tensor})[0]
        
        results = []
        for i in range(len(images)):
            prob_vector = probs[i]
            
            # Get dict of Tag -> Prob
            # Tags are likely: Rating tags first (4), then general/character tags
            # We skip rating indices [0,1,2,3]
            
            found_tags = []
            # We iterate from 4 onwards (skipping ratings)
            threshold = args.get('threshold', 0.35)
            
            for idx in range(4, len(self.tags)):
                if idx < len(prob_vector) and prob_vector[idx] > threshold:
                    found_tags.append((self.tags[idx], prob_vector[idx]))
            
            # Sort by confidence (descending)
            found_tags.sort(key=lambda x: x[1], reverse=True)
            
            # Build string
            tag_names = [t[0] for t in found_tags]
            
            # Additional cleanup: replace underscores with spaces
            tag_names = [t.replace('_', ' ') for t in tag_names]
            
            results.append(", ".join(tag_names))
            
        return results
    
    def unload(self):
        """Free model resources using shared utility."""
        from src.core.model_utils import unload_model, UnloadMode
        unload_model(self.model, None, UnloadMode.ONNX)
        self.model = None
        self.tags = []

