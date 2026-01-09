from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, List, Optional
from PIL import Image
import subprocess
import tempfile
import os

# Video extensions for detection
VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}

@dataclass
class MediaObject:
    path: Path
    original_size: tuple = (0, 0)
    metadata: Dict[str, Any] = field(default_factory=dict)
    caption: str = ""
    error: Optional[str] = None
    media_type: str = "image"  # "image" or "video"
    
    # Track changes
    _modified: bool = False
    
    def __post_init__(self):
        """Automatically detect media type based on file extension."""
        if self.path.suffix.lower() in VIDEO_EXTENSIONS:
            self.media_type = "video"
        else:
            self.media_type = "image"

    def is_video(self) -> bool:
        """Check if this media object is a video."""
        return self.media_type == "video"

    def load_image(self) -> Optional[Image.Image]:
        """Load image or generate video thumbnail."""
        try:
            if self.is_video():
                return self._extract_video_thumbnail()
            else:
                return Image.open(self.path).convert("RGB")
        except Exception as e:
            self.error = str(e)
            return None
    
    def _extract_video_thumbnail(self) -> Optional[Image.Image]:
        """Extract first frame from video as thumbnail using ffmpeg."""
        try:
            # Create temporary file for thumbnail
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
                tmp_path = tmp.name
            
            # Use ffmpeg to extract first frame
            cmd = [
                'ffmpeg',
                '-i', str(self.path),
                '-vframes', '1',
                '-f', 'image2',
                '-y',  # Overwrite output file
                tmp_path
            ]
            
            result = subprocess.run(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=5
            )
            
            if result.returncode == 0 and os.path.exists(tmp_path):
                img = Image.open(tmp_path).convert("RGB")
                os.unlink(tmp_path)
                return img
            else:
                # Fallback: return a placeholder if ffmpeg fails
                os.unlink(tmp_path) if os.path.exists(tmp_path) else None
                return None
                
        except Exception as e:
            self.error = f"Video thumbnail extraction failed: {e}"
            return None

    def update_caption(self, new_caption: str):
        if self.caption != new_caption:
            self.caption = new_caption
            self._modified = True

    def save_caption(self, extension: str = ".txt", output_dir: Optional[Path] = None):
        """Saves the caption to a text file."""
        if not self.caption and not self.error:
            return # Don't save empty if valid
            
        target_dir = output_dir if output_dir else self.path.parent
        if not target_dir.exists():
            target_dir.mkdir(parents=True, exist_ok=True)
            
        target_path = target_dir / self.path.with_suffix(extension).name
        try:
            with open(target_path, 'w', encoding='utf-8') as f:
                f.write(self.caption)
        except Exception as e:
            self.error = f"Failed to save caption: {e}"

class Dataset:
    def __init__(self, images: List[MediaObject] = None):
        self.images = images or []  # Name kept as 'images' for backward compatibility

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx]
        
    def add(self, media: MediaObject):
        self.images.append(media)
        
    def get_paths(self) -> List[Path]:
        return [img.path for img in self.images]
