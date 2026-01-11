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
    
    # Runtime cache for video thumbnail (not persisted to disk across sessions, but kept for app lifetime)
    _temp_thumbnail_path: Optional[str] = field(default=None, init=False, repr=False)
    
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
                thumb_path = self.get_thumbnail_path()
                if thumb_path:
                    return Image.open(thumb_path).convert("RGB")
                return None
            else:
                return Image.open(self.path).convert("RGB")
        except Exception as e:
            self.error = str(e)
            return None
    
    def get_thumbnail_path(self) -> Optional[str]:
        """
        Get path to a valid thumbnail image.
        For images, returns the file path.
        For videos, generates a temp thumbnail if one doesn't exist, and returns that path.
        """
        if not self.is_video():
            return str(self.path)
            
        # Check if we already have a generated temp thumbnail
        if self._temp_thumbnail_path and os.path.exists(self._temp_thumbnail_path):
            return self._temp_thumbnail_path
            
        # Generate new one
        thumb_path = self._extract_video_thumbnail()
        if thumb_path:
            self._temp_thumbnail_path = thumb_path
            return thumb_path
            
        return None
    
    def _extract_video_thumbnail(self) -> Optional[str]:
        """Extract first frame from video as thumbnail using ffmpeg to a temp file."""
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
                # Successfully created
                # --- MODIFY: Add Play Overlay ---
                try:
                    from PIL import ImageDraw
                    with Image.open(tmp_path) as img:
                        # Convert to RGB to ensure drawing works
                        img = img.convert("RGB")
                        draw = ImageDraw.Draw(img)
                        w, h = img.size
                        
                        # Calculate center
                        cx, cy = w // 2, h // 2
                        
                        # Size of triangle relative to image
                        # Use 2.5 divisor (approx 2x larger than previous // 5)
                        s = min(w, h) // 2
                        
                        # Points for triangle (pointing right)
                        # Center is (cx, cy)
                        # Left-top: (cx - s/2, cy - s/2)
                        # Left-bottom: (cx - s/2, cy + s/2)
                        # Right-middle: (cx + s/2, cy)
                        # Adjust to make it look centered visually
                        
                        p1 = (cx - s//2, cy - s//2)
                        p2 = (cx - s//2, cy + s//2)
                        p3 = (cx + s//1.5, cy)
                        
                        # Draw semi-transparent triangle
                        # PIL doesn't support alpha on RGB easily without composition
                        # So we draw a solid white triangle for now, maybe with outline
                        
                        # To do transparency, we need RGBA
                        overlay = Image.new('RGBA', img.size, (0, 0, 0, 0))
                        draw_overlay = ImageDraw.Draw(overlay)
                        
                        # Draw semi-transparent white triangle
                        # Reduced opacity by 50% (180->90, 100->50)
                        draw_overlay.polygon([p1, p2, p3], fill=(255, 255, 255, 90), outline=(0, 0, 0, 50))
                        
                        # Composite
                        img = Image.alpha_composite(img.convert("RGBA"), overlay)
                        
                        # Save back as JPG (no alpha)
                        img.convert("RGB").save(tmp_path, "JPEG", quality=85)
                        
                except Exception as e:
                    # If overlay fails, we still return the original thumbnail
                    print(f"Failed to draw overlay: {e}")
                    pass
                
                return tmp_path
            else:
                # Failed
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
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
