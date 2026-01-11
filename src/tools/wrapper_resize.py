from PIL import Image
import logging

class ResizeTool:
    @staticmethod
    @staticmethod
    def resize_image_file(image_path: str, max_dimension: int, output_path: str = None, 
                          overwrite: bool = True) -> tuple[bool, str]:
        """
        Resizes an image. Returns (success, message).
        """
        import os
        try:
            # Check if output exists and overwrite is False
            target = output_path if output_path else image_path
            if not overwrite and os.path.exists(target) and target != image_path:
                return False, "Skipped (Exists)"

            with Image.open(image_path) as img:
                width, height = img.size
                if max(width, height) <= max_dimension:
                    # If we are saving to a new location, we still need to save even if not resized
                    if output_path and output_path != image_path:
                        img.save(target, quality=95)
                        return True, "Copied (No Resize)"
                    return False, "Skipped (Small Enough)"

                # Calculate new size
                if width > height:
                    new_width = max_dimension
                    new_height = int(height * (max_dimension / width))
                else:
                    new_height = max_dimension
                    new_width = int(width * (max_dimension / height))

                img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
                
                img.save(target, quality=95)
                return True, "Resized"
        except Exception as e:
            logging.error(f"Failed to resize {image_path}: {e}")
            return False, str(e)

    @staticmethod
    def apply_to_dataset(dataset, max_dim: int, output_dir: str = None, 
                         prefix: str = "", suffix: str = "", extension: str = ".jpg",
                         overwrite: bool = True):
        from pathlib import Path
        import os
        
        count = 0
        skipped = 0
        errors = 0
        
        # Prepare output directory
        out_path_obj = Path(output_dir) if output_dir else None
        if out_path_obj and not out_path_obj.exists():
            out_path_obj.mkdir(parents=True, exist_ok=True)
            
        for img_obj in dataset.images:
            src_path = Path(img_obj.path)
            
            # Determine output filename
            # 1. Base name (with or without extension?)
            # Usually users expect name + suffix + extension
            base_name = src_path.stem
            
            # Ensure extension starts with dot
            if extension and not extension.startswith('.'):
                ext = '.' + extension
            else:
                ext = extension or src_path.suffix
                
            new_name = f"{prefix}{base_name}{suffix}{ext}"
            
            if out_path_obj:
                target_path = str(out_path_obj / new_name)
            else:
                # In-place or same dir
                target_path = str(src_path.with_name(new_name))
            
            success, msg = ResizeTool.resize_image_file(str(src_path), max_dim, target_path, overwrite)
            
            if success:
                count += 1
            elif "Skipped" in msg:
                skipped += 1
            else:
                errors += 1
                
        return f"Processed {len(dataset)} images. Resized/Saved: {count}, Skipped: {skipped}, Errors: {errors}"
