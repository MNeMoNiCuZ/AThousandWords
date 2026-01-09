from PIL import Image
import logging

class ResizeTool:
    @staticmethod
    def resize_image_file(image_path: str, max_dimension: int, output_path: str = None) -> bool:
        """
        Resizes an image so that its longest side does not exceed max_dimension.
        If output_path is None, overwrites the original.
        """
        try:
            with Image.open(image_path) as img:
                width, height = img.size
                if max(width, height) <= max_dimension:
                    return False # No resize needed

                # Calculate new size
                if width > height:
                    new_width = max_dimension
                    new_height = int(height * (max_dimension / width))
                else:
                    new_height = max_dimension
                    new_width = int(width * (max_dimension / height))

                img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
                
                target = output_path if output_path else image_path
                img.save(target, quality=95)
                return True
        except Exception as e:
            logging.error(f"Failed to resize {image_path}: {e}")
            return False

    @staticmethod
    def apply_to_dataset(dataset, max_dim: int):
        count = 0
        for img_obj in dataset.images:
            if ResizeTool.resize_image_file(str(img_obj.path), max_dim):
                count += 1
        return f"Resized {count} images."
