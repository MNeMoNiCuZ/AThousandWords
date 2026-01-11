from abc import ABC, abstractmethod
from typing import Dict, Any, List
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import time
import torch

# Import console kit
from src.core.console_kit import console, Fore, Style
import gc

# Import feature processing
from src.features import get_feature
from src.features import (
    CleanTextFeature, 
    CollapseNewlinesFeature,
    NormalizeTextFeature,
    RemoveChineseFeature,
    StripLoopFeature,
    StripThinkingTagsFeature,
    resize_image_proportionally
)


class BaseCaptionModel(ABC):
    """
    Base class for all caption model wrappers.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model = None
        self.prompt_presets = config.get("prompt_presets", {})
    
    @abstractmethod
    def _load_model(self):
        pass
    
    @abstractmethod
    def _run_inference(self, images: List[Image.Image], prompt: List[str], args: Dict[str, Any]) -> List[str]:
        pass
    
    def unload(self):
        pass
    
    # =========================================================================
    # CONSOLE OUTPUT HELPERS
    # =========================================================================
    
    def _print_header(self, text: str, color=None, char: str = "=", width: int = 60, force=False):
        """Print a centered header with border."""
        console.header(text, char=char, width=width, color=color, force=force)
    
    def _print_section(self, title: str, color=None, force=False):
        """Print a colored section header."""
        console.section(title, color=color, force=force)
    
    def _print_item(self, label: str, value: str = "", color=None, force=False):
        """Print a labeled item."""
        console.item(label, value, color=color, force=force)
    
    # =========================================================================
    # MAIN ORCHESTRATION
    # =========================================================================
    
    def run(self, dataset, args: Dict[str, Any]) -> List[str]:
        model_name = self.__class__.__name__.replace("Wrapper", "")
        
        # Initialize console verbosity
        print_console = get_feature("print_console").validate(args.get('print_console', True))
        console.set_verbose(print_console)
        
        # Performance Tracking
        start_time = time.time()
        start_background_vram = 0
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            try:
                # Measure "System/Idle" usage that PyTorch doesn't track directly as its own
                # Total Used - PyTorch Reserved = Background (OS, Chrome, etc)
                free_mem, total_mem = torch.cuda.mem_get_info()
                current_used = total_mem - free_mem
                current_reserved = torch.cuda.memory_reserved()
                start_background_vram = max(0, current_used - current_reserved)
            except Exception:
                start_background_vram = 0
        
        # Header is ALWAYS forced
        self._print_header(f"INITIALIZING {model_name.upper()}", Fore.WHITE, force=True)
        
        # Feature validation
        
        batch_size = get_feature("batch_size").validate(args.get('batch_size', 1))
        max_tokens = get_feature("max_tokens").validate(args.get('max_tokens', 500))
        overwrite = get_feature("overwrite").validate(args.get('overwrite', True))
        
        # Output options
        output_dir = args.get('output_dir', '')
        output_format = args.get('output_format', '.txt')
        if output_format and not output_format.startswith('.'):
            output_format = '.' + output_format
            
        # Determine output path base
        if output_dir and output_dir.strip():
            out_path_base = Path(output_dir)
            out_path_base.mkdir(parents=True, exist_ok=True)
        else:
            out_path_base = None

        # Filter images to process
        images_to_process = []
        skip_count = 0
        total_images = len(dataset.images)
        
        input_root = args.get('input_root')
        
        for img_obj in dataset.images:
            out_file = self._get_output_path(img_obj.path, out_path_base, output_format, input_root)
            if overwrite or not out_file.exists():
                images_to_process.append(img_obj)
            else:
                skip_count += 1
        
        files_to_process = len(images_to_process)
        batch_count = (files_to_process + batch_size - 1) // batch_size if files_to_process > 0 else 0
        
        # EARLY EXIT if no images to process
        if files_to_process == 0:
            console.print(f"All {total_images} images already captioned. Skipping model load.", color=Fore.YELLOW, force=True)
            return []

        # System
        unload_model = args.get('unload_model', True)
        
        generated_files = []
        processed_count = 0
        empty_count = 0  # Track empty captions
        peak_vram_gb = 0
        pbar = None
        
        try:
            # Load model ONLY if we have work to do
            if self.model is None:
                self.current_args = args
                self._load_model()
            
            # Text processing features
            clean_text = get_feature("clean_text").validate(args.get('clean_text', True))
            collapse_newlines = get_feature("collapse_newlines").validate(args.get('collapse_newlines', True))
            normalize_text = get_feature("normalize_text").validate(args.get('normalize_text', True))
            remove_chinese = get_feature("remove_chinese").validate(args.get('remove_chinese', False))
            strip_loop = get_feature("strip_loop").validate(args.get('strip_loop', True))
            strip_thinking_tags = get_feature("strip_thinking_tags").validate(args.get('strip_thinking_tags', True))
            
            # Prefix/Suffix
            prefix = get_feature("prefix").validate(args.get('prefix', ''))
            suffix = get_feature("suffix").validate(args.get('suffix', ''))
            
            # Image resize - handle both int and string inputs
            max_width = args.get('max_width')
            max_height = args.get('max_height')
            try:
                max_width = int(max_width) if max_width else None
                if max_width and max_width <= 0:
                    max_width = None
            except (ValueError, TypeError):
                max_width = None
            try:
                max_height = int(max_height) if max_height else None
                if max_height and max_height <= 0:
                    max_height = None
            except (ValueError, TypeError):
                max_height = None
            
            # Prompt Source System - Get mode and settings
            from src.features import get_prompt_for_image
            
            prompt_source_mode = args.get('prompt_source', 'Prompt Presets')
            prompt_prefix = args.get('prompt_prefix', '')
            prompt_suffix = args.get('prompt_suffix', '')
            prompt_file_extension = args.get('prompt_file_extension', 'prompt')
            if not prompt_file_extension.startswith('.'):
                prompt_file_extension = '.' + prompt_file_extension
            prompt_presets = args.get('prompt_presets', '')
            task_prompt = args.get('task_prompt', '')
            
            # Fallback to defaults if task_prompt is empty
            config_defaults = self.config.get('defaults', {})
            default_task_prompt = config_defaults.get('task_prompt', 'Describe this image.') or ''
            
            if not task_prompt:
                task_prompt = default_task_prompt
            
            # For "Prompt Presets" mode, resolve prompt once before batch loop
            base_prompt = None
            if prompt_source_mode == "Prompt Presets":
                # Smart conflict resolution:
                # If task_prompt differs from the default, assume user Intent overrides the Template.
                # This fixes CLI issues where --task-prompt is ignored because --prompt_presets defaults to "Detailed"
                effective_template = prompt_presets
                
                if task_prompt != default_task_prompt:
                    # User provided a custom prompt (or loaded one that differs from default)
                    
                    # Check if model supports custom prompts
                    if self.config.get('supports_custom_prompts', True) is False:
                        # Custom prompts NOT supported. Ignore user custom prompt.
                        # effective_template remains as prompt_presets (which forces use of preset value)
                        pass
                    else:
                        # We should prioritize this over the template preset
                        effective_template = ""
                    
                base_prompt = get_prompt_for_image(
                    image_path="",  # Not used in this mode
                    mode=prompt_source_mode,
                    prompt_preset_name=effective_template,
                    task_prompt=task_prompt,
                    presets_map=self.prompt_presets
                )
            
            # File statistics - ALWAYS forced
            self._print_header("FILE STATISTICS", Fore.WHITE, "-", force=True)
            console.print(f"  Total images: {total_images}", force=True)
            if not overwrite and skip_count > 0:
                console.print(f"  Skipping (exists): {skip_count}", force=True)
            console.print(f"  To process: {files_to_process}", force=True)
            console.print(f"  Batches: {batch_count} (size {batch_size})", force=True)
            console.print("", force=True)
            
            # Process in batches
            pbar = tqdm(total=files_to_process, desc="Processing", unit="img")
            
            for batch_idx in range(0, files_to_process, batch_size):
                batch = images_to_process[batch_idx:batch_idx + batch_size]
                current_batch_num = (batch_idx // batch_size) + 1
                
                # Get filenames for this batch
                batch_filenames = [img.path.name for img in batch]
                
                # =====================================================
                # PRE-PROCESSING SECTION (CYAN) - ONE per batch
                # =====================================================
                filenames_str = ", ".join(batch_filenames)
                self._print_section(f"PRE-PROCESSING [Batch {current_batch_num}/{batch_count}] {filenames_str}", Fore.CYAN)
                
                batch_images = []
                valid_batch = []
                
                for img_obj in batch:
                    img_name = img_obj.path.name
                    
                    # SAFETY CHECK: Ensure file exists before attempting any processing
                    if not img_obj.path.exists():
                        console.print(f"  ❌ File not found: {img_name}", color=Fore.RED, force=True)
                        continue
                        
                    try:
                        # Check if this is a video
                        if img_obj.is_video():
                            # Check if model supports video
                            supported_media_types = self.config.get('media_type', ['Image'])
                            if 'Video' in supported_media_types:
                                # Model supports video - pass Path object directly
                                batch_images.append(img_obj.path)
                                valid_batch.append(img_obj)
                                self._print_item("Video", f"{img_name} (will be processed by model)", Fore.CYAN)
                            else:
                                # Model doesn't support video - extract first frame
                                self._print_item("Video→Image", f"{img_name} (extracting first frame for image-only model)", Fore.YELLOW)
                                image_or_path = img_obj._extract_video_thumbnail()
                                
                                if image_or_path is None:
                                    raise ValueError(f"Failed to extract frame from video {img_name}")
                                
                                # CRITICAL FIX: Ensure we have a PIL Image, not a path string
                                if isinstance(image_or_path, (str, Path)):
                                    image = Image.open(image_or_path).convert("RGB")
                                else:
                                    image = image_or_path
                                
                                orig_w, orig_h = image.size
                                
                                # Resize if needed
                                if max_width or max_height:
                                    image = resize_image_proportionally(image, max_width, max_height)
                                    new_w, new_h = image.size
                                    if (new_w, new_h) != (orig_w, orig_h):
                                        self._print_item("Resized", f"{img_name}: {orig_w}x{orig_h} to {new_w}x{new_h}", Fore.CYAN)
                                
                                batch_images.append(image)
                                valid_batch.append(img_obj)
                        else:
                            # For images, load and optionally resize
                            image = Image.open(img_obj.path).convert("RGB")
                            orig_w, orig_h = image.size
                            
                            # Resize if needed
                            if max_width or max_height:
                                image = resize_image_proportionally(image, max_width, max_height)
                                new_w, new_h = image.size
                                if (new_w, new_h) != (orig_w, orig_h):
                                    self._print_item("Resized", f"{img_name}: {orig_w}x{orig_h} to {new_w}x{new_h}", Fore.CYAN)
                            
                            batch_images.append(image)
                            valid_batch.append(img_obj)
                        
                    except Exception as e:
                         console.print(f"  Error loading {img_name}: {e}", color=Fore.RED, force=True)
                         # FATAL ERROR: Re-raise to abort processing
                         raise RuntimeError(f"Fatal error processing {img_name}: {e}") from e
                
                if not batch_images:
                    continue
                
                # Check for mixed media types (videos + images)
                # Videos are Path objects, images are PIL Images
                has_videos = any(isinstance(item, (str, Path)) and str(item).lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.webm')) for item in batch_images)
                has_images = any(not (isinstance(item, (str, Path)) and str(item).lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.webm'))) for item in batch_images)
                
                if has_videos and has_images:
                    console.print("  ⚠ Warning: Mixed media types detected (videos + images). Converting videos to images for processing.", color=Fore.YELLOW, force=True)
                    # Convert video paths to PIL Images (extract first frame)
                    converted_images = []
                    for item in batch_images:
                        is_video = isinstance(item, (str, Path)) and str(item).lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.webm'))
                        if is_video:
                            # Extract first frame from video
                            from src.core.dataset import MediaObject
                            img_obj = MediaObject(path=Path(item))
                            frame = img_obj._extract_video_thumbnail()
                            if frame is None:
                                console.print(f"  Failed to extract frame from {Path(item).name}, skipping", color=Fore.RED, force=True)
                                continue
                            # Ensure it's a PIL Image
                            if isinstance(frame, (str, Path)):
                                frame = Image.open(frame).convert("RGB")
                            converted_images.append(frame)
                        else:
                            converted_images.append(item)
                    
                    batch_images = converted_images
                    
                    if not batch_images:
                        console.print("  No media remaining after conversion. Skipping batch.", color=Fore.YELLOW, force=True)
                        continue
                elif has_videos and not has_images:
                     # Only videos detected AND model supports them (since they are in batch_images as Paths)
                     console.print("  Only videos detected, captioning videos", color=Fore.YELLOW, force=True)
                
                
                # =====================================================
                # INFERENCE SECTION (YELLOW)
                # =====================================================
                self._print_section(f"INFERENCE [Batch {current_batch_num}/{batch_count}]", Fore.YELLOW)
                self._print_item("Images", str(len(batch_images)), Fore.YELLOW)
                
                # Resolve prompt based on mode
                prompt_list = []
                
                if prompt_source_mode == "Prompt Presets":
                    # Optimization: Use the pre-resolved base_prompt for all images
                    # This ensures identical prompts for the whole batch
                    final_prompt = base_prompt or task_prompt
                    prompt_list = [final_prompt] * len(batch_images)
                else:
                    # For File/Metadata modes, we need per-image prompts
                    # We must validate EACH image to ensure we have a prompt
                    # If strict mode (implied by "From File"/"From Metadata"), we skip images without prompts
                    
                    filtered_batch_images = []
                    filtered_valid_batch = []
                    
                    for i, (img_item, img_obj) in enumerate(zip(batch_images, valid_batch)):
                        try:
                            # Use strict=True to get None if missing
                            p = get_prompt_for_image(
                                image_path=str(img_obj.path),
                                mode=prompt_source_mode,
                                prefix=prompt_prefix,
                                extension=prompt_file_extension,
                                suffix=prompt_suffix,
                                prompt_preset_name=prompt_presets,
                                task_prompt=task_prompt,
                                presets_map=self.prompt_presets,
                                strict=True 
                            )
                            
                            if p is None:
                                console.print(f"  Skipping {img_obj.path.name}: No prompt found in {prompt_source_mode} mode", color=Fore.RED, force=True)
                                # Increment skip count? Ideally yes, but we are inside batch loop. 
                                # Just log for now.
                                continue

                            # DEBUG: Always print resolved prompt for custom sources
                            # Print FULL prompt - no truncation, to verify content
                            self._print_item("Prompt", f"{p}", Fore.CYAN)
                                
                            prompt_list.append(p)
                            filtered_batch_images.append(img_item)
                            filtered_valid_batch.append(img_obj)
                            
                        except Exception as ex:
                            console.print(f"  Error resolving prompt for {img_obj.path.name}: {ex}", color=Fore.RED, force=True)
                            continue
                    
                    # Update the batch lists to only contain items with valid prompts
                    batch_images = filtered_batch_images
                    valid_batch = filtered_valid_batch
                    
                    if not batch_images:
                        console.print("  No valid images left in batch after prompt resolution. Skipping batch.", color=Fore.YELLOW)
                        continue

                # Pass everything to inference
                # CRITICAL CHANGE: Now passing List[str] instead of str
                raw_captions = self._run_inference(batch_images, prompt_list, args)
                
                
                # Print raw captions in inference section
                # ONLY if verbose
                if console.verbose:
                    for img_obj, raw_cap in zip(valid_batch, raw_captions):
                        console.print(f"\n  Raw Caption ({img_obj.path.name}):", color=Fore.YELLOW)
                        console.print(f"  {raw_cap}")
                
                # =====================================================
                # POST-PROCESSING SECTION (GREEN) - ONE per batch
                # =====================================================
                self._print_section(f"POST-PROCESSING [Batch {current_batch_num}/{batch_count}]", Fore.GREEN)
                
                for img_idx, (img_obj, caption) in enumerate(zip(valid_batch, raw_captions)):
                    img_name = img_obj.path.name
                    
                    console.print(f"\n  [{img_name}]", color=Fore.GREEN)
                    
                    final_cap = caption
                    
                    if strip_thinking_tags:
                        final_cap = StripThinkingTagsFeature.apply(final_cap)
                    
                    if remove_chinese:
                        final_cap = RemoveChineseFeature.apply(final_cap)
                    
                    # 1. Normalize Text (Strip Markdown/Bullets/Unicode) - PRESERVES NEWLINES
                    if normalize_text:
                        final_cap = NormalizeTextFeature.apply(final_cap)
                    
                    # 2. Collapse Newlines (Convert 'Item 1\nItem 2' -> 'Item 1. Item 2')
                    if collapse_newlines:
                        final_cap = CollapseNewlinesFeature.apply(final_cap)
                    
                    if strip_loop:
                        final_cap = StripLoopFeature.apply(final_cap)
                        
                    # 3. Clean Text (Final whitespace polish)
                    if clean_text:
                        final_cap = CleanTextFeature.apply(final_cap)
                    
                    # Apply prefix/suffix
                    if prefix or suffix:
                        final_cap = f"{prefix}{final_cap}{suffix}"
                        self._print_item("Added", f"prefix/suffix", Fore.GREEN)
                        
                    # CHECK FOR EMPTY CAPTION
                    if not final_cap or not final_cap.strip():
                        empty_count += 1
                        console.print(f"  ⚠️ Warning: Generated caption is empty for {img_name}", color=Fore.RED)
                    
                    # Print processed caption (FULL, not truncated)
                    # ONLY if verbose
                    if console.verbose:
                        console.print(f"  Processed Caption:", color=Fore.GREEN)
                        console.print(f"  {final_cap}")
                    
                    # Save caption
                    out_file = self._get_output_path(img_obj.path, out_path_base, output_format, input_root)
                    try:
                        out_file.parent.mkdir(parents=True, exist_ok=True)
                        with open(out_file, 'w', encoding='utf-8') as f:
                            f.write(final_cap)
                        
                        self._print_item("Saved", str(out_file.absolute()), Fore.GREEN)
                        img_obj.update_caption(final_cap)
                        generated_files.append(str(out_file.absolute()))
                        
                    except Exception as e:
                        console.print(f"  Save failed: {e}", color=Fore.RED, force=True)
                    
                    pbar.update(1)
                
                processed_count += len(valid_batch)
        
        except Exception as e:
            if pbar: 
                pbar.close()
            
            # Enhanced OOM detection and handling
            is_oom = "out of memory" in str(e).lower()
            
            if is_oom and torch.cuda.is_available():
                # Simple, concise OOM message in light red
                console.print("\n🔴 CUDA OUT OF MEMORY - Reduce batch size, resize images in general settings, use another model, or close other GPU intensive processes to free up memory", color=Fore.LIGHTRED_EX, force=True)
                console.print("Clearing CUDA cache...", color=Fore.YELLOW, force=True)
                torch.cuda.empty_cache()
                console.print("Cache cleared.\n", color=Fore.GREEN, force=True)
                
                # Re-raise with GUI-friendly message (no traceback spam)
                raise RuntimeError(
                    "🔴 CUDA OUT OF MEMORY - Reduce batch size, resize images in general settings, use another model, or close other GPU intensive processes to free up memory"
                ) from None  # from None suppresses the original traceback
            
            # Non-OOM error - show full traceback
            console.error(f"Inference error: {e}")
            import traceback
            traceback.print_exc()
            
            # Re-raise for GUI
            raise
            
        finally:
            if pbar:
                pbar.close()
            
            # Performance Stats (Calculate BEFORE final summary)
            end_time = time.time()
            elapsed = end_time - start_time
            
            if torch.cuda.is_available():
                peak_reserved = torch.cuda.max_memory_reserved()
                total_peak_bytes = start_background_vram + peak_reserved
                peak_vram_gb = total_peak_bytes / (1024 ** 3)

            # Final summary - ALWAYS forced
            console.print("", force=True)
            self._print_header(f"FINISHED PROCESSING: {model_name}", Fore.WHITE, force=True)
            console.print(f"  Processed: {processed_count}", color=Fore.GREEN, force=True)
            console.print(f"  Skipped:   {skip_count}", color=Fore.YELLOW, force=True)
            console.print(f"  Total:     {total_images}", force=True)
            if empty_count > 0:
                 console.print(f"  Empty:     {empty_count}", color=Fore.RED, force=True)
            console.print("", force=True)
            console.print(f"  Time Taken: {elapsed:.2f}s", color=Fore.CYAN, force=True)
            if peak_vram_gb > 0:
                console.print(f"  Peak VRAM:  {peak_vram_gb:.2f} GB", color=Fore.CYAN, force=True)
            console.print("", force=True)
            
            # Unload model if requested (Moved to finally to ensure cleanup)
            if unload_model:
                console.print("Unloading model...", color=Fore.YELLOW, force=True)
                self.unload()
                if torch.cuda.is_available():
                    gc.collect()
                    torch.cuda.empty_cache()
                console.print("Model unloaded and cache cleared.", color=Fore.GREEN, force=True)

        return generated_files, {
            "processed": processed_count,
            "skipped": skip_count,
            "total": total_images,
            "time": elapsed,
            "peak_vram": peak_vram_gb,
            "model_name": model_name,
            "empty_count": empty_count
        }
    
    def _get_output_path(self, input_path: Path, out_path_base: Path, output_format: str, input_root: Path = None) -> Path:
        """
        Generate output path for caption file.
        
        Strips UUID prefixes from filenames to use clean original names.
        """
        import re
        
        # Helper function to strip UUID prefix from filename
        def strip_uuid_prefix(filename):
            """Strip UUID prefix from filename if present."""
            uuid_pattern = re.compile(r'^[0-9a-f]{32}_(.+)$')
            match = uuid_pattern.match(filename)
            if match:
                return match.group(1)  # Return filename without UUID prefix
            return filename  # No prefix, return as-is
        
        if out_path_base:
            # Get clean filename without UUID prefix
            clean_filename = strip_uuid_prefix(input_path.name)
            rel_path = Path(clean_filename)
            
            # Try to preserve structure if input_root is provided
            if input_root:
                try:
                    # Get relative path and clean the final filename
                    rel_path_from_root = input_path.relative_to(input_root)
                    
                    # Check if the file is directly in the root (no subdirectories)
                    if rel_path_from_root.parent == Path('.'):
                        # File is directly in input_root, use clean filename only
                        rel_path = Path(clean_filename)
                    else:
                        # File is in a subdirectory, preserve structure
                        rel_path = rel_path_from_root.parent / clean_filename
                except ValueError:
                    # Fallback to flattening if relative path fails (different drive etc)
                    rel_path = Path(clean_filename)
            else:
                try:
                    rel_path_from_input = input_path.relative_to(Path("input"))
                    
                    # Check if the file is directly in the root (no subdirectories)
                    if rel_path_from_input.parent == Path('.'):
                        # File is directly in input folder, use clean filename only
                        rel_path = Path(clean_filename)
                    else:
                        # File is in a subdirectory, preserve structure
                        rel_path = rel_path_from_input.parent / clean_filename
                except ValueError:
                    rel_path = Path(clean_filename)
                    
            return out_path_base / rel_path.with_suffix(output_format)
        
        # No output directory specified - save alongside input with clean name
        clean_filename = strip_uuid_prefix(input_path.name)
        return input_path.parent / Path(clean_filename).with_suffix(output_format)

