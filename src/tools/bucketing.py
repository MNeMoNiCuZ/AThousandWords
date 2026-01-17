"""
Resolution Bucketing Tool

Analyzes and organizes images by aspect ratio buckets for ML training.
Supports automatic detection and manual ratio specification.
"""

import logging
import shutil
import gradio as gr
from pathlib import Path
from math import gcd
from collections import defaultdict
from PIL import Image

from .base import BaseTool, ToolConfig


class BucketingTool(BaseTool):
    """Tool for analyzing and organizing images by aspect ratio buckets."""
    
    COLORS = {
        "bucket": "#4CAF50",
        "ok": "#8BC34A", 
        "partial": "#FFC107",
        "outlier": "#ef5350",
        "filtered": "#78909C",
        "resolution": "#FF9800"
    }
    
    @property
    def config(self) -> ToolConfig:
        """
        Provide the ToolConfig metadata for the Bucketing tool used in the UI.
        
        Returns:
            ToolConfig: Configuration for the tool with name "bucketing", display name "Bucketing", a short description about resolution bucketing, and an icon placeholder.
        """
        return ToolConfig(
            name="bucketing",
            display_name="Bucketing",
            description="### Resolution Bucketing\nBalance your dataset by distributing images across aspect ratio buckets.",
            icon=""
        )
    
    def apply_to_dataset(self, dataset, num_buckets: int = 3,
                         tolerance: float = 20.0,
                         max_per_bucket: int = 50,
                         manual_buckets: str = "",
                         create_landscape: bool = True,
                         create_portrait: bool = True,
                         create_square: bool = True,
                         min_resolution: int = 512,
                         max_resolution: int = 2048,
                         unassigned_action: str = "Include in _unassigned",
                         file_action: str = "Copy",
                         output_dir: str = "",
                         action: str = "analyze") -> str:
        
        """
                         Analyze a dataset of images for aspect-ratio bucketing and optionally organize or prune the images.
                         
                         Performs aspect-ratio analysis using either automatic bucketing or user-specified manual ratios, assigns images to buckets respecting tolerance and per-bucket limits, and generates an HTML report. Depending on `action`, the function will:
                         - "analyze": return an HTML report only,
                         - "organize": copy or move files into per-bucket folders under `output_dir` according to `file_action`,
                         - "prune": balance buckets by moving excess images into the result's unassigned list and return a report.
                         
                         Parameters:
                             dataset: Dataset-like object containing an iterable `images` attribute; each image entry should expose a filesystem path.
                             num_buckets (int): Target number of buckets when using automatic bucketing.
                             tolerance (float): Allowed percentage difference between an image's aspect ratio and a bucket's ratio for eligibility (e.g., 20.0 means ±20% tolerance).
                             max_per_bucket (int): Maximum number of images to place in any single bucket.
                             manual_buckets (str): Newline- or comma-separated manual ratio definitions in the form "W:H" (e.g., "16:9,4:3"); empty means automatic bucketing.
                             create_landscape (bool): Include Landscape-orientation buckets when auto-generating buckets.
                             create_portrait (bool): Include Portrait-orientation buckets when auto-generating buckets.
                             create_square (bool): Include Square-orientation buckets when auto-generating buckets.
                             min_resolution (int): Minimum allowed image maximum dimension (width or height) in pixels; images below this are flagged as too small.
                             max_resolution (int): Maximum allowed image maximum dimension (width or height) in pixels; images above this are flagged as too large.
                             unassigned_action (str): Behavior label for handling unassigned images in organization preview or when organizing (e.g., "Include in _unassigned").
                             file_action (str): File operation to perform when organizing; expected values include "Copy" or "Move".
                             output_dir (str): Destination directory used when `action` is "organize"; if empty, organization is not performed.
                             action (str): One of "analyze", "organize", or "prune" determining the operation performed.
                         
                         Returns:
                             str: HTML report summarizing analysis results, bucket summaries, ratio breakdowns, and organize/prune previews. When `action` is "organize", files may be copied or moved into `output_dir` as a side effect.
                         """
                         if not dataset or not dataset.images:
            gr.Warning("No images loaded.")
            return self._empty_state()
        
        bucket_orientations = set()
        if create_landscape:
            bucket_orientations.add("Landscape")
        if create_portrait:
            bucket_orientations.add("Portrait")
        if create_square:
            bucket_orientations.add("Square")
        
        image_data = self._collect_image_data(dataset)
        if not image_data:
            gr.Warning("No valid images found.")
            return self._empty_state()
        
        use_manual = manual_buckets.strip() != ""
        
        result = self._analyze_buckets(
            image_data, num_buckets, tolerance, max_per_bucket,
            bucket_orientations, min_resolution, max_resolution,
            use_manual, manual_buckets
        )
        
        if action == "organize":
            if not output_dir:
                gr.Warning("Set output directory to organize.")
                return self._generate_report(result, output_dir, file_action, unassigned_action)
            return self._organize(result, output_dir, file_action, unassigned_action)
        
        if action == "prune":
            return self._prune_to_balanced(result, output_dir, file_action, unassigned_action)
        
        return self._generate_report(result, output_dir, file_action, unassigned_action)
    
    def _empty_state(self) -> str:
        """
        Return a minimal HTML message prompting the user to load images and start analysis.
        
        Returns:
            html (str): A small HTML fragment with a user-facing prompt ("Load images and click Analyze.").
        """
        return '<div style="padding:20px;color:#888;">Load images and click Analyze.</div>'
    
    def _get_image_size(self, path: Path) -> tuple:
        """
        Return the dimensions of an image file.
        
        Attempts to open the image at `path` and returns its size as a (width, height) tuple. Returns `None` if the file cannot be opened or read as an image.
        
        Parameters:
            path (Path): Path to the image file.
        
        Returns:
            tuple or None: `(width, height)` on success, `None` if the image is unreadable.
        """
        try:
            with Image.open(path) as img:
                return img.size
        except Exception:
            return None
    
    def _get_orientation(self, ratio: float) -> str:
        """
        Classify an aspect ratio as Landscape, Portrait, or Square.
        
        Parameters:
            ratio (float): The aspect ratio computed as width divided by height.
        
        Returns:
            str: "Landscape" if ratio > 1.1, "Portrait" if ratio < 0.9, otherwise "Square".
        """
        if ratio > 1.1:
            return "Landscape"
        elif ratio < 0.9:
            return "Portrait"
        return "Square"
    
    def _get_ratio_string(self, w: int, h: int) -> str:
        """
        Produce a reduced aspect-ratio string from width and height.
        
        Parameters:
            w (int): Image width in pixels.
            h (int): Image height in pixels.
        
        Returns:
            str: Aspect ratio in the form "W:G", where W and G are width and height divided by their greatest common divisor.
        """
        g = gcd(w, h)
        return f"{w//g}:{h//g}"
    
    def _ratio_diff_pct(self, r1: float, r2: float) -> float:
        """
        Compute the percentage difference of r1 relative to r2.
        
        Parameters:
        	r1 (float): First ratio (value to compare).
        	r2 (float): Reference ratio (denominator for percentage calculation).
        
        Returns:
        	percent_diff (float): Absolute percentage difference computed as abs(r1 - r2) / r2 * 100. Returns 100.0 if `r2` is zero.
        """
        if r2 == 0:
            return 100.0
        return abs(r1 - r2) / r2 * 100
    
    def _parse_manual_ratio(self, ratio_str: str) -> tuple:
        """
        Parse a manual aspect-ratio string of the form "W:H" into a normalized ratio string and its numeric value.
        
        Parameters:
            ratio_str (str): Aspect-ratio text expected as two positive integers separated by a colon (e.g., "3:2", "16:9"); surrounding whitespace is allowed.
        
        Returns:
            tuple or None: Tuple (ratio_str, ratio_value) where `ratio_str` is the trimmed "W:H" string and `ratio_value` is the numeric ratio W / H, or `None` if the input is invalid.
        """
        try:
            parts = ratio_str.strip().split(':')
            if len(parts) != 2:
                return None
            w, h = int(parts[0]), int(parts[1])
            if w <= 0 or h <= 0:
                return None
            ratio_val = w / h
            return (ratio_str.strip(), ratio_val)
        except:
            return None
    
    def _collect_image_data(self, dataset) -> list:
        """
        Collect readable image metadata from a dataset into a list of info dictionaries.
        
        Parameters:
            dataset: An object with an iterable `images` attribute; each image must expose a `path` (Path or str) to the image file.
        
        Returns:
            list: A list of dictionaries, one per successfully read image, each containing:
                - path (str): Full image path.
                - filename (str): Base filename.
                - width (int): Image width in pixels.
                - height (int): Image height in pixels.
                - max_dim (int): Larger of width and height.
                - ratio (float): Aspect ratio (width / height).
                - ratio_str (str): Reduced aspect ratio as "W:G".
                - orientation (str): One of "Landscape", "Portrait", or "Square" based on the ratio.
        """
        data = []
        for img_obj in dataset.images:
            size = self._get_image_size(img_obj.path)
            if size is None:
                continue
            w, h = size
            ratio = w / h if h > 0 else 1.0
            data.append({
                "path": str(img_obj.path),
                "filename": Path(img_obj.path).name,
                "width": w,
                "height": h,
                "max_dim": max(w, h),
                "ratio": ratio,
                "ratio_str": self._get_ratio_string(w, h),
                "orientation": self._get_orientation(ratio)
            })
        return data
    
    def _analyze_buckets(self, image_data: list, num_buckets: int,
                          tolerance: float, max_per_bucket: int,
                          bucket_orientations: set, min_res: int, max_res: int,
                          use_manual: bool, manual_buckets_str: str) -> dict:
        
        """
                          Analyze a collection of images and assign them into aspect-ratio buckets either from manual specifications or by selecting the most common ratios, applying tolerance and per-bucket limits.
                          
                          Parameters:
                              image_data (list): List of image metadata dicts. Each dict is expected to contain at least
                                  'path', 'filename', 'width', 'height', 'max_dim', 'ratio', 'ratio_str', and 'orientation'.
                              num_buckets (int): Desired number of buckets when selecting automatically.
                              tolerance (float): Maximum allowed percentage difference between an image ratio and a bucket ratio for eligibility.
                              max_per_bucket (int): Maximum number of images to place into each bucket (<= 0 means no limit).
                              bucket_orientations (set): Subset of {'Landscape', 'Portrait', 'Square'} to restrict which orientations are considered for bucket selection; empty set uses all orientations.
                              min_res (int): Minimum allowed largest image dimension; images below this are recorded as resolution_low.
                              max_res (int): Maximum allowed largest image dimension; images above this are recorded as resolution_high.
                              use_manual (bool): If True, parse and use manual_buckets_str to create buckets; otherwise create buckets automatically.
                              manual_buckets_str (str): Comma-separated manual ratios in the form "W:H" (e.g., "3:2,1:1").
                          
                          Returns:
                              dict: A result dictionary containing:
                                  - total (int): Total number of images processed.
                                  - buckets (dict): Mapping from bucket ratio_str to bucket info dict with keys:
                                      'name', 'ratio', 'ratio_str', 'orientation', 'representative', 'images' (assigned images list),
                                      'eligible' (list of tuples (img, diff, match_pct)), and flags 'manual' or 'virtual' when applicable.
                                  - resolution_low (list): Images with max_dim < min_res.
                                  - resolution_high (list): Images with max_dim > max_res.
                                  - outliers (list): Images whose best bucket match exceeds tolerance.
                                  - unassigned (list): Images not placed into buckets (but within tolerance relative to their best bucket).
                                  - ratio_summary (defaultdict): Per-ratio summary objects keyed by ratio_str; each contains
                                      'images' (list), 'status' (one of 'bucket', 'ok', 'partial', 'outlier'), 'bucket' (assigned bucket name or None), and 'match_pct' (int 0-100).
                          """
                          result = {
            "total": len(image_data),
            "buckets": {},
            "resolution_low": [],
            "resolution_high": [],
            "outliers": [],
            "unassigned": [],
            "ratio_summary": defaultdict(lambda: {"images": [], "status": None, "bucket": None, "match_pct": 0})
        }
        
        for img in image_data:
            if img["max_dim"] < min_res:
                result["resolution_low"].append(img)
            elif img["max_dim"] > max_res:
                result["resolution_high"].append(img)
            
            result["ratio_summary"][img["ratio_str"]]["images"].append(img)
        
        if use_manual:
            ratios = [r.strip() for r in manual_buckets_str.split(',') if r.strip()]
            for ratio_str in ratios:
                parsed = self._parse_manual_ratio(ratio_str)
                if parsed:
                    ratio_str, ratio_val = parsed
                    orient = self._get_orientation(ratio_val)
                    result["buckets"][ratio_str] = {
                        "name": ratio_str,
                        "ratio": ratio_val,
                        "ratio_str": ratio_str,
                        "orientation": orient,
                        "representative": f"{int(1024 * ratio_val)}x1024" if ratio_val >= 1 else f"1024x{int(1024 / ratio_val)}",
                        "images": [],
                        "eligible": [],
                        "manual": True
                    }
        else:
            if bucket_orientations:
                images_for_buckets = [img for img in image_data if img["orientation"] in bucket_orientations]
            else:
                images_for_buckets = image_data
            
            ratio_counts = defaultdict(list)
            for img in images_for_buckets:
                ratio_counts[(img["ratio_str"], img["ratio"], img["orientation"])].append(img)
            
            selected_buckets = []
            virtual_ratios = {"Landscape": ("3:2", 1.5), "Portrait": ("2:3", 0.667), "Square": ("1:1", 1.0)}
            
            if bucket_orientations:
                for orient in ["Landscape", "Portrait", "Square"]:
                    if orient not in bucket_orientations:
                        continue
                    
                    orient_ratios = [(rs, rv, o, imgs) for (rs, rv, o), imgs in ratio_counts.items() if o == orient]
                    if orient_ratios:
                        orient_ratios.sort(key=lambda x: len(x[3]), reverse=True)
                        selected_buckets.append(orient_ratios[0])
                    else:
                        ratio_str, ratio_val = virtual_ratios[orient]
                        selected_buckets.append((ratio_str, ratio_val, orient, []))
                
                remaining = num_buckets - len(selected_buckets)
                if remaining > 0:
                    all_ratios = [(rs, rv, o, imgs) for (rs, rv, o), imgs in ratio_counts.items()]
                    all_ratios.sort(key=lambda x: len(x[3]), reverse=True)
                    
                    for ratio_str, ratio_val, orient, imgs in all_ratios:
                        if (ratio_str, ratio_val, orient, imgs) in selected_buckets:
                            continue
                        selected_buckets.append((ratio_str, ratio_val, orient, imgs))
                        if len(selected_buckets) >= num_buckets:
                            break
            else:
                all_ratios = [(rs, rv, o, imgs) for (rs, rv, o), imgs in ratio_counts.items()]
                all_ratios.sort(key=lambda x: len(x[3]), reverse=True)
                selected_buckets = all_ratios[:num_buckets]
            
            for ratio_str, ratio_val, orient, imgs in selected_buckets:
                if imgs:
                    sample = imgs[0]
                    rep = f"{sample['width']}x{sample['height']}"
                    virtual = False
                else:
                    rep = f"{int(1024 * ratio_val)}x1024" if ratio_val >= 1 else f"1024x{int(1024 / ratio_val)}"
                    virtual = True
                    
                result["buckets"][ratio_str] = {
                    "name": ratio_str,
                    "ratio": ratio_val,
                    "ratio_str": ratio_str,
                    "orientation": orient,
                    "representative": rep,
                    "images": [],
                    "eligible": [],
                    "virtual": virtual
                }
        
        for img in image_data:
            eligible_buckets = []
            for bucket_name, bucket in result["buckets"].items():
                diff = self._ratio_diff_pct(img["ratio"], bucket["ratio"])
                match_pct = max(0, 100 - diff)
                if diff <= tolerance:
                    eligible_buckets.append((bucket_name, diff, match_pct))
            
            img["eligible"] = eligible_buckets
            for bucket_name, diff, match_pct in eligible_buckets:
                result["buckets"][bucket_name]["eligible"].append((img, diff, match_pct))
        
        assigned = set()
        
        while True:
            bucket_counts = {name: len(b["images"]) for name, b in result["buckets"].items()}
            if not bucket_counts:
                break
                
            available = [n for n, c in bucket_counts.items() if max_per_bucket <= 0 or c < max_per_bucket]
            
            if not available:
                break
            
            min_bucket = min(available, key=lambda n: bucket_counts[n])
            candidates = [(img, diff, mp) for img, diff, mp in result["buckets"][min_bucket]["eligible"] if id(img) not in assigned]
            
            if not candidates:
                other = [n for n in available if n != min_bucket]
                found = False
                for bn in sorted(other, key=lambda n: bucket_counts[n]):
                    candidates = [(img, diff, mp) for img, diff, mp in result["buckets"][bn]["eligible"] if id(img) not in assigned]
                    if candidates:
                        min_bucket = bn
                        found = True
                        break
                if not found:
                    break
            
            candidates.sort(key=lambda x: x[1])
            img, diff, match_pct = candidates[0]
            
            img["assigned_bucket"] = min_bucket
            img["diff"] = diff
            img["match_pct"] = match_pct
            img["status"] = "ok"
            result["buckets"][min_bucket]["images"].append(img)
            assigned.add(id(img))
        
        for img in image_data:
            if id(img) not in assigned:
                if not result["buckets"]:
                    img["status"] = "unassigned"
                    img["match_pct"] = 0
                    result["unassigned"].append(img)
                else:
                    best_bucket = None
                    best_diff = float('inf')
                    for bucket_name, bucket in result["buckets"].items():
                        diff = self._ratio_diff_pct(img["ratio"], bucket["ratio"])
                        if diff < best_diff:
                            best_diff = diff
                            best_bucket = bucket_name
                    
                    img["assigned_bucket"] = best_bucket
                    img["diff"] = best_diff
                    img["match_pct"] = max(0, 100 - best_diff)
                    
                    if best_diff > tolerance:
                        img["status"] = "outlier"
                        result["outliers"].append(img)
                    else:
                        img["status"] = "unassigned"
                        result["unassigned"].append(img)
        
        for ratio_str, data in result["ratio_summary"].items():
            images = data["images"]
            if not images:
                continue
            
            if ratio_str in result["buckets"]:
                data["status"] = "bucket"
                data["bucket"] = ratio_str
                data["match_pct"] = 100
            else:
                ok = sum(1 for img in images if img.get("status") == "ok")
                outlier = sum(1 for img in images if img.get("status") == "outlier")
                unassigned = sum(1 for img in images if img.get("status") == "unassigned")
                
                if outlier == 0 and unassigned == 0:
                    data["status"] = "ok"
                elif ok == 0:
                    data["status"] = "outlier"
                else:
                    data["status"] = "partial"
                
                assigned_bucket_name = None
                for img in images:
                    if img.get("assigned_bucket"):
                        assigned_bucket_name = img["assigned_bucket"]
                        break
                
                if assigned_bucket_name and assigned_bucket_name in result["buckets"]:
                    bucket = result["buckets"][assigned_bucket_name]
                    
                    ratio_parts = ratio_str.split(':')
                    if len(ratio_parts) == 2:
                        try:
                            img_ratio = int(ratio_parts[0]) / int(ratio_parts[1])
                        except:
                            img_ratio = images[0]["ratio"]
                    else:
                        img_ratio = images[0]["ratio"]
                    
                    diff = self._ratio_diff_pct(img_ratio, bucket["ratio"])
                    data["match_pct"] = max(0, 100 - diff)
                    data["bucket"] = assigned_bucket_name
                else:
                    data["match_pct"] = 0
        
        return result
    
    def _generate_report(self, result: dict, output_dir: str, file_action: str, unassigned_action: str) -> str:
        """
        Generate an HTML report summarizing bucketing analysis, including bucket tables, ratio summaries, lists of unassigned/outlier/resolution-issue images, and an organize preview.
        
        Parameters:
            result (dict): Analysis result produced by _analyze_buckets. Expected keys include:
                - total (int)
                - buckets (dict): mapping bucket name -> bucket dict with keys "images", "eligible", "orientation", "ratio_str", "manual"
                - outliers (list): images outside tolerance
                - unassigned (list): images not placed in any bucket
                - resolution_low (list): images below min resolution
                - resolution_high (list): images above max resolution
                - ratio_summary (dict): mapping ratio_str -> summary dict with "images", "status", "bucket", "match_pct"
            output_dir (str): Path to the target output directory used in the organize preview; empty string disables path rendering.
            file_action (str): Action label shown in the organize preview (e.g., "Copy" or "Move").
            unassigned_action (str): User-selected handling for unassigned images; affects which preview folders (_unassigned, _issues) are shown.
        
        Returns:
            str: An HTML string that presents counts, bucket breakdowns, per-ratio statistics, lists of unassigned/outlier/resolution items, and a preview of the proposed folder structure.
        """
        html = ['<div style="font-family:system-ui,-apple-system,sans-serif;">']
        
        total = result["total"]
        assigned = sum(len(b["images"]) for b in result["buckets"].values())
        outliers = len(result["outliers"])
        unassigned = len(result["unassigned"])
        res_low = len(result["resolution_low"])
        res_high = len(result["resolution_high"])
        
        html.append('<div style="margin-bottom:16px;">')
        html.append(f'<span style="color:#64B5F6;font-weight:600;margin-right:16px;">Total: {total}</span>')
        html.append(f'<span style="color:{self.COLORS["ok"]};font-weight:600;margin-right:16px;">Assigned: {assigned}</span>')
        if unassigned:
            html.append(f'<span style="color:{self.COLORS["partial"]};font-weight:600;margin-right:16px;">Unassigned: {unassigned}</span>')
        if outliers:
            html.append(f'<span style="color:{self.COLORS["outlier"]};font-weight:600;margin-right:16px;">Outliers: {outliers}</span>')
        if res_low:
            html.append(f'<span style="color:{self.COLORS["resolution"]};font-weight:600;margin-right:16px;">Too Small: {res_low}</span>')
        if res_high:
            html.append(f'<span style="color:{self.COLORS["resolution"]};font-weight:600;">Too Large: {res_high}</span>')
        html.append('</div>')
        
        html.append('<h4 style="color:#fff;margin:16px 0 8px 0;">Buckets</h4>')
        html.append('<table style="width:100%;border-collapse:collapse;font-size:13px;">')
        html.append('<tr style="border-bottom:1px solid #444;">')
        html.append('<th style="text-align:center;padding:6px;color:#aaa;">Bucket</th>')
        html.append('<th style="text-align:center;padding:6px;color:#aaa;">Orientation</th>')
        html.append('<th style="text-align:center;padding:6px;color:#aaa;">Assigned</th>')
        html.append('<th style="text-align:center;padding:6px;color:#aaa;">%</th>')
        html.append('<th style="text-align:center;padding:6px;color:#aaa;">Eligible</th>')
        html.append('</tr>')
        
        bucket_counts = [len(b["images"]) for b in result["buckets"].values()]
        max_bucket_count = max(bucket_counts) if bucket_counts else 0
        
        for name, bucket in sorted(result["buckets"].items(), key=lambda x: len(x[1]["images"]), reverse=True):
            count = len(bucket["images"])
            eligible = len(bucket["eligible"])
            pct = (count / assigned * 100) if assigned > 0 else 0
            
            if count == 0:
                color = self.COLORS["filtered"]
            elif max_bucket_count > 0 and count < max_bucket_count * 0.5:
                color = self.COLORS["outlier"]
            elif max_bucket_count > 0 and count < max_bucket_count * 0.8:
                color = self.COLORS["partial"]
            else:
                color = self.COLORS["bucket"]
            
            manual_mark = " 📝" if bucket.get("manual") else ""
            html.append(f'<tr style="border-bottom:1px solid #333;">')
            html.append(f'<td style="padding:6px;color:{color};font-weight:600;text-align:center;">{name}{manual_mark}</td>')
            html.append(f'<td style="padding:6px;color:{color};text-align:center;">{bucket["orientation"]}</td>')
            html.append(f'<td style="padding:6px;text-align:center;color:{color};">{count}</td>')
            html.append(f'<td style="padding:6px;text-align:center;color:{color};">{pct:.1f}%</td>')
            html.append(f'<td style="padding:6px;text-align:center;color:#888;">{eligible}</td>')
            html.append('</tr>')
        
        if not result["buckets"]:
            html.append('<tr><td colspan="5" style="text-align:center;padding:12px;color:#888;">No buckets created. Select orientation preferences or enter manual ratios.</td></tr>')
        
        html.append('</table>')
        
        counts = [len(b["images"]) for b in result["buckets"].values() if len(b["images"]) > 0]
        if len(counts) >= 2:
            balance = (min(counts) / max(counts)) * 100 if max(counts) > 0 else 0
            color = self.COLORS["ok"] if balance >= 70 else (self.COLORS["partial"] if balance >= 40 else self.COLORS["outlier"])
            min_count = min(counts)
            html.append(f'<div style="margin:12px 0;color:{color};font-weight:600;">Balance: {balance:.0f}% (Prune would keep {min_count} per bucket)</div>')
        
        html.append('<h4 style="color:#fff;margin:20px 0 8px 0;">All Ratios in Dataset</h4>')
        html.append('<table style="width:100%;border-collapse:collapse;font-size:13px;">')
        html.append('<tr style="border-bottom:1px solid #444;">')
        html.append('<th style="text-align:center;padding:6px;color:#aaa;">Ratio</th>')
        html.append('<th style="text-align:center;padding:6px;color:#aaa;">Orientation</th>')
        html.append('<th style="text-align:center;padding:6px;color:#aaa;">Count</th>')
        html.append('<th style="text-align:center;padding:6px;color:#aaa;">Match %</th>')
        html.append('<th style="text-align:center;padding:6px;color:#aaa;">Status</th>')
        html.append('<th style="text-align:center;padding:6px;color:#aaa;">Bucket</th>')
        html.append('</tr>')
        
        sorted_ratios = sorted(result["ratio_summary"].items(), key=lambda x: len(x[1]["images"]), reverse=True)
        for ratio_str, data in sorted_ratios:
            images = data["images"]
            if not images:
                continue
            count = len(images)
            status = data["status"] or "unassigned"
            bucket = data["bucket"] or "-"
            match_pct = data.get("match_pct", 0)
            orient = images[0]["orientation"]
            color = self.COLORS.get(status, "#fff")
            
            match_color = self.COLORS["ok"] if match_pct >= 80 else (self.COLORS["partial"] if match_pct >= 50 else self.COLORS["outlier"])
            
            html.append(f'<tr style="border-bottom:1px solid #333;">')
            html.append(f'<td style="padding:6px;color:{color};font-weight:600;text-align:center;">{ratio_str}</td>')
            html.append(f'<td style="padding:6px;color:{color};text-align:center;">{orient}</td>')
            html.append(f'<td style="padding:6px;text-align:center;color:{color};">{count}</td>')
            html.append(f'<td style="padding:6px;text-align:center;color:{match_color};">{match_pct:.1f}%</td>')
            html.append(f'<td style="padding:6px;text-align:center;color:{color};">{status.upper()}</td>')
            html.append(f'<td style="padding:6px;text-align:center;color:{color};">→ {bucket}</td>')
            html.append('</tr>')
        html.append('</table>')
        
        if result["unassigned"]:
            html.append(f'<h4 style="color:{self.COLORS["partial"]};margin:20px 0 8px 0;">Unassigned ({len(result["unassigned"])})</h4>')
            html.append('<div style="max-height:200px;overflow-y:auto;font-size:12px;">')
            for img in result["unassigned"]:
                html.append(f'<div style="color:{self.COLORS["partial"]};">{img["filename"]} ({img["ratio_str"]} - {img["orientation"]})</div>')
            html.append('</div>')
        
        if result["outliers"]:
            html.append(f'<h4 style="color:{self.COLORS["outlier"]};margin:20px 0 8px 0;">Outliers: Outside Tolerance ({len(result["outliers"])})</h4>')
            html.append('<div style="max-height:200px;overflow-y:auto;font-size:12px;">')
            for img in sorted(result["outliers"], key=lambda x: -x.get("diff", 0)):
                diff = img.get("diff", 0)
                html.append(f'<div style="color:{self.COLORS["outlier"]};">{img["filename"]} ({img["ratio_str"]}) - {diff:.0f}% from nearest</div>')
            html.append('</div>')
        
        if result["resolution_low"]:
            html.append(f'<h4 style="color:{self.COLORS["resolution"]};margin:20px 0 8px 0;">Too Small ({len(result["resolution_low"])})</h4>')
            html.append('<div style="max-height:150px;overflow-y:auto;font-size:12px;">')
            for img in result["resolution_low"]:
                html.append(f'<div style="color:{self.COLORS["resolution"]};">{img["filename"]} ({img["max_dim"]}px)</div>')
            html.append('</div>')
        
        if result["resolution_high"]:
            html.append(f'<h4 style="color:{self.COLORS["resolution"]};margin:20px 0 8px 0;">Too Large ({len(result["resolution_high"])})</h4>')
            html.append('<div style="max-height:150px;overflow-y:auto;font-size:12px;">')
            for img in result["resolution_high"]:
                html.append(f'<div style="color:{self.COLORS["resolution"]};">{img["filename"]} ({img["max_dim"]}px)</div>')
            html.append('</div>')
        
        html.append('<h4 style="color:#fff;margin:20px 0 8px 0;">Organize Preview</h4>')
        html.append('<div style="background:#1a1a2e;padding:12px;border-radius:6px;font-family:monospace;font-size:13px;margin-bottom:16px;">')
        if output_dir:
            html.append(f'<div style="color:#64B5F6;">📁 {output_dir}/ ({file_action})</div>')
        else:
            html.append('<div style="color:#888;">📁 [set output directory]/</div>')
        
        for name, bucket in sorted(result["buckets"].items(), key=lambda x: len(x[1]["images"]), reverse=True):
            count = len(bucket["images"])
            folder = f'{bucket["orientation"]}_{bucket["ratio_str"].replace(":", "x")}'
            color = self.COLORS["ok"] if count > 0 else self.COLORS["filtered"]
            html.append(f'<div style="margin-left:16px;color:{color};">├── 📂 {folder}/ ({count})</div>')
        
        issue_count = len(result["outliers"]) + len(result["unassigned"]) + res_low + res_high
        if issue_count:
            if "unassigned" in unassigned_action.lower():
                html.append(f'<div style="margin-left:16px;color:{self.COLORS["partial"]};"> ├── 📂 _unassigned/ ({len(result["unassigned"])})</div>')
            if "issues" in unassigned_action.lower() or outliers or res_low or res_high:
                other_issues = outliers + res_low + res_high
                html.append(f'<div style="margin-left:16px;color:{self.COLORS["resolution"]};"> └── 📂 _issues/ ({other_issues})</div>')
        html.append('</div>')
        
        html.append('</div>')
        return ''.join(html)
    
    def _prune_to_balanced(self, result: dict, output_dir: str, file_action: str, unassigned_action: str) -> str:
        """
        Balance all buckets by reducing each to the size of the smallest bucket and moving excess images to `result["unassigned"]`.
        
        Parameters:
            result (dict): Bucketing result structure; this function mutates `result["buckets"]` and appends moved images to `result["unassigned"]`.
            output_dir (str): Directory used when generating the final report (passed through to the report generator).
            file_action (str): File action label (e.g., "Copy" or "Move") passed to the report generator.
            unassigned_action (str): Behavior for unassigned files passed to the report generator.
        
        Returns:
            str: HTML report summarizing the post-prune state.
        """
        counts = [len(b["images"]) for b in result["buckets"].values()]
        if not counts:
            gr.Warning("No buckets to prune.")
            return self._generate_report(result, output_dir, file_action, unassigned_action)
        
        min_count = min(counts) if counts else 0
        pruned = []
        
        for bucket in result["buckets"].values():
            bucket["images"].sort(key=lambda x: x.get("diff", 0))
            pruned.extend(bucket["images"][min_count:])
            bucket["images"] = bucket["images"][:min_count]
        
        result["unassigned"].extend(pruned)
        
        gr.Info(f"Pruned to {min_count} images per bucket. {len(pruned)} moved to unassigned.")
        return self._generate_report(result, output_dir, file_action, unassigned_action)
    
    def _organize(self, result: dict, output_dir: str, file_action: str, unassigned_action: str) -> str:
        """
        Organize images from a bucketing result into per-bucket folders and optionally move/copy unassigned or problematic files.
        
        Parameters:
            result (dict): Bucketing analysis output containing keys "buckets", "unassigned", "outliers", "resolution_low", and "resolution_high".
            output_dir (str): Destination directory where bucket folders and special folders (_unassigned, _issues) will be created.
            file_action (str): Either "Copy" or "Move"; determines whether files are copied or moved into the output folders.
            unassigned_action (str): Controls handling of unassigned images; if this string contains "unassigned" (case-insensitive), unassigned images are placed into an "_unassigned" folder.
        
        Returns:
            str: An HTML report generated by _generate_report summarizing the (post-)organization state.
        
        Notes:
            - For each bucket with images, a folder named "<Orientation>_<W>x<H>" is created and images (and matching .txt sidecar files when present) are copied or moved there.
            - If enabled, unassigned images are placed in "_unassigned".
            - Outliers and resolution issues are collected into an "_issues" folder; duplicate source paths are skipped.
            - File operation errors are logged or ignored per-file; the function proceeds without raising on individual file failures.
        """
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        copied = 0
        do_move = file_action == "Move"
        
        for name, bucket in result["buckets"].items():
            if not bucket["images"]:
                continue
            folder = out_path / f'{bucket["orientation"]}_{bucket["ratio_str"].replace(":", "x")}'
            folder.mkdir(exist_ok=True)
            for img in bucket["images"]:
                src = Path(img["path"])
                try:
                    if do_move:
                        shutil.move(str(src), str(folder / src.name))
                        txt = src.with_suffix(".txt")
                        if txt.exists():
                            shutil.move(str(txt), str(folder / txt.name))
                    else:
                        shutil.copy2(src, folder / src.name)
                        txt = src.with_suffix(".txt")
                        if txt.exists():
                            shutil.copy2(txt, folder / txt.name)
                    copied += 1
                except Exception as e:
                    logging.error(f"Failed: {src}: {e}")
        
        if "unassigned" in unassigned_action.lower() and result["unassigned"]:
            unassigned_folder = out_path / "_unassigned"
            unassigned_folder.mkdir(exist_ok=True)
            for img in result["unassigned"]:
                src = Path(img["path"])
                try:
                    if do_move:
                        shutil.move(str(src), str(unassigned_folder / src.name))
                    else:
                        shutil.copy2(src, unassigned_folder / src.name)
                    copied += 1
                except Exception:
                    pass
        
        all_issues = result["outliers"] + result["resolution_low"] + result["resolution_high"]
        if all_issues:
            issue_folder = out_path / "_issues"
            issue_folder.mkdir(exist_ok=True)
            done = set()
            for img in all_issues:
                src = Path(img["path"])
                if str(src) in done:
                    continue
                done.add(str(src))
                try:
                    if do_move:
                        shutil.move(str(src), str(issue_folder / src.name))
                    else:
                        shutil.copy2(src, issue_folder / src.name)
                    copied += 1
                except Exception:
                    pass
        
        action_word = "Moved" if do_move else "Copied"
        gr.Info(f"{action_word} {copied} files to {output_dir}")
        return self._generate_report(result, output_dir, file_action, unassigned_action)
    
    def create_gui(self, app) -> tuple:
        """
        Create the Gradio user interface controls for the Bucketing tool and return the primary action button and a list of input widgets.
        
        Parameters:
            app: Application context or UI container used to integrate the generated controls.
        
        Returns:
            tuple: (analyze_btn, inputs) where `analyze_btn` is the main Analyze button control and `inputs` is a list of all created input widgets and related controls (used for wiring event handlers).
        """
        gr.Markdown(self.config.description)
        
        gr.Markdown("**Bucket Settings**")
        with gr.Row():
            num_buckets = gr.Number(
                value=3, precision=0, minimum=1, label="Number of Buckets",
                info="How many ratio groups to create (auto mode)"
            )
            tolerance = gr.Number(
                value=20, precision=0, minimum=1, label="Tolerance (%)",
                info="Maximum ratio difference to accept into a bucket"
            )
            max_per_bucket = gr.Number(
                value=50, precision=0, minimum=0, label="Max per Bucket",
                info="Limit images per bucket, 0 means no limit"
            )
            manual_buckets = gr.Textbox(
                label="Manual Buckets (Optional)",
                placeholder="e.g., 3:2, 2:3, 1:1",
                info="Comma-separated ratios. Leave empty for auto"
            )
        
        gr.Markdown("**Buckets to Create**")
        with gr.Row():
            create_landscape = gr.Checkbox(
                label="Landscape", value=True,
                info="Create buckets from landscape images"
            )
            create_portrait = gr.Checkbox(
                label="Portrait", value=True,
                info="Create buckets from portrait images"
            )
            create_square = gr.Checkbox(
                label="Square", value=True,
                info="Create buckets from square images"
            )
            min_res = gr.Number(
                value=512, precision=0, label="Min Resolution",
                info="Flag images below this"
            )
            max_res = gr.Number(
                value=2048, precision=0, label="Max Resolution",
                info="Flag images above this"
            )
        
        gr.Markdown("**Output**")
        with gr.Row():
            output_dir = gr.Textbox(
                label="Directory", placeholder="Required for Organize",
                info="Destination folder for organized files"
            )
            file_action = gr.Dropdown(
                choices=["Copy", "Move"], value="Copy", label="Action",
                info="Copy preserves originals, Move deletes source"
            )
            unassigned_action = gr.Dropdown(
                choices=["Include in _unassigned", "Include in _issues", "Skip"],
                value="Include in _unassigned", label="Unassigned",
                info="What to do with images that didn't fit"
            )
        
        with gr.Row():
            analyze_btn = gr.Button("Analyze", variant="secondary", elem_id="bucket_analyze_btn")
            prune_btn = gr.Button("Prune to Balanced", variant="secondary", elem_id="bucket_prune_btn")
            organize_btn = gr.Button("Organize", variant="primary", elem_id="bucket_organize_btn")
        
        result_output = gr.HTML(value=self._empty_state())
        
        inputs = [num_buckets, tolerance, max_per_bucket, manual_buckets, create_landscape, create_portrait, create_square,
                  min_res, max_res, unassigned_action, file_action, output_dir, result_output, analyze_btn, prune_btn, organize_btn]
        return (analyze_btn, inputs)
    
    def wire_events(self, app, run_button, inputs: list, gallery_output: gr.Gallery,
                    limit_count=None) -> None:
        """
                    Attach GUI event handlers to the BucketingTool controls and bind Analyze/Prune/Organize actions.
                    
                    Binds click and change callbacks for the provided Gradio controls so user interactions run the tool (with optional dataset limiting), produce HTML reports, and print a concise run summary to the console. Handlers apply the current UI settings when invoking analysis, pruning, or organization flows and update the result output component.
                    
                    Parameters:
                        app: Application object exposing the current dataset (app.dataset) used when running actions.
                        run_button: Unused placeholder for compatibility with wiring call sites.
                        inputs (list): Ordered list of Gradio input components expected as:
                            [num_buckets, tolerance, max_per_bucket, manual_buckets,
                             create_landscape, create_portrait, create_square,
                             min_resolution, max_resolution, unassigned_action, file_action, output_dir,
                             result_output, analyze_btn, prune_btn, organize_btn]
                        gallery_output (gr.Gallery): Gallery component showing dataset thumbnails (not modified by this function).
                        limit_count (Optional[int]): If provided, appended to callbacks and used to limit the number of images processed.
                    
                    Side effects:
                        - Registers callbacks on analyze_btn, prune_btn, organize_btn and change events on key inputs.
                        - Prints brief run summaries to stdout.
                    """
                    import copy
        
        # ANSI color codes
        CYAN = "\033[96m"
        GREEN = "\033[92m"
        YELLOW = "\033[93m"
        RESET = "\033[0m"
        BOLD = "\033[1m"
        
        num_buckets, tolerance, max_per_bucket, manual_buckets = inputs[0], inputs[1], inputs[2], inputs[3]
        create_l, create_p, create_s = inputs[4], inputs[5], inputs[6]
        min_res, max_res, unassigned_action, file_action, output_dir = inputs[7], inputs[8], inputs[9], inputs[10], inputs[11]
        result_output, analyze_btn, prune_btn, organize_btn = inputs[12], inputs[13], inputs[14], inputs[15]
        
        # Build all_inputs - append limit_count if provided
        all_inputs = [num_buckets, tolerance, max_per_bucket, manual_buckets, create_l, create_p, create_s, min_res, max_res, unassigned_action, file_action, output_dir]
        if limit_count is not None:
            all_inputs.append(limit_count)
        
        def _get_limited_dataset(limit_val):
            """
            Return a dataset limited to the first `limit_val` images along with the number of images in the returned dataset.
            
            Parameters:
                limit_val (int | str | None): Maximum number of images to include. If convertible to an integer greater than zero and smaller than the current dataset size, the returned dataset will contain only the first `limit_val` images. If `limit_val` is None, zero, non-positive, or not convertible to int, the full dataset is returned.
            
            Returns:
                tuple: (run_dataset, count)
                    - run_dataset: a dataset object containing up to `limit_val` images, or None if no dataset or no images are available.
                    - count (int): number of images in `run_dataset` (0 if no dataset).
            
            Notes:
                - If no dataset or no images are present, returns (None, 0).
                - When a positive numeric limit is applied and reduces the dataset, a console message is printed indicating the applied limit.
            """
            if not app.dataset or not app.dataset.images:
                return None, 0
            
            run_dataset = app.dataset
            total_count = len(app.dataset.images)
            
            # Apply limit if set
            if limit_val:
                try:
                    limit = int(limit_val)
                    if limit > 0 and total_count > limit:
                        run_dataset = copy.copy(app.dataset)
                        run_dataset.images = app.dataset.images[:limit]
                        print(f"{YELLOW}Limiting to first {limit} images (from {total_count} loaded).{RESET}")
                except (ValueError, TypeError):
                    pass
            
            return run_dataset, len(run_dataset.images)
        
        def _print_summary(result_html):
            """
            Prints a concise colorized summary extracted from an HTML report.
            
            Parses the provided HTML report for total, assigned, unassigned, and outlier counts and prints a single-line summary to standard output.
            
            Parameters:
                result_html (str): HTML report string containing summary metrics (e.g., "Total:", "Assigned:", "Unassigned:", "Outliers:").
            """
            import re
            # Extract Total, Assigned, Outliers, etc from the HTML
            total_match = re.search(r'Total: (\d+)', result_html)
            assigned_match = re.search(r'Assigned: (\d+)', result_html)
            outliers_match = re.search(r'Outliers: (\d+)', result_html)
            unassigned_match = re.search(r'Unassigned: (\d+)', result_html)
            
            total = total_match.group(1) if total_match else "?"
            assigned = assigned_match.group(1) if assigned_match else "0"
            outliers = outliers_match.group(1) if outliers_match else "0"
            unassigned = unassigned_match.group(1) if unassigned_match else "0"
            
            print(f"{CYAN}Summary:{RESET} Total: {total} | {GREEN}Assigned: {assigned}{RESET} | {YELLOW}Unassigned: {unassigned}{RESET} | Outliers: {outliers}")
        
        def analyze(*args):
            # Extract limit_val from end if limit_count was added
            """
            Run a bucketing analysis using positional GUI inputs and return the generated HTML report.
            
            Parameters:
                args (tuple): Ordered values from the GUI controls:
                    0: num_buckets (int | None) — number of buckets to create.
                    1: tolerance (float | None) — allowable percent difference for matching ratios.
                    2: max_per_bucket (int | None) — maximum images per bucket (0 or None = unlimited).
                    3: manual_buckets (str) — manual ratio definitions (e.g., "3:2,1:1").
                    4: create_landscape (bool) — include landscape buckets.
                    5: create_portrait (bool) — include portrait buckets.
                    6: create_square (bool) — include square buckets.
                    7: min_resolution (int | None) — minimum allowed image dimension in pixels.
                    8: max_resolution (int | None) — maximum allowed image dimension in pixels.
                    9: unassigned_action (str) — behavior for unassigned images.
                    10: file_action (str) — "Copy" or "Move" when organizing files.
                    11: output_dir (str) — target directory for organize/prune actions.
                    If a limit_count was provided when wiring events, an extra final value supplies the image limit.
            
            Returns:
                str: HTML report produced by apply_to_dataset (or the empty-state HTML if no dataset is available).
            """
            limit_val = args[-1] if limit_count is not None else None
            
            run_dataset, count = _get_limited_dataset(limit_val)
            if run_dataset is None:
                return self._empty_state()
            
            print(f"")
            print(f"{BOLD}{CYAN}--- Running Bucketing Tool (Analyze) on {count} images ---{RESET}")
            print(f"{CYAN}Settings:{RESET}")
            print(f"  Buckets: {args[0]} | Tolerance: {args[1]}% | Max per Bucket: {args[2] or 'unlimited'}")
            print(f"  Min Res: {args[7]}px | Max Res: {args[8]}px")
            if args[3]:
                print(f"  Manual Buckets: {args[3]}")
            print(f"")
            
            result = self.apply_to_dataset(
                run_dataset,
                num_buckets=int(args[0]) if args[0] else 3,
                tolerance=float(args[1]) if args[1] else 20,
                max_per_bucket=int(args[2]) if args[2] else 0,
                manual_buckets=args[3],
                create_landscape=args[4],
                create_portrait=args[5],
                create_square=args[6],
                min_resolution=int(args[7]) if args[7] else 512,
                max_resolution=int(args[8]) if args[8] else 2048,
                unassigned_action=args[9],
                file_action=args[10],
                output_dir=args[11],
                action="analyze"
            )
            
            _print_summary(result)
            print(f"{BOLD}{CYAN}--- Bucketing Tool (Analyze) Complete ---{RESET}")
            print(f"")
            return result
        
        def prune(*args):
            """
            Run the pruning workflow using provided GUI-style arguments, balance bucket sizes, and return the generated HTML report.
            
            Parameters:
            	args: A sequence of inputs (typically from the GUI) in the following order:
            		0: num_buckets (int or empty) — desired number of buckets.
            		1: tolerance (float or empty) — percent tolerance for ratio matching.
            		2: max_per_bucket (int or empty) — maximum images per bucket (0 for no limit).
            		3: manual_buckets (str) — manual ratio definitions (e.g., "3:2,1:1").
            		4: create_landscape (bool) — include landscape buckets.
            		5: create_portrait (bool) — include portrait buckets.
            		6: create_square (bool) — include square buckets.
            		7: min_resolution (int or empty) — minimum allowed image dimension.
            		8: max_resolution (int or empty) — maximum allowed image dimension.
            		9: unassigned_action (str) — how to handle unassigned images.
            		10: file_action (str) — "Copy" or "Move" when organizing files.
            		11: output_dir (str) — directory for organize/prune outputs.
            		If a module-level `limit_count` is set, an additional trailing value may be provided as the dataset limit.
            
            Returns:
            	HTML report string summarizing the prune operation, bucket balances, and any moved/unassigned images; returns the tool's empty-state HTML if no dataset is available.
            """
            limit_val = args[-1] if limit_count is not None else None
            
            run_dataset, count = _get_limited_dataset(limit_val)
            if run_dataset is None:
                return self._empty_state()
            
            print(f"")
            print(f"{BOLD}{CYAN}--- Running Bucketing Tool (Prune) on {count} images ---{RESET}")
            
            result = self.apply_to_dataset(
                run_dataset,
                num_buckets=int(args[0]) if args[0] else 3,
                tolerance=float(args[1]) if args[1] else 20,
                max_per_bucket=int(args[2]) if args[2] else 0,
                manual_buckets=args[3],
                create_landscape=args[4],
                create_portrait=args[5],
                create_square=args[6],
                min_resolution=int(args[7]) if args[7] else 512,
                max_resolution=int(args[8]) if args[8] else 2048,
                unassigned_action=args[9],
                file_action=args[10],
                output_dir=args[11],
                action="prune"
            )
            
            _print_summary(result)
            print(f"{BOLD}{CYAN}--- Bucketing Tool (Prune) Complete ---{RESET}")
            print(f"")
            return result
        
        def organize(*args):
            """
            Run the organize action of the BucketingTool on a (optionally limited) dataset, copy/move files into bucket folders, print a compact run summary, and return the resulting HTML report.
            
            Parameters:
                *args: Positional values expected in the following order:
                    0: num_buckets (int or empty) — number of buckets to create.
                    1: tolerance (float or empty) — percent tolerance for bucket matching.
                    2: max_per_bucket (int or empty) — maximum images to assign per bucket (0 for no limit).
                    3: manual_buckets (str) — manual ratio definitions, e.g., "3:2,1:1".
                    4: create_landscape (bool) — whether to create landscape buckets.
                    5: create_portrait (bool) — whether to create portrait buckets.
                    6: create_square (bool) — whether to create square buckets.
                    7: min_resolution (int or empty) — minimum allowed image dimension.
                    8: max_resolution (int or empty) — maximum allowed image dimension.
                    9: unassigned_action (str) — behavior for unassigned images.
                    10: file_action (str) — "Copy" or "Move" for organizing files.
                    11: output_dir (str) — target directory for organized output.
                    (If a dataset limit is configured via closure, an additional final arg may be present and is treated as the limit value.)
            
            Returns:
                html_report (str): The generated HTML report produced by apply_to_dataset summarizing organization results.
            """
            limit_val = args[-1] if limit_count is not None else None
            
            run_dataset, count = _get_limited_dataset(limit_val)
            if run_dataset is None:
                return self._empty_state()
            
            print(f"")
            print(f"{BOLD}{CYAN}--- Running Bucketing Tool (Organize) on {count} images ---{RESET}")
            print(f"  Output: {args[11]}")
            print(f"  Action: {args[10]}")
            print(f"")
            
            result = self.apply_to_dataset(
                run_dataset,
                num_buckets=int(args[0]) if args[0] else 3,
                tolerance=float(args[1]) if args[1] else 20,
                max_per_bucket=int(args[2]) if args[2] else 0,
                manual_buckets=args[3],
                create_landscape=args[4],
                create_portrait=args[5],
                create_square=args[6],
                min_resolution=int(args[7]) if args[7] else 512,
                max_resolution=int(args[8]) if args[8] else 2048,
                unassigned_action=args[9],
                file_action=args[10],
                output_dir=args[11],
                action="organize"
            )
            
            _print_summary(result)
            print(f"{BOLD}{CYAN}--- Bucketing Tool (Organize) Complete ---{RESET}")
            print(f"")
            return result
        
        analyze_btn.click(fn=analyze, inputs=all_inputs, outputs=[result_output])
        prune_btn.click(fn=prune, inputs=all_inputs, outputs=[result_output])
        organize_btn.click(fn=organize, inputs=all_inputs, outputs=[result_output])
        
        for c in [num_buckets, tolerance, max_per_bucket, manual_buckets, create_l, create_p, create_s, min_res, max_res]:
            c.change(fn=analyze, inputs=all_inputs, outputs=[result_output])

