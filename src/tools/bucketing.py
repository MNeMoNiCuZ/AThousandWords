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
        return ToolConfig(
            name="bucketing",
            display_name="Bucketing",
            description="### Resolution Bucketing\nBalance your dataset by distributing images across aspect ratio buckets.",
            icon=""
        )
    
    def _get_defaults(self) -> dict:
        """Return default values for all settings."""
        return {
            "num_buckets": 3,
            "tolerance": 20,
            "max_per_bucket": 50,
            "manual_buckets": "",
            "create_landscape": True,
            "create_portrait": True,
            "create_square": True,
            "min_resolution": 512,
            "max_resolution": 2048,
            "unassigned_action": "Include in _unassigned",
            "file_action": "Copy",
            "output_dir": "",
        }
    
    def get_loaded_values(self, app) -> list:
        """Load saved settings from user config."""
        import gradio as gr
        
        defaults = self._get_defaults()
        saved = {}
        
        try:
            tool_settings = app.config_mgr.user_config.get("tool_settings", {})
            saved = tool_settings.get("bucketing", {})
        except Exception:
            pass
        
        values = {**defaults, **saved}
        
        # Order must match create_gui inputs list (16 items including result_output and buttons)
        return [
            gr.update(value=values["num_buckets"]),
            gr.update(value=values["tolerance"]),
            gr.update(value=values["max_per_bucket"]),
            gr.update(value=values["manual_buckets"]),
            gr.update(value=values["create_landscape"]),
            gr.update(value=values["create_portrait"]),
            gr.update(value=values["create_square"]),
            gr.update(value=values["min_resolution"]),
            gr.update(value=values["max_resolution"]),
            gr.update(value=values["unassigned_action"]),
            gr.update(value=values["file_action"]),
            gr.update(value=values["output_dir"]),
            gr.update(),  # result_output - no change
            gr.update(),  # analyze_btn - no change
            gr.update(),  # prune_btn - no change
            gr.update(),  # organize_btn - no change
        ]
    
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
        return '<div style="padding:20px;color:#888;">Load images and click Analyze.</div>'
    
    def _get_image_size(self, path: Path) -> tuple:
        try:
            with Image.open(path) as img:
                return img.size
        except Exception:
            return None
    
    def _get_orientation(self, ratio: float) -> str:
        if ratio > 1.1:
            return "Landscape"
        elif ratio < 0.9:
            return "Portrait"
        return "Square"
    
    def _get_ratio_string(self, w: int, h: int) -> str:
        g = gcd(w, h)
        return f"{w//g}:{h//g}"
    
    def _ratio_diff_pct(self, r1: float, r2: float) -> float:
        if r2 == 0:
            return 100.0
        return abs(r1 - r2) / r2 * 100
    
    def _parse_manual_ratio(self, ratio_str: str) -> tuple:
        """Parse ratio string like '3:2' or '16:9' to (ratio_str, ratio_val)"""
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
            # Only apply resolution checks if the limit is > 0 (0 = disabled)
            if min_res and min_res > 0 and img["max_dim"] < min_res:
                result["resolution_low"].append(img)
            elif max_res and max_res > 0 and img["max_dim"] > max_res:
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
        gr.Markdown(self.config.description)
        
        with gr.Accordion("Bucketing Settings", open=True):
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
            save_btn = gr.Button("Save Settings", variant="secondary", scale=0)
            analyze_btn = gr.Button("Analyze", variant="secondary", elem_id="bucket_analyze_btn")
            prune_btn = gr.Button("Prune to Balanced", variant="secondary", elem_id="bucket_prune_btn")
            organize_btn = gr.Button("Organize", variant="primary", elem_id="bucket_organize_btn")
        
        result_output = gr.HTML(value=self._empty_state())
        
        # Store for wire_events
        self._save_btn = save_btn
        
        inputs = [num_buckets, tolerance, max_per_bucket, manual_buckets, create_landscape, create_portrait, create_square,
                  min_res, max_res, unassigned_action, file_action, output_dir, result_output, analyze_btn, prune_btn, organize_btn]
        return (analyze_btn, inputs)
    
    def wire_events(self, app, run_button, inputs: list, gallery_output: gr.Gallery,
                    limit_count=None) -> None:
        import copy
        
        # Use colorama for consistent console colors
        try:
            from colorama import Fore, Style
            CYAN = Fore.CYAN
            GREEN = Fore.GREEN
            YELLOW = Fore.YELLOW
            RED = Fore.RED
            RESET = Style.RESET_ALL
        except ImportError:
            CYAN = GREEN = YELLOW = RED = RESET = ""
        
        num_buckets, tolerance, max_per_bucket, manual_buckets = inputs[0], inputs[1], inputs[2], inputs[3]
        create_l, create_p, create_s = inputs[4], inputs[5], inputs[6]
        min_res, max_res, unassigned_action, file_action, output_dir = inputs[7], inputs[8], inputs[9], inputs[10], inputs[11]
        result_output, analyze_btn, prune_btn, organize_btn = inputs[12], inputs[13], inputs[14], inputs[15]
        
        # Build all_inputs - append limit_count if provided
        all_inputs = [num_buckets, tolerance, max_per_bucket, manual_buckets, create_l, create_p, create_s, min_res, max_res, unassigned_action, file_action, output_dir]
        if limit_count is not None:
            all_inputs.append(limit_count)
        
        def _get_limited_dataset(limit_val):
            """Get dataset with limit applied."""
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
            """Extract and print summary from HTML result."""
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
            limit_val = args[-1] if limit_count is not None else None
            
            run_dataset, count = _get_limited_dataset(limit_val)
            if run_dataset is None:
                return self._empty_state()
            
            print(f"")
            print(f"{CYAN}--- Running Bucketing Tool (Analyze) on {count} images ---{RESET}")
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
                min_resolution=int(args[7]) if args[7] else 0,
                max_resolution=int(args[8]) if args[8] else 0,
                unassigned_action=args[9],
                file_action=args[10],
                output_dir=args[11],
                action="analyze"
            )
            
            _print_summary(result)
            print(f"{CYAN}--- Bucketing Tool (Analyze) Complete ---{RESET}")
            print(f"")
            return result
        
        def prune(*args):
            limit_val = args[-1] if limit_count is not None else None
            
            run_dataset, count = _get_limited_dataset(limit_val)
            if run_dataset is None:
                return self._empty_state()
            
            print(f"")
            print(f"{CYAN}--- Running Bucketing Tool (Prune) on {count} images ---{RESET}")
            
            result = self.apply_to_dataset(
                run_dataset,
                num_buckets=int(args[0]) if args[0] else 3,
                tolerance=float(args[1]) if args[1] else 20,
                max_per_bucket=int(args[2]) if args[2] else 0,
                manual_buckets=args[3],
                create_landscape=args[4],
                create_portrait=args[5],
                create_square=args[6],
                min_resolution=int(args[7]) if args[7] else 0,
                max_resolution=int(args[8]) if args[8] else 0,
                unassigned_action=args[9],
                file_action=args[10],
                output_dir=args[11],
                action="prune"
            )
            
            _print_summary(result)
            print(f"{CYAN}--- Bucketing Tool (Prune) Complete ---{RESET}")
            print(f"")
            return result
        
        def organize(*args):
            limit_val = args[-1] if limit_count is not None else None
            
            run_dataset, count = _get_limited_dataset(limit_val)
            if run_dataset is None:
                return self._empty_state()
            
            print(f"")
            print(f"{CYAN}--- Running Bucketing Tool (Organize) on {count} images ---{RESET}")
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
                min_resolution=int(args[7]) if args[7] else 0,
                max_resolution=int(args[8]) if args[8] else 0,
                unassigned_action=args[9],
                file_action=args[10],
                output_dir=args[11],
                action="organize"
            )
            
            _print_summary(result)
            print(f"{CYAN}--- Bucketing Tool (Organize) Complete ---{RESET}")
            print(f"")
            return result
        
        analyze_btn.click(fn=analyze, inputs=all_inputs, outputs=[result_output])
        prune_btn.click(fn=prune, inputs=all_inputs, outputs=[result_output])
        organize_btn.click(fn=organize, inputs=all_inputs, outputs=[result_output])
        
        # Save settings handler
        def save_settings(*args):
            from src.gui.constants import filter_user_overrides
            
            settings = {
                "num_buckets": args[0],
                "tolerance": args[1],
                "max_per_bucket": args[2],
                "manual_buckets": args[3],
                "create_landscape": args[4],
                "create_portrait": args[5],
                "create_square": args[6],
                "min_resolution": args[7],
                "max_resolution": args[8],
                "unassigned_action": args[9],
                "file_action": args[10],
                "output_dir": args[11],
            }
            
            try:
                if "tool_settings" not in app.config_mgr.user_config:
                    app.config_mgr.user_config["tool_settings"] = {}
                if "bucketing" not in app.config_mgr.user_config["tool_settings"]:
                    app.config_mgr.user_config["tool_settings"]["bucketing"] = {}
                
                defaults = self._get_defaults()
                
                for key, value in settings.items():
                    default_val = defaults.get(key)
                    if value != default_val:
                        app.config_mgr.user_config["tool_settings"]["bucketing"][key] = value
                    elif key in app.config_mgr.user_config["tool_settings"]["bucketing"]:
                        del app.config_mgr.user_config["tool_settings"]["bucketing"][key]
                
                if not app.config_mgr.user_config["tool_settings"]["bucketing"]:
                    del app.config_mgr.user_config["tool_settings"]["bucketing"]
                if not app.config_mgr.user_config.get("tool_settings"):
                    if "tool_settings" in app.config_mgr.user_config:
                        del app.config_mgr.user_config["tool_settings"]
                
                filtered = filter_user_overrides(app.config_mgr.user_config)
                app.config_mgr._save_yaml(app.config_mgr.user_config_path, filtered)
                gr.Info("Bucketing settings saved!")
            except Exception as e:
                gr.Warning(f"Failed to save settings: {e}")
        
        save_btn = self._save_btn
        settings_inputs = [num_buckets, tolerance, max_per_bucket, manual_buckets, create_l, create_p, create_s, min_res, max_res, unassigned_action, file_action, output_dir]
        save_btn.click(save_settings, inputs=settings_inputs, outputs=[])
        
        for c in [num_buckets, tolerance, max_per_bucket, manual_buckets, create_l, create_p, create_s, min_res, max_res]:
            c.change(fn=analyze, inputs=all_inputs, outputs=[result_output])


