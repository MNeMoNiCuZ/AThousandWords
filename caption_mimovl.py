"""Caption every image in ./input with MimoVL via the running API (localhost:8585).

Usage:  python caption_mimovl.py
Writes a .txt caption next to each image and prints the results.
"""

from pathlib import Path
import sys
import requests

API = "http://localhost:8585/api/caption"
MODEL = "mimovl"
INPUT_DIR = Path(__file__).parent / "input"
EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}

images = sorted(p for p in INPUT_DIR.rglob("*") if p.suffix.lower() in EXTS)
if not images:
    print(f"No images found in {INPUT_DIR}")
    sys.exit(1)

print(f"Captioning {len(images)} image(s) with '{MODEL}'...\n")

for img in images:
    with open(img, "rb") as f:
        resp = requests.post(
            API,
            files={"files": (img.name, f)},
            data={"model": MODEL},
            timeout=600,
        )
    if resp.status_code != 200:
        print(f"[FAIL] {img.name}: {resp.status_code} {resp.text}")
        continue

    caption = resp.json().get("results", [{}])[0].get("caption", "").strip()
    img.with_suffix(".txt").write_text(caption, encoding="utf-8")
    print(f"[OK] {img.name}\n{caption}\n")

print("Done.")
