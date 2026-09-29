"""Build the windowed server launcher with PyInstaller.

Search keywords: LAUNCHER_BUILD, PYINSTALLER, WINDOWED_EXECUTABLE
"""

from __future__ import annotations

import argparse
import importlib.util
import shutil
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
EXE_NAME = "A Thousand Words"
CONTENTS_DIR_NAME = "_internal"


def _emit(message: str) -> None:
    print(message, flush=True)


def _check_dependencies() -> bool:
    missing = [
        name
        for name in ("PyQt6", "psutil", "PyInstaller")
        if importlib.util.find_spec(name) is None
    ]
    if not missing:
        return True
    _emit(f"ERROR: Missing launcher dependencies: {', '.join(missing)}")
    _emit("Install the updated requirements.txt in the project virtual environment, then run build.bat again.")
    return False


def build(keep_build: bool = False) -> int:
    if not _check_dependencies():
        return 1
    temp_root = REPO_ROOT / "temp" / "launcher_build"
    dist_root = temp_root / "dist"
    work_root = temp_root / "work"
    spec_root = temp_root / "spec"
    source_bundle = dist_root / EXE_NAME
    source_exe = source_bundle / f"{EXE_NAME}.exe"
    source_contents = source_bundle / CONTENTS_DIR_NAME
    target_exe = REPO_ROOT / f"{EXE_NAME}.exe"
    target_contents = REPO_ROOT / CONTENTS_DIR_NAME
    legacy_target_bundle = REPO_ROOT / "launcher"
    icon_png = REPO_ROOT / "src" / "assets" / "server_launcher.png"
    icon_ico = REPO_ROOT / "src" / "assets" / "server_launcher.ico"
    if not icon_png.is_file() or not icon_ico.is_file():
        _emit("ERROR: Launcher PNG or ICO asset is missing from src\\assets.")
        return 1
    if temp_root.exists():
        shutil.rmtree(temp_root)
    dist_root.mkdir(parents=True, exist_ok=True)
    work_root.mkdir(parents=True, exist_ok=True)
    spec_root.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-m",
        "PyInstaller",
        "--noconfirm",
        "--clean",
        "--onedir",
        "--contents-directory",
        CONTENTS_DIR_NAME,
        "--windowed",
        "--name",
        EXE_NAME,
        "--icon",
        str(icon_ico),
        "--add-data",
        f"{icon_png};src\\assets",
        "--distpath",
        str(dist_root),
        "--workpath",
        str(work_root),
        "--specpath",
        str(spec_root),
        str(REPO_ROOT / "src" / "launcher" / "main.py"),
    ]
    _emit("Building A Thousand Words server launcher...")
    result = subprocess.run(command, cwd=REPO_ROOT, check=False)
    if result.returncode != 0:
        _emit(f"ERROR: PyInstaller exited with code {result.returncode}.")
        return int(result.returncode)
    if not source_exe.is_file() or not source_contents.is_dir():
        _emit(f"ERROR: Built executable was not found in {source_bundle}.")
        return 1
    if keep_build:
        _emit(f"Created {source_exe}")
        _emit("Temporary build content was preserved because --keep-build was used.")
        return 0
    try:
        if target_contents.exists():
            shutil.rmtree(target_contents)
        shutil.copytree(source_contents, target_contents)
        shutil.copyfile(source_exe, target_exe)
        if legacy_target_bundle.exists():
            shutil.rmtree(legacy_target_bundle)
    except OSError as exc:
        _emit(f"ERROR: Could not place the launcher in the project root: {exc}")
        _emit(f"The completed build remains available at {source_bundle}.")
        return 1
    shutil.rmtree(temp_root)
    _emit(f"Created {target_exe}")
    _emit(f"Created {target_contents}")
    _emit("Removed legacy launcher bundle if present.")
    _emit("Temporary build content removed.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--keep-build",
        action="store_true",
        help="Keep the compiled bundle under temp instead of placing it in the project root.",
    )
    args = parser.parse_args()
    return build(keep_build=args.keep_build)


if __name__ == "__main__":
    raise SystemExit(main())
