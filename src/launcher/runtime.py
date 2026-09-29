"""Runtime paths and Windows process helpers for the server launcher.

Search keywords: LAUNCHER_RUNTIME, REPO_ROOT, HIDDEN_PROCESS
"""

from __future__ import annotations

import socket
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def get_repo_root() -> Path:
    """Return the project root for source and frozen execution."""
    if getattr(sys, "frozen", False):
        executable_dir = Path(sys.executable).resolve().parent
        for candidate in (executable_dir, *executable_dir.parents):
            if (candidate / "gui.py").is_file() and (candidate / "src").is_dir():
                return candidate
        return executable_dir
    return Path(__file__).resolve().parents[2]


def get_python_executable(repo_root: Path) -> Path:
    return repo_root / "venv" / "Scripts" / "python.exe"


def get_settings_path(repo_root: Path) -> Path:
    return repo_root / "user" / "launcher" / "settings.json"


def get_pid_path(repo_root: Path) -> Path:
    return repo_root / "user" / "launcher" / "server.pid"


def get_session_log_path(repo_root: Path) -> Path:
    stamp = datetime.now().strftime("%Y-%m-%d_%H.%M.%S")
    return repo_root / "user" / "launcher" / "logs" / f"session-{stamp}.log"


def get_icon_png_path(repo_root: Path) -> Path:
    """Return the source or bundled launcher PNG path."""
    bundled_root = Path(getattr(sys, "_MEIPASS", repo_root))
    bundled = bundled_root / "src" / "assets" / "server_launcher.png"
    if bundled.is_file():
        return bundled
    return repo_root / "src" / "assets" / "server_launcher.png"


def get_icon_ico_path(repo_root: Path) -> Path:
    """Return the Windows executable icon path used during builds."""
    return repo_root / "src" / "assets" / "server_launcher.ico"


def get_lan_ip() -> str:
    """Return the LAN-facing IPv4 address used for display."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.connect(("8.8.8.8", 80))
        return str(sock.getsockname()[0])
    except OSError:
        return "127.0.0.1"
    finally:
        sock.close()


def hidden_subprocess_kwargs() -> dict:
    """Return subprocess flags that prevent Windows console windows."""
    kwargs: dict = {}
    creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    if creationflags:
        kwargs["creationflags"] = creationflags
    return kwargs
