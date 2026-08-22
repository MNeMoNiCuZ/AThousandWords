"""Persisted settings for the server launcher.

Search keywords: LAUNCHER_SETTINGS, WINDOW_GEOMETRY, SERVER_PORT
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class LauncherSettings:
    server_port: int = 8585
    auto_start_server: bool = True
    console_font_size: int = 11
    console_line_limit: int = 5000
    window_x: int = -1
    window_y: int = -1
    window_width: int = -1
    window_height: int = -1
    window_maximized: bool = False


def clamp_geometry(
    x: int,
    y: int,
    width: int,
    height: int,
    screen_x: int,
    screen_y: int,
    screen_width: int,
    screen_height: int,
) -> tuple[int, int, int, int]:
    width = max(320, min(int(width), int(screen_width)))
    height = max(240, min(int(height), int(screen_height)))
    x = max(screen_x, min(int(x), screen_x + screen_width - width))
    y = max(screen_y, min(int(y), screen_y + screen_height - height))
    return x, y, width, height


class LauncherSettingsStore:
    """Load and save validated launcher settings as UTF-8 JSON."""

    def __init__(self, path: Path):
        self._path = path

    def load(self) -> LauncherSettings:
        try:
            payload = json.loads(self._path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, TypeError):
            return LauncherSettings()

        def number(name: str, default: int) -> int:
            try:
                return int(payload.get(name, default))
            except (TypeError, ValueError):
                return default

        return LauncherSettings(
            server_port=max(1, min(number("server_port", 8585), 65535)),
            auto_start_server=bool(payload.get("auto_start_server", True)),
            console_font_size=max(8, min(number("console_font_size", 11), 32)),
            console_line_limit=max(200, min(number("console_line_limit", 5000), 200000)),
            window_x=number("window_x", -1),
            window_y=number("window_y", -1),
            window_width=number("window_width", -1),
            window_height=number("window_height", -1),
            window_maximized=bool(payload.get("window_maximized", False)),
        )

    def save(self, settings: LauncherSettings) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(
            json.dumps(asdict(settings), indent=2) + "\n",
            encoding="utf-8",
        )
