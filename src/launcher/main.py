"""Application entry point for the A Thousand Words server launcher.

Search keywords: LAUNCHER_ENTRYPOINT, QAPPLICATION, START_MINIMIZED
"""

from __future__ import annotations

import ctypes
import sys
from pathlib import Path

from PyQt6.QtWidgets import QApplication, QMessageBox, QSystemTrayIcon

REPO_ROOT = Path(__file__).resolve().parents[2]
if not getattr(sys, "frozen", False) and str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.launcher.process_manager import ServerProcessManager
from src.launcher.runtime import get_repo_root, get_settings_path
from src.launcher.settings_store import LauncherSettingsStore
from src.launcher.ui import LauncherWindow, create_launcher_icon


def _set_windows_app_id() -> None:
    if sys.platform == "win32":
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(
            "AThousandWords.ServerLauncher"
        )


def main() -> int:
    repo_root = get_repo_root()
    settings_store = LauncherSettingsStore(get_settings_path(repo_root))
    settings = settings_store.load()
    _set_windows_app_id()
    app = QApplication(sys.argv)
    app.setApplicationName("A Thousand Words Server")
    app.setQuitOnLastWindowClosed(True)
    app.setWindowIcon(create_launcher_icon(repo_root))
    if not QSystemTrayIcon.isSystemTrayAvailable():
        QMessageBox.critical(
            None,
            "System tray unavailable",
            "A system tray is required to run the server launcher.",
        )
        return 1
    manager = ServerProcessManager(repo_root, settings.server_port)
    window = LauncherWindow(repo_root, settings_store, settings, manager)
    if "--start-minimized" in sys.argv:
        window.start_in_tray()
    else:
        window.show_restored()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
