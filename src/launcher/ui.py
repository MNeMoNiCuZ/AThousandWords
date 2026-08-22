"""Main window and system tray interface for the server launcher.

Search keywords: LAUNCHER_WINDOW, SYSTEM_TRAY, SERVER_CONTROLS
"""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

from PyQt6.QtCore import QTimer, Qt
from PyQt6.QtGui import QAction, QIcon
from PyQt6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMenu,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QSystemTrayIcon,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from .console_mixin import ConsoleMixin
from .geometry_mixin import GeometryMixin
from .process_manager import ServerProcessManager
from .runtime import get_icon_png_path, get_session_log_path
from .settings_dialog import SettingsDialog
from .settings_store import LauncherSettings, LauncherSettingsStore


def create_launcher_icon(repo_root: Path) -> QIcon:
    """Load the dedicated launcher icon from source or bundled assets."""
    return QIcon(str(get_icon_png_path(repo_root)))


class LauncherWindow(GeometryMixin, ConsoleMixin, QMainWindow):
    """Control the project server from a window or system tray icon."""

    def __init__(
        self,
        repo_root: Path,
        settings_store: LauncherSettingsStore,
        settings: LauncherSettings,
        process_manager: ServerProcessManager,
    ):
        super().__init__()
        self._repo_root = repo_root
        self._settings_store = settings_store
        self._settings = settings
        self._process_manager = process_manager
        self._ansi_state: dict[str, str | bool] = {"color": "", "bold": False}
        self._log_buffer: list[str] = []
        self._log_follow_tail = True
        self._log_follow_tail_sync_suspended = False
        self._log_flush_timer = QTimer(self)
        self._log_flush_timer.setInterval(40)
        self._log_flush_timer.timeout.connect(self._flush_log_buffer)
        self._session_log_path = get_session_log_path(repo_root)
        try:
            self._session_log_path.parent.mkdir(parents=True, exist_ok=True)
            self._session_log_file = self._session_log_path.open("a", encoding="utf-8")
        except OSError:
            self._session_log_file = None

        self._window_icon = create_launcher_icon(repo_root)
        self._tray_icon = QSystemTrayIcon(self)
        self._was_maximized = False
        self.setWindowTitle("A Thousand Words Server")
        self.setWindowIcon(self._window_icon)
        self._restore_geometry()
        self._build_ui()
        self._build_tray()
        self._process_manager.log_ready.connect(self._append_log)
        self._process_manager.state_changed.connect(self._update_status)
        self._update_status("stopped")
        self._append_log(
            f"[launcher] Session started {datetime.now().strftime('%Y-%m-%d - %H.%M.%S')}\n"
        )
        if settings.auto_start_server:
            QTimer.singleShot(150, self._start_server)

    def _build_ui(self) -> None:
        container = QWidget(self)
        self.setCentralWidget(container)
        layout = QVBoxLayout(container)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(10)

        toolbar = QHBoxLayout()
        toolbar.setSpacing(8)
        self._start_button = self._button(
            "Start", "Start the server on the configured port.", self._start_server
        )
        self._restart_button = self._button(
            "Restart", "Stop the current server process and start a new one.", self._restart_server
        )
        self._stop_button = self._button(
            "Stop", "Stop the server and its child processes.", self._stop_server
        )
        open_button = self._button(
            "Open", "Open the local server interface in the default browser.", self._open_server
        )
        clear_button = self._button(
            "Clear", "Clear the visible console without deleting the session log.", self._clear_log
        )
        settings_button = self._button(
            "Settings", "Configure the server port and launcher console.", self._open_settings
        )
        for button in (
            self._start_button,
            self._restart_button,
            self._stop_button,
            open_button,
            clear_button,
            settings_button,
        ):
            toolbar.addWidget(button)
        toolbar.addStretch(1)

        info = QVBoxLayout()
        info.setSpacing(2)
        self._status_label = QLabel("Stopped")
        self._status_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        self._status_label.setToolTip("Current state of the supervised server process.")
        self._log_link = QLabel(
            '<a href="open-log" style="color:#94a3b8; text-decoration:underline;">'
            f"{self._session_log_path.name}</a>"
        )
        self._log_link.setAlignment(Qt.AlignmentFlag.AlignRight)
        self._log_link.setToolTip(
            "Open the complete UTF-8 session log in the default text editor.\n"
            f"Location: {self._session_log_path}"
        )
        self._log_link.linkActivated.connect(self._open_log)
        info.addWidget(self._status_label)
        info.addWidget(self._log_link)
        toolbar.addLayout(info)
        layout.addLayout(toolbar)

        self._log_view = QTextEdit(self)
        self._log_view.setReadOnly(True)
        self._log_view.setLineWrapMode(QTextEdit.LineWrapMode.WidgetWidth)
        self._log_view.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._log_view.setToolTip(
            "Live server output. Scroll up to pause follow mode.\n"
            "Hold Ctrl and use the mouse wheel to change text size."
        )
        self._log_view.installEventFilter(self)
        self._log_view.verticalScrollBar().valueChanged.connect(self._sync_follow_tail)
        self._log_view.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self._log_view.customContextMenuRequested.connect(self._show_console_menu)
        layout.addWidget(self._log_view, 1)
        self._apply_console_font_size(self._settings.console_font_size)
        self.setStyleSheet(
            """
            QMainWindow, QWidget { background-color:#111827; color:#e5e7eb; }
            QTextEdit {
                background-color:#000000; border:1px solid #334155; border-radius:8px;
                color:#cccccc; font-family:Consolas, "Courier New", monospace;
                font-weight:600; padding:6px;
            }
            QPushButton {
                background-color:#1f2937; border:1px solid #334155; border-radius:7px;
                color:#f8fafc; padding:7px 11px;
            }
            QPushButton:hover { background-color:#273449; }
            QPushButton:disabled { color:#64748b; }
            QMenu { background-color:#1f2937; color:#f8fafc; border:1px solid #334155; }
            QMenu::item { padding:6px 24px 6px 12px; }
            QMenu::item:selected { background-color:#273449; }
            QDialog, QSpinBox, QCheckBox { background-color:#111827; color:#e5e7eb; }
            QSpinBox { border:1px solid #334155; border-radius:6px; padding:5px; }
            """
        )

    def _button(self, text: str, tooltip: str, callback) -> QPushButton:
        button = QPushButton(text)
        button.setToolTip(tooltip)
        button.clicked.connect(callback)
        return button

    def _build_tray(self) -> None:
        menu = QMenu(self)
        restore_action = self._tray_action(
            "Restore", "Show the launcher window.", self.show_restored
        )
        start_action = self._tray_action(
            "Start", "Start the server on the configured port.", self._start_server
        )
        restart_action = self._tray_action(
            "Restart", "Restart the supervised server process.", self._restart_server
        )
        stop_action = self._tray_action(
            "Stop", "Stop the server and its child processes.", self._stop_server
        )
        quit_action = self._tray_action(
            "Quit", "Stop the server and close the launcher.", self._quit_from_tray
        )
        menu.addAction(restore_action)
        menu.addSeparator()
        menu.addAction(start_action)
        menu.addAction(restart_action)
        menu.addAction(stop_action)
        menu.addSeparator()
        menu.addAction(quit_action)
        self._tray_start_action = start_action
        self._tray_restart_action = restart_action
        self._tray_stop_action = stop_action
        self._tray_icon.setIcon(self._window_icon)
        self._tray_icon.setToolTip("A Thousand Words Server")
        self._tray_icon.setContextMenu(menu)
        self._tray_icon.activated.connect(self._on_tray_activated)
        self._tray_icon.show()

    def _tray_action(self, text: str, tooltip: str, callback) -> QAction:
        action = QAction(text, self)
        action.setToolTip(tooltip)
        action.triggered.connect(callback)
        return action

    def _update_status(self, state: str) -> None:
        running = state == "running"
        self._status_label.setText(f"Running on port {self._process_manager.port}" if running else "Stopped")
        self._status_label.setStyleSheet(f"color:{'#63e6be' if running else '#ff8787'}; font-weight:700;")
        self._start_button.setEnabled(not running)
        self._restart_button.setEnabled(running)
        self._stop_button.setEnabled(running)
        self._tray_start_action.setEnabled(not running)
        self._tray_restart_action.setEnabled(running)
        self._tray_stop_action.setEnabled(running)

    def _start_server(self) -> None:
        self._process_manager.set_port(self._settings.server_port)
        self._process_manager.start()

    def _restart_server(self) -> None:
        self._process_manager.set_port(self._settings.server_port)
        self._process_manager.restart()

    def _stop_server(self) -> None:
        self._process_manager.stop()

    def _open_server(self) -> None:
        os.startfile(f"http://127.0.0.1:{self._process_manager.port}")

    def _open_log(self, _link: str) -> None:
        handle = getattr(self, "_session_log_file", None)
        if handle is not None:
            handle.flush()
        if self._session_log_path.exists():
            os.startfile(str(self._session_log_path))

    def _open_settings(self) -> None:
        dialog = SettingsDialog(self._settings, self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        previous_port = self._settings.server_port
        self._settings = dialog.get_settings()
        self._settings_store.save(self._settings)
        self._apply_console_font_size(self._settings.console_font_size)
        if self._process_manager.is_running() and previous_port != self._settings.server_port:
            QMessageBox.information(
                self,
                "Server restart required",
                "Restart the server to use the new port.",
            )
        elif not self._process_manager.is_running():
            self._process_manager.set_port(self._settings.server_port)
