"""Window geometry and system tray lifecycle behavior.

Search keywords: WINDOW_GEOMETRY, MINIMIZE_TO_TRAY, TRAY_RESTORE
"""

from __future__ import annotations

from PyQt6.QtCore import QEvent, QPoint, Qt, QTimer
from PyQt6.QtWidgets import QApplication, QSystemTrayIcon

from .settings_store import clamp_geometry


class GeometryMixin:
    """Restore, save, minimize, and close the launcher window."""

    def _restore_geometry(self) -> None:
        settings = self._settings
        if settings.window_width > 0 and settings.window_height > 0:
            center = QPoint(
                settings.window_x + settings.window_width // 2,
                settings.window_y + settings.window_height // 2,
            )
            screen = QApplication.screenAt(center) or QApplication.primaryScreen()
            if screen is not None:
                area = screen.availableGeometry()
                geometry = clamp_geometry(
                    settings.window_x,
                    settings.window_y,
                    settings.window_width,
                    settings.window_height,
                    area.x(),
                    area.y(),
                    area.width(),
                    area.height(),
                )
                self.setGeometry(*geometry)
            self._was_maximized = settings.window_maximized
            return
        screen = QApplication.primaryScreen()
        if screen is None:
            self.setGeometry(100, 100, 1000, 620)
            return
        area = screen.availableGeometry()
        width, height = min(1000, area.width()), min(620, area.height())
        self.setGeometry(
            area.x() + (area.width() - width) // 2,
            area.y() + (area.height() - height) // 2,
            width,
            height,
        )

    def _save_geometry(self) -> None:
        if not self.isMinimized():
            self._was_maximized = bool(self.windowState() & Qt.WindowState.WindowMaximized)
        rect = self.normalGeometry()
        if rect.width() <= 0 or rect.height() <= 0:
            rect = self.geometry()
        self._settings.window_x = rect.x()
        self._settings.window_y = rect.y()
        self._settings.window_width = rect.width()
        self._settings.window_height = rect.height()
        self._settings.window_maximized = self._was_maximized
        self._settings_store.save(self._settings)

    def show_restored(self) -> None:
        if self._was_maximized:
            self.showMaximized()
        else:
            self.showNormal()
        self.raise_()
        self.activateWindow()

    def start_in_tray(self) -> None:
        self.setWindowState(self.windowState() | Qt.WindowState.WindowMinimized)
        self.hide()

    def _on_tray_activated(self, reason: QSystemTrayIcon.ActivationReason) -> None:
        if reason in {
            QSystemTrayIcon.ActivationReason.Trigger,
            QSystemTrayIcon.ActivationReason.DoubleClick,
        }:
            self.show_restored()

    def changeEvent(self, event) -> None:
        if event.type() == QEvent.Type.WindowStateChange:
            if self.isMinimized():
                self._was_maximized = bool(
                    (self.windowState() | event.oldState()) & Qt.WindowState.WindowMaximized
                )
                self._save_geometry()
                QTimer.singleShot(0, self.hide)
                self._tray_icon.showMessage(
                    "A Thousand Words",
                    "Server launcher minimized to the system tray.",
                    self._window_icon,
                    1500,
                )
            else:
                self._save_geometry()
        super().changeEvent(event)

    def closeEvent(self, event) -> None:
        self._save_geometry()
        self._tray_icon.hide()
        self._process_manager.stop()
        self._close_session_log()
        super().closeEvent(event)

    def _quit_from_tray(self) -> None:
        self.close()
