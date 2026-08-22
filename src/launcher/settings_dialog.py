"""Settings dialog for the server launcher.

Search keywords: LAUNCHER_SETTINGS_DIALOG, SERVER_PORT, CONSOLE_LIMIT
"""

from __future__ import annotations

from dataclasses import replace

from PyQt6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from .settings_store import LauncherSettings


class SettingsDialog(QDialog):
    """Edit launcher settings without mutating the active values."""

    def __init__(self, settings: LauncherSettings, parent: QWidget | None = None):
        super().__init__(parent)
        self._original = settings
        self.setWindowTitle("Launcher Settings")
        layout = QVBoxLayout(self)
        form = QFormLayout()

        self._port = QSpinBox()
        self._port.setRange(1, 65535)
        self._port.setValue(settings.server_port)
        self._port.setToolTip(
            "TCP port used by the web interface and REST API.\n"
            "Changing it takes effect the next time the server starts."
        )
        self._auto_start = QCheckBox("Start server when the launcher opens")
        self._auto_start.setChecked(settings.auto_start_server)
        self._auto_start.setToolTip(
            "Start the server automatically after the launcher window and tray icon are ready."
        )
        self._font_size = QSpinBox()
        self._font_size.setRange(8, 32)
        self._font_size.setValue(settings.console_font_size)
        self._font_size.setToolTip(
            "Text size for the launcher console. Hold Ctrl and use the mouse wheel to adjust it."
        )
        self._line_limit = QSpinBox()
        self._line_limit.setRange(200, 200000)
        self._line_limit.setSingleStep(500)
        self._line_limit.setValue(settings.console_line_limit)
        self._line_limit.setToolTip(
            "Maximum lines retained in the visible console.\n"
            "The complete unmodified output remains available in the session log."
        )

        form.addRow("Server port", self._port)
        form.addRow("Console text size", self._font_size)
        form.addRow("Console line limit", self._line_limit)
        form.addRow("", self._auto_start)
        layout.addLayout(form)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        ok_button = buttons.button(QDialogButtonBox.StandardButton.Ok)
        cancel_button = buttons.button(QDialogButtonBox.StandardButton.Cancel)
        ok_button.setToolTip("Save these launcher settings.")
        cancel_button.setToolTip("Close without changing launcher settings.")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def get_settings(self) -> LauncherSettings:
        return replace(
            self._original,
            server_port=self._port.value(),
            auto_start_server=self._auto_start.isChecked(),
            console_font_size=self._font_size.value(),
            console_line_limit=self._line_limit.value(),
        )
