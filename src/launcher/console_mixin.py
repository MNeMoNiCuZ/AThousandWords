"""Buffered console rendering for the server launcher.

Search keywords: CONSOLE_BUFFER, FOLLOW_TAIL, SESSION_LOG
"""

from __future__ import annotations

from PyQt6.QtCore import QEvent, QPoint, Qt
from PyQt6.QtGui import QAction, QTextCursor, QWheelEvent

from .ansi_html import ansi_text_to_html


class ConsoleMixin:
    """Manage the rolling console view and complete session log."""

    def _apply_console_font_size(self, size: int) -> None:
        font = self._log_view.font()
        font.setPointSize(size)
        self._log_view.setFont(font)

    def eventFilter(self, watched, event) -> bool:
        if (
            watched is self._log_view
            and event.type() == QEvent.Type.Wheel
            and isinstance(event, QWheelEvent)
            and event.modifiers() & Qt.KeyboardModifier.ControlModifier
        ):
            delta = 1 if event.angleDelta().y() > 0 else -1
            size = max(8, min(self._settings.console_font_size + delta, 32))
            if size != self._settings.console_font_size:
                self._settings.console_font_size = size
                self._settings_store.save(self._settings)
                self._apply_console_font_size(size)
            return True
        return super().eventFilter(watched, event)

    def _append_log(self, text: str) -> None:
        if not text:
            return
        self._write_session_log(text)
        self._log_buffer.append(text)
        if not self._log_flush_timer.isActive():
            self._log_flush_timer.start()

    def _write_session_log(self, text: str) -> None:
        handle = getattr(self, "_session_log_file", None)
        if handle is None:
            return
        try:
            handle.write(text)
            handle.flush()
        except OSError:
            self._session_log_file = None

    def _close_session_log(self) -> None:
        handle = getattr(self, "_session_log_file", None)
        if handle is not None:
            try:
                handle.close()
            except OSError:
                pass
            self._session_log_file = None

    def _flush_log_buffer(self) -> None:
        if not self._log_buffer:
            self._log_flush_timer.stop()
            return
        pending = "".join(self._log_buffer)
        self._log_buffer.clear()
        scrollbar = self._log_view.verticalScrollBar()
        follow_tail = self._log_follow_tail
        previous_value = scrollbar.value()
        cursor = QTextCursor(self._log_view.document())
        cursor.movePosition(QTextCursor.MoveOperation.End)
        segments = pending.split("\n")
        for index, segment in enumerate(segments):
            if segment:
                cursor.insertHtml(ansi_text_to_html(segment, self._ansi_state))
            if index < len(segments) - 1:
                cursor.insertBlock()
        self._trim_console()
        if follow_tail:
            self._scroll_to_bottom()
        elif scrollbar.value() != previous_value:
            self._log_follow_tail_sync_suspended = True
            scrollbar.setValue(previous_value)
            self._log_follow_tail_sync_suspended = False

    def _trim_console(self) -> None:
        document = self._log_view.document()
        excess = document.blockCount() - self._settings.console_line_limit
        if excess <= 0:
            return
        cursor = QTextCursor(document)
        cursor.movePosition(QTextCursor.MoveOperation.Start)
        cursor.movePosition(
            QTextCursor.MoveOperation.NextBlock,
            QTextCursor.MoveMode.KeepAnchor,
            excess,
        )
        cursor.removeSelectedText()

    def _scroll_to_bottom(self) -> None:
        scrollbar = self._log_view.verticalScrollBar()
        self._log_follow_tail_sync_suspended = True
        scrollbar.setValue(scrollbar.maximum())
        self._log_follow_tail = True
        self._log_follow_tail_sync_suspended = False

    def _sync_follow_tail(self) -> None:
        if self._log_follow_tail_sync_suspended:
            return
        scrollbar = self._log_view.verticalScrollBar()
        self._log_follow_tail = scrollbar.value() >= max(0, scrollbar.maximum() - 4)

    def _show_console_menu(self, position: QPoint) -> None:
        menu = self._log_view.createStandardContextMenu()
        menu.addSeparator()
        action = QAction("Clear", menu)
        action.setToolTip("Clear the visible console without deleting the session log.")
        action.triggered.connect(self._clear_log)
        menu.addAction(action)
        menu.exec(self._log_view.viewport().mapToGlobal(position))

    def _clear_log(self) -> None:
        self._log_buffer.clear()
        self._log_flush_timer.stop()
        self._log_view.clear()
        self._ansi_state = {"color": "", "bold": False}
