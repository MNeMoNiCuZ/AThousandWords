"""Server process supervision and output forwarding.

Search keywords: SERVER_PROCESS, PROCESS_TREE, CONSOLE_FORWARDING
"""

from __future__ import annotations

import os
import subprocess
import threading
import time
from pathlib import Path

import psutil
from PyQt6.QtCore import QObject, pyqtSignal

from .runtime import get_lan_ip, get_pid_path, get_python_executable


class ServerProcessManager(QObject):
    """Start, stop, and stream output from the project server."""

    log_ready = pyqtSignal(str)
    state_changed = pyqtSignal(str)

    def __init__(self, repo_root: Path, port: int):
        super().__init__()
        self._repo_root = repo_root
        self._port = port
        self._process: subprocess.Popen[str] | None = None
        self._reader_thread: threading.Thread | None = None
        self._started_at: float | None = None

    @property
    def port(self) -> int:
        return self._port

    def set_port(self, port: int) -> None:
        self._port = int(port)

    def started_at(self) -> float | None:
        return self._started_at if self.is_running() else None

    def is_running(self) -> bool:
        return self._process is not None and self._process.poll() is None

    def start(self) -> bool:
        if self.is_running():
            return False
        python_exe = get_python_executable(self._repo_root)
        if not python_exe.exists():
            self.log_ready.emit(f"ERROR: Virtual environment interpreter not found: {python_exe}\n")
            self.state_changed.emit("stopped")
            return False

        environment = os.environ.copy()
        environment.update(
            PYTHONIOENCODING="utf-8",
            PYTHONUTF8="1",
            PYTHONUNBUFFERED="1",
            ATW_LAUNCHER_CONSOLE="1",
            FORCE_COLOR="1",
        )
        command = [
            str(python_exe),
            "-u",
            "gui.py",
            "--server",
            "--port",
            str(self._port),
        ]
        try:
            process = subprocess.Popen(
                command,
                cwd=self._repo_root,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                env=environment,
            )
        except OSError as exc:
            self.log_ready.emit(f"ERROR: Server failed to start: {exc}\n")
            self.state_changed.emit("stopped")
            return False

        self._process = process
        self._started_at = time.time()
        self._write_pid(process.pid)
        self.log_ready.emit(
            f"\x1b[1m[launcher] Server: \x1b[36mhttp://127.0.0.1:{self._port}\x1b[0m\n"
            f"\x1b[1m[launcher] Network: \x1b[36mhttp://{get_lan_ip()}:{self._port}\x1b[0m\n\n"
        )
        self.state_changed.emit("running")
        self._reader_thread = threading.Thread(
            target=self._read_output,
            args=(process,),
            name="server-log-reader",
            daemon=True,
        )
        self._reader_thread.start()
        return True

    def stop(self) -> bool:
        process = self._process
        if process is None or process.poll() is not None:
            self._process = None
            self._started_at = None
            self._clear_pid()
            self.state_changed.emit("stopped")
            return False
        self._terminate_tree(process.pid)
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self._terminate_tree(process.pid, force=True)
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass
        if self._process is process:
            self._process = None
            self._started_at = None
            self._clear_pid()
            self.state_changed.emit("stopped")
            self.log_ready.emit("\n[launcher] Server stopped.\n")
        return True

    def restart(self) -> None:
        self.stop()
        self.start()

    def _read_output(self, process: subprocess.Popen[str]) -> None:
        if process.stdout is None:
            return
        for line in process.stdout:
            self.log_ready.emit(line)
        return_code = process.wait()
        if self._process is process:
            self._process = None
            self._started_at = None
            self._clear_pid()
            self.state_changed.emit("stopped")
            if return_code != 0:
                self.log_ready.emit(f"ERROR: Server exited with code {return_code}.\n")
            self.log_ready.emit("\n[launcher] Server stopped.\n")

    def _pid_path(self) -> Path:
        return get_pid_path(self._repo_root)

    def _write_pid(self, pid: int) -> None:
        try:
            path = self._pid_path()
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(str(pid), encoding="utf-8")
        except OSError:
            pass

    def _clear_pid(self) -> None:
        try:
            self._pid_path().unlink(missing_ok=True)
        except OSError:
            pass

    @staticmethod
    def _terminate_tree(pid: int, force: bool = False) -> None:
        try:
            parent = psutil.Process(pid)
        except psutil.Error:
            return
        processes = parent.children(recursive=True) + [parent]
        for process in processes:
            try:
                process.kill() if force else process.terminate()
            except psutil.Error:
                pass
        _, alive = psutil.wait_procs(processes, timeout=5)
        if not force:
            for process in alive:
                try:
                    process.kill()
                except psutil.Error:
                    pass
