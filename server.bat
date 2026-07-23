@echo off
cd /d "%~dp0"
call gui.bat --server --port 8585 %*
