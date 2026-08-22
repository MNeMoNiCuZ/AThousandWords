@echo off
setlocal
cd /d "%~dp0"

if not exist "venv\Scripts\python.exe" (
    echo ERROR: Virtual environment interpreter not found: venv\Scripts\python.exe
    echo Run setup.bat before building the launcher.
    exit /b 1
)

"venv\Scripts\python.exe" -u "src\launcher\build_launcher.py" %*
exit /b %ERRORLEVEL%
