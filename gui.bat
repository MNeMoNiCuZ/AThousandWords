@echo off
setlocal enabledelayedexpansion

:: Check if venv exists
if not exist "venv\Scripts\activate.bat" (
    echo [ERROR] Virtual environment not found!
    echo.
    echo The venv folder does not exist or is not properly set up.
    echo Please create a virtual environment first by running:
    echo   python -m venv venv
    echo.
    pause
    exit /b 1
)

:: Activate venv
call venv\Scripts\activate >nul 2>&1

:: Run pre-launch validation script (silent on success)
python validate_environment.py
if not errorlevel 1 goto launch

:: Validation failed - show warning
echo.
echo ========================================
echo Continue anyway? (Press any key)
echo Or press Ctrl+C to abort and fix issues
echo ========================================
pause >nul
if errorlevel 1 exit /b 1

:launch
:: Launch the application
cmd /k "cd /d %~dp0 & call venv\Scripts\activate & python gui.py"
