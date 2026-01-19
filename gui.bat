@echo off
cd /d "%~dp0"

REM Check if venv exists
if not exist "venv" (
    echo.
    echo ===============================================================================
    echo  ERROR: Virtual environment 'venv' not found!
    echo.
    echo  Please run 'setup.bat' first to install the application.
    echo ===============================================================================
    echo.
    pause
    exit /b 1
)

call venv\Scripts\activate.bat

REM Run pre-launch validation script (silent on success)
python src/core/validate_environment.py
if errorlevel 1 (
    echo.
    echo ========================================
    echo Validation warnings detected.
    echo Continue anyway? (Press any key)
    echo Or press Ctrl+C to abort and fix issues.
    echo ========================================
    pause >nul
)

echo.
echo Launching A Thousand Words GUI...
echo Usage: gui.bat [options]
echo   --server       : Enable network access (0.0.0.0)
echo   --enable-api   : Enable REST API endpoint
echo.

python gui.py %*

if %errorlevel% neq 0 (
    echo.
    echo Process exited with error code %errorlevel%.
    pause
)
pause
