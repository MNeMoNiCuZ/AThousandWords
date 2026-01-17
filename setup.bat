@echo off
setlocal enabledelayedexpansion

echo ================================================================
echo   A Thousand Words - Environment Setup
echo ================================================================
echo.

:: Check for path issues with "!" character
setlocal disabledelayedexpansion
echo Setting up virtual environment in: %CD%
set "CURRENT_PATH=%CD%"
set "MODIFIED_PATH=%CURRENT_PATH:!=%"
if not "%CURRENT_PATH%"=="%MODIFIED_PATH%" (
    echo.
    echo WARNING: The current directory contains a "!" character.
    echo This may cause issues with pip installations.
    echo Consider moving to a different directory.
    echo.
)
endlocal
setlocal enabledelayedexpansion

:: Initialize counter for Python versions
set COUNT=0

:: Parse Python versions from py launcher
for /f "tokens=1,*" %%a in ('py -0p 2^>nul') do (
    echo %%a | findstr /R "^[ ]*-" > nul && (
        set /a COUNT+=1
        set "pythonVersion=%%a"
        set "pythonVersion=!pythonVersion:-32=!"
        set "pythonVersion=!pythonVersion:-64=!"
        set "pythonVersion=!pythonVersion:-=!"
        set "pythonVersion=!pythonVersion:V:=!"
        set "PYTHON_VER_!COUNT!=!pythonVersion!"
        set "PYTHON_PATH_!COUNT!=%%b"
    )
)

:: Make sure at least one Python version was found
if %COUNT%==0 (
    echo ERROR: No Python installations found via Python Launcher.
    echo Please install Python 3.10+ and ensure 'py' launcher is available.
    pause
    goto :eof
)

echo.
echo --- Python Version ---
echo Please choose which Python version to use:
for /L %%i in (1,1,%COUNT%) do (
    echo %%i. -V:!PYTHON_VER_%%i! at !PYTHON_PATH_%%i!
)
echo.

:: Prompt user to select a Python version (default is 1)
set /p PYTHON_SELECTION="Select a Python version by number [default=1]: "
if "!PYTHON_SELECTION!"=="" set PYTHON_SELECTION=1

:: Extract the selected Python version tag
set SELECTED_PYTHON_VER=!PYTHON_VER_%PYTHON_SELECTION%!
echo.
echo Using Python version %SELECTED_PYTHON_VER%

:: Create the virtual environment
echo.
echo --- Creating Virtual Environment ---
echo Creating virtual environment named 'venv'...
py -%SELECTED_PYTHON_VER% -m venv venv

if not exist "venv\Scripts\activate.bat" (
    echo ERROR: Failed to create virtual environment.
    pause
    goto :eof
)

:: Create .gitignore in venv folder
echo Creating .gitignore in venv folder...
(
echo # Ignore all content in the virtual environment directory
echo *
echo # Except this file
echo !.gitignore
) > venv\.gitignore

:: Create venv_activate.bat helper
echo Creating venv_activate.bat...
(
echo @echo off
echo cd %%~dp0
echo set VENV_PATH=venv
echo.
echo echo Activating virtual environment...
echo call "%%VENV_PATH%%\Scripts\activate"
echo echo Virtual environment activated.
echo cmd /k
) > venv_activate.bat

:: Activate the virtual environment
echo.
echo Activating virtual environment...
call venv\Scripts\activate.bat

:: Upgrade pip
echo.
echo --- Upgrading pip ---
python -m pip install --upgrade pip

:: Install uv
echo.
echo --- Installing uv (fast package installer) ---
pip install uv

echo.
echo.
echo ================================================================
echo   Environment Setup Complete!
echo ================================================================
echo.
echo Your virtual environment 'venv' has been created and activated.
echo.
echo ----------------------------------------------------------------
echo   NEXT STEPS - Please complete these manually:
echo ----------------------------------------------------------------
echo.
echo   1. INSTALL PYTORCH
echo      Go to: https://pytorch.org/get-started/locally/
echo      Select your CUDA version and copy the install command.
echo      Example for CUDA 12.8:
echo        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
echo.
echo   2. INSTALL FLASH ATTENTION (Optional, for better performance)
echo      Download a pre-built wheel from:
echo        https://github.com/mjun0812/flash-attention-prebuild-wheels/releases
echo      or: https://huggingface.co/lldacing/flash-attention-windows-wheel/tree/main
echo      Place the .whl file in this folder, then run:
echo        pip install flash_attn-2.8.2+cu128torch2.8-cp312-cp312-win_amd64.whl
echo.
echo   3. INSTALL REQUIREMENTS
echo      Run: pip install -r requirements.txt
echo      Or with uv (faster): uv pip install -r requirements.txt
echo.
echo   4. LAUNCH THE APPLICATION
echo      Run: gui.bat
echo.
echo ----------------------------------------------------------------
echo.

cmd /k
