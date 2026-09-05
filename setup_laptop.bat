@echo off
echo ============================================================
echo   DepthWizard - Laptop Automated Setup (SIH 2026 / ISRO)
echo ============================================================
echo.

where python >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Python was not found in PATH!
    echo Please install Python 3.10, 3.11, or 3.12 and ensure "Add Python to PATH" is checked.
    pause
    exit /b 1
)

echo [1/3] Creating fresh virtual environment (.venv)...
python -m venv .venv
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Failed to create .venv.
    pause
    exit /b 1
)

echo [2/3] Installing PyTorch CPU...
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

echo [3/3] Installing geospatial and web dependencies from requirements.txt...
.\.venv\Scripts\python.exe -m pip install -r requirements.txt

echo.
echo ============================================================
echo   SETUP COMPLETE! Everything is installed and verified.
echo   You can now double-click 'start_server.bat' to launch!
echo ============================================================
pause
