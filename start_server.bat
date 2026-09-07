@echo off
echo ============================================================
echo   DepthWizard 3D Flythrough Server (ISRO PS 26175)
echo ============================================================
echo.

if not exist ".venv\Scripts\python.exe" (
    echo [ERROR] Virtual environment [.venv] not found!
    echo Please run 'setup_laptop.bat' first to install dependencies.
    pause
    exit /b 1
)

echo Starting FastAPI & Three.js 3D Viewer...
echo Access in your browser at: http://localhost:8000
echo.

start http://localhost:8000
.\.venv\Scripts\python.exe -m uvicorn api.main:app --host 127.0.0.1 --port 8000
