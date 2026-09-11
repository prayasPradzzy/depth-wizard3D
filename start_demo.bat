@echo off
REM ============================================================
REM  DepthWizard - PRESENTATION / LIVE DEMO LAUNCHER
REM  Tuned for fast on-stage uploads:
REM    DW_TILED=off    single-pass inference (tiled costs ~20s extra)
REM    DW_PREWARM=true model loaded at startup, first upload is fast
REM ============================================================
echo Starting DepthWizard (demo mode)...
set DW_TILED=off
set DW_PREWARM=true
set DW_MESH_DIM=1024
if not exist ".venv\Scripts\python.exe" (
    echo [ERROR] .venv not found. Run setup_laptop.bat first.
    pause
    exit /b 1
)
start "" http://localhost:8000
.\.venv\Scripts\python.exe -m uvicorn api.main:app --host 127.0.0.1 --port 8000
