@echo off
TITLE Weapon Detection System Launcher

echo ===================================================
echo Starting Real-Time Weapon Detection System...
echo ===================================================
echo.

:: 1. Activate Python virtual environment and start FastAPI backend
if exist "venv\Scripts\activate.bat" (
    echo [1/2] Starting FastAPI backend on port 8000...
    start "FastAPI Backend" cmd /k "call venv\Scripts\activate.bat && uvicorn src.api.main:app --host 0.0.0.0 --port 8000"
) else (
    echo Error: Virtual environment 'venv' not found.
    pause
    exit /b 1
)

:: 2. Start React frontend
echo [2/2] Starting React frontend...
if exist "ui" (
    cd ui
    start "React Frontend" cmd /k "npm run dev"
    cd ..
) else (
    echo Error: 'ui' directory not found.
    pause
    exit /b 1
)

echo.
echo ===================================================
echo System is starting up in separate windows!
echo Backend API: http://localhost:8000
echo Frontend UI: http://localhost:5173 (usually)
echo.
echo To stop the system gracefully, simply close the two 
echo newly opened command windows.
echo ===================================================
echo.
pause
