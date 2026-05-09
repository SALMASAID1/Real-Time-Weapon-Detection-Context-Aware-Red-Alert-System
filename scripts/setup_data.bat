@echo off
:: Real-Time Weapon Detection — Data Setup Pipeline (Windows)
:: This script automates the fetching of external data and the assembly of the unified dataset.

echo ====================================================
echo    Weapon Detection System — Data Setup Pipeline    
echo ====================================================

:: 1. Activate virtual environment if it exists
if exist venv\Scripts\activate.bat (
    echo [1/4] Activating virtual environment...
    call venv\Scripts\activate.bat
) else (
    echo [1/4] Warning: venv not found. Ensure dependencies are installed.
)

:: 2. Fetch external datasets
echo [2/4] Fetching external datasets (OI, COCO)...
python scripts\data\fetch_external_data.py

:: 3. Build unified dataset
echo [3/4] Assembling unified YOLO dataset...
python scripts\data\build_unified_dataset.py

:: 4. Synthetic Injection (Weather & Low-light)
echo [4/4] Injecting synthetic weather and low-light conditions...
python scripts\data\augment_synthetic.py

echo.
echo ====================================================
echo    Data Setup Complete! Ready for Training.         
echo ====================================================
pause
