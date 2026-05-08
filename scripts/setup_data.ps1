# Real-Time Weapon Detection — Data Setup Pipeline (PowerShell)
# This script automates the fetching of external data and the assembly of the unified dataset.

Write-Host "====================================================" -ForegroundColor Cyan
Write-Host "   Weapon Detection System — Data Setup Pipeline    " -ForegroundColor Cyan
Write-Host "====================================================" -ForegroundColor Cyan

# 1. Activate virtual environment if it exists
if (Test-Path "venv\Scripts\Activate.ps1") {
    Write-Host "[1/3] Activating virtual environment..."
    . venv\Scripts\Activate.ps1
} else {
    Write-Host "[1/3] Warning: venv not found. Ensure dependencies are installed." -ForegroundColor Yellow
}

# 2. Fetch external datasets
Write-Host "[2/3] Fetching external datasets (OI, COCO)..."
python scripts\data\fetch_external_data.py

# 3. Build unified dataset
Write-Host "[3/3] Assembling unified YOLO dataset..."
python scripts\data\build_unified_dataset.py

Write-Host ""
Write-Host "====================================================" -ForegroundColor Green
Write-Host "   Data Setup Complete! Ready for Training.         " -ForegroundColor Green
Write-Host "====================================================" -ForegroundColor Green

Pause
