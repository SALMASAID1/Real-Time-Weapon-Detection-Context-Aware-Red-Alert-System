<#
.SYNOPSIS
    Starts the Real-Time Weapon Detection System backend and frontend.
.DESCRIPTION
    This script activates the Python virtual environment, starts the FastAPI backend,
    and starts the React frontend in the background. It cleanly kills both processes
    when the user presses CTRL+C.
#>

$ErrorActionPreference = "Stop"

$global:backendProcess = $null
$global:frontendProcess = $null

function Stop-System {
    Write-Host "`nShutting down Weapon Detection System..." -ForegroundColor Yellow

    if ($global:frontendProcess -and -not $global:frontendProcess.HasExited) {
        Write-Host "Stopping React frontend (PID: $($global:frontendProcess.Id))..."
        # Stop the process tree (since cmd spawns node)
        taskkill /PID $($global:frontendProcess.Id) /T /F *>$null
    }

    if ($global:backendProcess -and -not $global:backendProcess.HasExited) {
        Write-Host "Stopping FastAPI backend (PID: $($global:backendProcess.Id))..."
        # Stop the process tree (since powershell spawns python)
        taskkill /PID $($global:backendProcess.Id) /T /F *>$null
    }

    Write-Host "System shut down successfully." -ForegroundColor Green
    [Environment]::Exit(0)
}

# Setup the CTRL+C handler
[Console]::TreatControlCAsInput = $false
Register-EngineEvent -SourceIdentifier Console.CancelKeyPress -Action {
    Stop-System
} | Out-Null

Write-Host "Starting Real-Time Weapon Detection System..." -ForegroundColor Cyan

# 1. Start FastAPI Backend
$venvActivate = ".\venv\Scripts\activate.ps1"
if (-Not (Test-Path $venvActivate)) {
    Write-Host "Error: Virtual environment not found at $venvActivate." -ForegroundColor Red
    exit 1
}

Write-Host "Starting FastAPI backend on port 8000..."
# We start the backend by launching a new PowerShell process that activates venv and runs uvicorn
$backendStartArgs = "-NoProfile -Command `".\venv\Scripts\activate.ps1; uvicorn src.api.main:app --host 0.0.0.0 --port 8000`""
$global:backendProcess = Start-Process powershell -ArgumentList $backendStartArgs -PassThru -NoNewWindow

# 2. Start React Frontend
Write-Host "Starting React frontend in ui/ directory..."
# We start the frontend using cmd directly
$global:frontendProcess = Start-Process cmd -ArgumentList "/c cd ui && npm run dev" -PassThru -NoNewWindow

Write-Host "`n===================================================" -ForegroundColor Cyan
Write-Host "System is running!" -ForegroundColor Green
Write-Host "Backend API: http://localhost:8000"
Write-Host "Frontend UI: http://localhost:5173 (usually)"
Write-Host "Press CTRL+C to stop the system gracefully." -ForegroundColor Yellow
Write-Host "===================================================`n" -ForegroundColor Cyan

try {
    # Wait indefinitely until CTRL+C is pressed
    while ($true) {
        Start-Sleep -Seconds 1
    }
}
finally {
    Stop-System
}
