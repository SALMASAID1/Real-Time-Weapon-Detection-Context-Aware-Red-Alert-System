#!/bin/bash

# Define cleanup function for graceful termination
cleanup() {
    echo ""
    echo "Shutting down Weapon Detection System..."
    
    # Kill the React frontend (running in background)
    if [ -n "$FRONTEND_PID" ]; then
        echo "Stopping React frontend (PID: $FRONTEND_PID)..."
        kill $FRONTEND_PID 2>/dev/null
    fi
    
    # Kill the FastAPI backend (running in background)
    if [ -n "$BACKEND_PID" ]; then
        echo "Stopping FastAPI backend (PID: $BACKEND_PID)..."
        kill $BACKEND_PID 2>/dev/null
    fi
    
    echo "System shut down successfully."
    exit 0
}

# Trap SIGINT (CTRL+C) and SIGTERM
trap cleanup SIGINT SIGTERM

echo "Starting Real-Time Weapon Detection System..."

# 1. Activate Python virtual environment
if [ -f "venv/bin/activate" ]; then
    echo "Activating Python virtual environment..."
    source venv/bin/activate
else
    echo "Error: Virtual environment 'venv' not found."
    exit 1
fi

# 2. Start FastAPI backend
echo "Starting FastAPI backend on port 8000..."
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 &
BACKEND_PID=$!

# 3. Start React frontend
echo "Starting React frontend in ui/ directory..."
cd ui || exit
npm run dev &
FRONTEND_PID=$!
cd ..

echo ""
echo "==================================================="
echo "System is running!"
echo "Backend API: http://localhost:8000"
echo "Frontend UI: http://localhost:5173 (usually)"
echo "Press CTRL+C to stop the system gracefully."
echo "==================================================="
echo ""

# Wait for background processes to finish (or until trap is triggered)
wait
