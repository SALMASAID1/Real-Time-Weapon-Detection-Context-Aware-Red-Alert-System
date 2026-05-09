#!/bin/bash

# Real-Time Weapon Detection — Data Setup Pipeline
# This script automates the fetching of external data and the assembly of the unified dataset.

# Exit on error
set -e

echo "===================================================="
echo "   Weapon Detection System — Data Setup Pipeline    "
echo "===================================================="

# 1. Activate virtual environment if it exists
if [ -d "venv" ]; then
    echo "[1/4] Activating virtual environment..."
    source venv/bin/activate
else
    echo "[1/4] Warning: venv not found. Ensure dependencies are installed."
fi

# 2. Fetch external datasets
echo "[2/4] Fetching external datasets (OI, COCO)..."
python3 scripts/data/fetch_external_data.py

# 3. Build unified dataset
echo "[3/4] Assembling unified YOLO dataset..."
python3 scripts/data/build_unified_dataset.py

# 4. Synthetic Injection (Weather & Low-light)
echo "[4/4] Injecting synthetic weather and low-light conditions..."
python3 scripts/data/augment_synthetic.py

echo ""
echo "===================================================="
echo "   Data Setup Complete! Ready for Training.         "
echo "===================================================="
