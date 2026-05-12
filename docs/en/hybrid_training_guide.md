# Hybrid Model Training Guide

This document provides a step-by-step workflow for training the Real-Time Weapon Detection system in different environments (Local, Lightning AI Studio, and Kaggle).

## 1. Environment Preparation

### Local / Lightning Studio
1. **Clone the Project**:
   ```bash
   git clone <your-repo-url>
   cd Real-Time-Weapon-Detection-Context-Aware-Red-Alert-System
   ```
2. **Install Dependencies**:
   ```bash
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   pip install -e .
   ```

### Kaggle
1. **Upload Source**: Zip the `models/`, `scripts/`, and `utils/` folders and upload them as a Kaggle Dataset or Utility Script.
2. **GPU Settings**: Go to **Settings -> Accelerator** and select **GPU T4 x2**.
3. **Internet**: Ensure **Internet on** is toggled in the sidebar.

---

## 2. Data Preparation
Before training, you must have the unified dataset ready.

1. **Build Dataset**:
   Run the following to merge multi-source data (OI v7, COCO, Simuletic) into the clean 3-class YOLO format:
   ```bash
   python scripts/data/build_unified_dataset.py
   ```
2. **Verify Paths**:
   Ensure `data/processed/yolo_dataset/data.yaml` exists and points to absolute image paths.

---

## 3. Training Workflow

### Choosing the Right Notebook
| Environment | Notebook to Use | Features |
| :--- | :--- | :--- |
| **Local / Studio** | `Phase2_Training_v2.ipynb` | Single GPU, Auto-resume, Periodic saving. |
| **Kaggle** | `Phase2_Training_v3.ipynb` | **Dual T4 GPUs**, AMP (Mixed Precision), Class Weighting. |

### Execution Steps
1. **Initialise Model**: The script will download the `yolo11m.pt` backbone automatically.
2. **Warm-up Phase (Epochs 1-10)**: 
   *   The YOLO backbone is **frozen**.
   *   Only the Swin Transformer Neck and Focal Head are training.
   *   Learning Rate: `1e-4`.
3. **Fine-Tuning Phase (Epochs 11-50)**:
   *   Backbone is **unfrozen**.
   *   The entire end-to-end architecture is optimized.
   *   Learning Rate: `1e-5` (lower to preserve pretrained features).

---

## 4. Monitoring & Artifacts

- **Loss Curves**: Monitor the training/val loss in the notebook output.
- **Weights**: 
  - `best.pt`: Saved automatically when validation loss improves.
  - `last.pt`: Saved every 2-5 epochs for auto-resume.
- **Class Balance**: In v3, we apply a **2.5x penalty** to "Confuser" errors to ensure the "Red Alert" system has a very low False Positive rate.

---

## 5. Troubleshooting

### "CUDA Out of Memory"
- Reduce `batch_size` in the `get_dataloaders` function (e.g., from 32 to 16).
- Ensure `autocast()` is active in the training loop.

### "ModuleNotFoundError: models"
- Ensure you have run `pip install -e .` or that the project root is added to `sys.path`.

### "FileNotFoundError: data.yaml"
- Double check the `DATA_YAML` path variable in the "Execution" cell of the notebook. In Kaggle, this must be `/kaggle/input/WD-data/data.yaml`.
