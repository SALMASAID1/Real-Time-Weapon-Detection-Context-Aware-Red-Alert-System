# Kaggle Dual T4 Training Guide (Pre-Processed Data)

This guide covers how to train your model on Kaggle using your **already processed dataset**. 

> [!IMPORTANT]
> **Do not run `build_unified_dataset.py` on Kaggle.** This guide assumes you have already processed your 50k images locally and are uploading them as a ready-to-use dataset.

## 1. Uploading your Processed Data
1. **Prepare the Folder**: Locate your local `data/processed/yolo_dataset` folder.
2. **Create Kaggle Dataset**:
   - Go to Kaggle -> **Datasets** -> **+ New Dataset**.
   - Name it exactly: `WD-data`.
   - Upload the **entire contents** of your `yolo_dataset` folder (including `train/`, `val/`, `test/` and `data.yaml`).
   - Create the dataset.

## 2. Uploading your Source Code
1. **Zip Code**: Zip these folders: `models/`, `utils/`, `requirements.txt`.
2. **Upload**: You can upload this as another dataset or simply drag-and-drop it into your Kaggle Notebook's `/kaggle/working` directory.

---

## 3. Kernel Configuration
1. **Accelerator**: Select **GPU T4 x2**.
2. **Internet**: Turn **ON** (needed for the initial YOLOv11 weight download).

## 4. Execution (The v3 Notebook)
Use **`Phase2_Training_v3.ipynb`**. It includes a **Path Fixer** that automatically aligns your local `data.yaml` paths with Kaggle's `/kaggle/input/WD-data/` structure.

- **Batch Size**: Set to `32`.
- **Num Workers**: Set to `2`.
- **Class Weights**: `Confuser` (Index 2) is automatically set to **2.5x weight** to reduce False Positives.

## 5. Artifact Retrieval
After training (or when the 12h limit is approaching):
- Your models will be in `/kaggle/working/models/weights/`.
- Download `best.pt` for deployment.
- If you need to resume later, download `last.pt` and upload it back to the same folder in your next session.
