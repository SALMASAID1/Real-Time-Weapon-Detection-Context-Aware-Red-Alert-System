# Kaggle Dual T4 Training Guide

This guide covers the specific steps to train the **Hybrid Weapon Detector** on Kaggle using Dual T4 GPUs.

## 1. Project Preparation
Kaggle works best when your code is uploaded as a "Utility Script" or a Zipped dataset.

1. **Zip the Project**: On your local machine, zip the following folders:
   - `models/`
   - `scripts/`
   - `utils/`
   - `requirements.txt`
2. **Upload to Kaggle**: 
   - Click **+ New Dataset**.
   - Name it `Weapon-Detection-Source`.
   - Upload your zip file.

---

## 2. Kernel Setup
1. **Create a New Notebook**: In Kaggle, click **+ New Notebook**.
2. **Add Data**:
   - Add your `WD-data` (the images and `data.yaml`).
   - Add your `Weapon-Detection-Source` (the zipped code).
3. **Accelerator**: 
   - Go to **Settings -> Accelerator**.
   - Select **GPU T4 x2**.
4. **Internet**:
   - Ensure **Internet on** is selected in the settings sidebar (required for downloading the pretrained YOLO backbone).

---

## 3. Execution (The v3 Notebook)
Use the code from **`Phase2_Training_v3.ipynb`**. This notebook is pre-configured for Kaggle:

### Path Logic
- **Input**: The script looks for data at `/kaggle/input/WD-data/data.yaml`.
- **Output**: Weights and logs are saved to `/kaggle/working/models/weights/`.

### Training Strategy
1. **Dual GPU**: The code automatically uses `nn.DataParallel` to use both T4 cards.
2. **AMP**: Mixed precision is enabled to fit `batch_size=32` into the 15GB VRAM.
3. **Class Weighting**: The "Confuser" class is weighted **2.5x** to minimize false alarms.

---

## 4. Managing the 12-Hour Limit
Kaggle sessions auto-terminate after 12 hours.

- **Auto-Save**: The trainer saves `last.pt` every **2 epochs**.
- **Resuming**: If your session ends, start a new one and run the notebook again. It will detect `/kaggle/working/models/weights/last.pt` and resume exactly where it stopped.
- **Persistence**: Files in `/kaggle/working` are deleted when the session ends **unless** you "Save Version" (Commit). 
  - **Tip**: To keep your weights permanently, run a "Save Version -> Save & Run All". Kaggle will run the whole thing in the background and save the final `best.pt` in the output section.

---

## 5. Key Metrics to Watch
- **Train/Val Loss**: Should decrease steadily.
- **Best Model**: Look for the log line `--> Best model saved with val_loss: X.XXXX`.
- **False Positives**: Because of the **2.5x weighting** on Confusers, the classification loss for the "Confuser" class will be higher than others initially—this is normal and expected.
