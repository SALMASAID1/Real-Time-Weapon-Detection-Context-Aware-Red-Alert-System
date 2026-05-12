# Training on Google Colab with DagsHub

This guide provides the complete environment setup and workflow for training the **Real-Time Weapon Detection & Context-Aware Red Alert System** using Google Colab and **DagsHub Storage**.

---

## 1. Two-Repo Architecture

This project uses **two separate DagsHub repos** with distinct purposes:

| Repo | Purpose | Contents |
|------|---------|----------|
| `dagshub-drive` | **Dataset Storage** | 41k+ images stored in DagsHub S3 Storage at `s3:/dagshub-drive/data/processed/yolo_dataset` |
| Source code | **Model Code** | Persisted on Google Drive (loaded into Colab at runtime) |

> **Key**: The `DagsHubFilesystem` must point to `dagshub-drive` — the storage repo — NOT the code repo name.
> This is confirmed by the dataset URL: `dagshub.com/TAMZIRT-MOHAMED/dagshub-drive/src/main/s3:/dagshub-drive/data/processed/yolo_dataset`

- The dataset should exist at `data/processed/yolo_dataset/` inside `dagshub-drive`.
- Ensure `data.yaml` uses relative paths (e.g., `train: train/images`).

---

## 2. Google Colab Initial Setup

1.  **Open Notebook**: Upload and open `notebooks/Phase2_Training_v5.ipynb` in [Google Colab](https://colab.research.google.com).
2.  **Hardware Accelerator**:
    *   Go to **Runtime** > **Change runtime type**.
    *   Select **T4 GPU** (Standard on free tier).
3.  **Environment Preparation**:
    *   Run the **Step 1** cell in the notebook.
    *   This installs `dagshub`, `ultralytics`, and fixes the **Numpy 2.0 incompatibility**.
    *   **CRITICAL**: If the output warns you about Numpy 2.x, click **"RESTART SESSION"** in the popup and proceed to Step 2.

---

## 3. Connecting to DagsHub Storage

In the notebook's **Step 2**, you will authenticate and mount both storage systems.

```python
# ── CORRECT PATTERN (v5) ──────────────────────────────────────────────────
from dagshub.streaming import DagsHubFilesystem
import dagshub
import dagshub.colab

# 1. Authenticate (opens browser OAuth once; token is cached)
dagshub.colab.login()

# 2. Instantiate DagsHubFilesystem with an explicit HTTPS repo URL
#    ⚠️  Do NOT use: dagshub.streaming.install_hooks(repo_url=...) ← broken
#    That form tries to read .git/HEAD which doesn't exist in Colab.
fs = DagsHubFilesystem(
    repo_url="https://dagshub.com/OWNER/REPO_NAME",
    branch="main",
    project_root="/content/dagshub_streaming",
)
fs.install_hooks()   # patches open(), os.listdir(), pathlib, etc.

# 3. Mount Google Drive (source code + weights)
drive.mount('/content/drive')
```

**What this does**:
- **Source Code**: Loaded from `/content/drive/MyDrive/...`. Your latest changes are instantly available.
- **Dataset**: Streamed from DagsHub. This bypasses the hours of sync time Google Drive usually requires for 50k+ images.
- **Checkpoints**: Automatically saved back to your GDrive `models/weights/` folder for persistence.


---

## 4. Environment Verification

Once mounted, the notebook will verify the structure:
- **CWD (Working Directory)**: Automatically moved to the project root.
- **Python Path**: The project root is added to `sys.path` so that `from models...` works.
- **Data Check**: Use the diagnostic cell (if provided) to ensure images are visible.

---

## 5. Training Logic (Red Alert Config)

The environment is pre-configured for our **Milestone 2** goals:
- **Optimizer**: AdamW with Mixed Precision (`torch.cuda.amp.autocast`).
- **Hardware**: `batch_size=16` and `num_workers=2`.
- **Intelligence**: Class weights `[1.0, 1.0, 2.5]` are applied to the head to prioritize reducing false positives (Confuser class).
- **Strategy**: 10-epoch frozen backbone followed by full fine-tuning.

---

## 6. Troubleshooting Common Issues

### ModuleNotFoundError: No module named 'models'
- **Cause**: The runtime was restarted and Step 2 (Alignment) hasn't been run yet.
- **Solution**: Re-run the Step 2 cell to re-add the project path to Python's memory.

### PytorchStreamReader failed reading zip archive
- **Cause**: A corrupted `.pt` weight file (often `yolo11m.pt`).
- **Solution**: Delete the corrupted file from your DagsHub storage and let the code re-download it.

### FileNotFoundError: `/content/dagshub_streaming/.git/HEAD` + UnsupportedProtocol
- **Root Cause**: The old code called `dagshub.streaming.install_hooks(repo_url=repo_url)` where `repo_url` was the return value of `dagshub.colab.login()`. The `install_hooks()` module-level function does **not** accept `repo_url` as an argument — it ignores it and instead tries to auto-detect the repository by reading `.git/HEAD` from the filesystem, which doesn't exist in a fresh Colab environment. The resulting internal URL is then malformed → `UnsupportedProtocol`.
- **Fix** (applied in v5): Always use `DagsHubFilesystem(repo_url="https://dagshub.com/USER/REPO", ...).install_hooks()` — instantiate the class first with an explicit HTTPS URL, then call `.install_hooks()` on the **instance**.

### No images found in .../images
- **Cause**: `labels.cache` files from a different machine are interfering.
- **Solution**: Delete all `labels.cache` files in the `data/processed/yolo_dataset/` subfolders to force a fresh scan.

---

## 7. Persistent Checkpointing
All weights are saved to `models/weights/` on the mounted DagsHub drive. 
- **last.pt**: Saved every epoch (contains optimizer state for auto-resume).
- **best.pt**: Saved whenever validation loss improves.
- **epoch_N.pt**: Permanent checkpoints saved every 5 epochs.
