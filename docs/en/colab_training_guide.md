# Training on Google Colab: A Step-by-Step Guide

This guide details how to transition the **Real-Time Weapon Detection & Context-Aware Red Alert System** to Google Colab for accelerated training using cloud GPUs.\n\n> [!TIP]\n> **New**: We now recommend using **DagsHub Storage** for better data versioning. See the [DagsHub Migration Guide](dagshub_colab_setup.md) for the latest workflow.\n

## 1. Prerequisites
- A Google Account.
- Project files uploaded to Google Drive or a GitHub repository.
- Recommended: Google Colab Pro for longer sessions (optional).

## 2. Environment Setup

### 2.1. Open Google Colab
Go to [colab.research.google.com](https://colab.research.google.com) and create a new notebook.

### 2.2. Select GPU Runtime
1. Click on **Runtime** in the top menu.
2. Select **Change runtime type**.
3. Choose **T4 GPU** (or A100 if available) from the Hardware accelerator dropdown.

### 2.3. Mount Google Drive (Recommended)
This allows you to persist your dataset and model weights.
```python
from google.colab import drive
drive.mount('/content/drive')
```

### 2.4. Clone Repository or Upload Files
If your code is on GitHub:
```bash
!git clone https://github.com/your-username/Real-Time-Weapon-Detection-Context-Aware-Red-Alert-System.git
%cd Real-Time-Weapon-Detection-Context-Aware-Red-Alert-System
```
Otherwise, upload your project folder to Drive and navigate to it:
```python
%cd /content/drive/MyDrive/Real-Time-Weapon-Detection-Context-Aware-Red-Alert-System
```

## 3. Installation

Install all required dependencies.
```bash
!pip install -r requirements.txt
```

## 4. Data Preparation

### 4.1. Setup Data Directories
Ensure the directory structure is ready:
```bash
!python scripts/data/build_unified_dataset.py
```
*(Note: If you haven't downloaded the raw data yet, run the fetch script first)*
```bash
!python scripts/data/fetch_external_data.py
```

### 4.2. Verify Dataset
Check if `data/processed/yolo_dataset/data.yaml` exists and points to the correct relative paths.

## 5. Training Execution

You can run the training directly using the provided notebook or via a CLI command if you convert the notebook to a script.

### 5.1. Running the Notebook
Open `notebooks/Phase2_Training.ipynb` in Colab. You may need to adjust the paths:
- Update `data_yaml` path in the execution cell.
- Ensure the `models` directory exists to save weights.

### 5.2. Training Script Adjustment
Ensure your `HybridTrainer` is configured for Colab:
```python
# In Phase2_Training.ipynb
trainer = HybridTrainer(model, train_loader, val_loader, device="cuda")
trainer.run(epochs=50)
```

## 6. Monitoring

### 6.1. TensorBoard
You can monitor training progress in real-time within Colab:
```python
%load_ext tensorboard
%tensorboard --logdir runs/train
```

### 6.2. Weights & Biases (Optional)
If you have a WandB account, it will automatically track the Ultralytics backbone training if you log in:
```bash
!pip install wandb
import wandb
wandb.login()
```

## 7. Saving and Exporting
After training, your best weights will be in `models/weights/best.pt`. 
- **Download**: You can download it directly from the Colab file explorer.
- **Drive**: If you mounted Drive, it's already saved there!

---
**Tip**: If you experience "Out of Memory" (OOM) errors, reduce the `batch_size` in the `get_dataloaders` function call (e.g., from 16 to 8 or 4).
