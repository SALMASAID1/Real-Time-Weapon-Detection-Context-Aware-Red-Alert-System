# Project Context: Real-Time Weapon Detection & Red Alert System

## Project Overview
This project, **Real-Time Weapon Detection & Context-Aware Red Alert System**, is an advanced deep learning application designed for active threat intelligence. It moves beyond simple object classification to identify firearms and bladed weapons in real-time surveillance feeds while distinguishing them from "confuser objects" like smartphones, umbrellas, or drills.

### Core Objectives
- **Precision Detection**: Utilizing YOLOv11/v12 for high-speed object detection.
- **Two-Stream Inference**: Implementing a parallel inference pipeline:
  - **Weapon Stream**: Our custom Hybrid YOLO-Swin model for Weapons and Confusers.
  - **Hand Stream**: A specialized, pre-trained YOLO model for robust Hand detection.
- **Contextual Understanding**: Using Swin Transformers to capture long-range spatial dependencies (e.g., relationship between a person and a weapon).
- **Active Threat Logic**: Implementing "Hand-Weapon Proximity" analysis to trigger alerts only when a weapon is being handled (IoU between Hand and Weapon boxes).
- **Multi-Modal Alerting**: Automated visual, auditory (local), and digital (Telegram) notifications.
- **Explainability (XAI)**: Grad-CAM integration to visualize why the model triggered an alert.

---

## Technical Architecture

### 1. Model Stack
- **Backbone**: YOLOv11/v12 CSP (Cross Stage Partial) for efficient feature extraction.
- **Neck**: Swin Transformer blocks with Shifted Windows (S-W-MSA) for global context.
- **Inference Strategy**: Two-Stream Parallel Inference:
  - **Stream A**: Custom YOLO-Swin (Weapons/Confusers).
  - **Stream B**: Pre-trained YOLO Hand Model (e.g., YOLOv8-Hand).
- **Inference Optimization**: SAHI (Slicing Aided Hyper Inference) for small object detection in 4K streams.
- **Optimization**: TensorRT/OpenVINO quantization for high FPS (≥ 40 FPS).

### 2. Software Stack
- **Backend**: FastAPI (Python) with WebSockets for real-time inference streaming.
- **Frontend**: React (Vite) dashboard for live monitoring and threat history.
- **Data Engineering**: FiftyOne for dataset curation, pruning, and visual QA.
- **Alerting**: `python-telegram-bot` and `pygame` for notifications.

---

## Development Roadmap (Milestones)

The project is structured into six distinct milestones, each focusing on a critical phase of the lifecycle.

### Milestone 1: Data Engineering & Environment Setup
- **Focus**: Building the foundation.
- **Tasks**:
  - Setting up the Python virtual environment and dependencies.
  - Aggregating ≈45,000 images from various sources (SOHAS, Unidpro, Synthetic).
  - Standardizing labels into a unified YOLO schema: `[0: Weapon, 1: Person, 2: Confuser]`.
  - Performing Exploratory Data Analysis (EDA) to identify class imbalances and hard samples.

### Milestone 2: Hybrid Model Development
- **Focus**: Architecting the neural network.
- **Tasks**:
  - Implementing the YOLOv11 backbone.
  - Integrating Swin Transformer blocks as the "neck" to enhance contextual awareness.
  - Setting up the training pipeline with custom loss functions (CIoU + Focal Loss).

### Milestone 3: Inference Intelligence & XAI
- **Focus**: Improving detection quality and transparency.
- **Tasks**:
  - Integrating SAHI for high-resolution tiling inference.
  - Developing "Hand-Weapon Proximity" logic (IoU-based) to validate active threats.
  - Implementing Grad-CAM for model explainability and debugging.

### Milestone 4: Multimodal Alerting & Dashboard
- **Focus**: Real-time response and user interface.
- **Tasks**:
  - Telegram Bot integration for remote notifications.
  - Local audio alert system using Pygame.
  - Integrating visual overlays (bounding boxes, proximity heatmaps) in the inference stream.
  - Building the React/FastAPI dashboard for live surveillance monitoring.

### Milestone 5: Optimization & Deployment
- **Focus**: Edge performance and validation.
- **Tasks**:
  - Full 50-epoch model training and mAP validation.
  - Quantizing models to TensorRT/OpenVINO for real-time performance (≥ 40 FPS).
  - Refining the alert dispatch logic to prevent "alert fatigue."

### Milestone 6: Dataset Optimization (Advanced)
- **Focus**: Refining accuracy through data surgery.
- **Tasks**:
  - Using FiftyOne Brain to compute image uniqueness.
  - Pruning redundant data in the "Weapon" class to balance it against "Confuser" objects.
  - Visual QA to fix mislabels in hard negative samples.

---

## Success Metrics
| Metric | Target |
| :--- | :--- |
| **mAP@0.5:0.95** | ≥ 0.72 |
| **False Positive Rate** | < 2% on "Confuser Objects" |
| **Inference Speed** | ≤ 25ms per 1080p frame |
| **System Latency** | < 500ms from Detection to Alert |

---

## Workspace Directory Breakdown
- `data/`: Raw and processed (YOLO) datasets.
- `models/`: Custom PyTorch implementations of the Hybrid architecture.
- `src/`:
  - `api/`: FastAPI backend and WebSocket handlers.
  - `inference/`: SAHI pipeline and engine logic.
  - `threat_logic/`: Proximity and alerting algorithms.
  - `xai/`: Grad-CAM visualization tools.
- `ui/`: React frontend source code.
- `plan/`: Detailed task-by-task execution plans.
- `scripts/`: Data fetching and automation scripts.
