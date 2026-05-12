# Real-Time Weapon Detection & Context-Aware Red Alert System

## Overview
This project implements a state-of-the-art weapon detection system using YOLOv11/v12 and Swin Transformers. It is designed for real-time monitoring with context-aware alerting logic (Proximity IoU) and multi-channel notification support (Telegram, Local Audio).

## Key Features
- **YOLOv11/v12 Backbone**: Efficient and accurate object detection.
- **Swin Transformer Neck**: Captures long-range spatial dependencies for improved context.
- **SAHI (Slicing Aided Hyper Inference)**: Enhances detection of small objects in high-resolution streams.
- **Context-Aware Alerting**: Triggers alerts based on Hand-Weapon proximity (GIoU) and temporal persistence.
- **Multi-Channel Alerts**: Notifications via Telegram Bot and local audio (Pygame).
- **Explainability (XAI)**: Grad-CAM heatmaps to visualize model decision-making.
- **React Dashboard**: Real-time monitoring UI with live video, threat history, and runtime settings.

## Project Structure
- `data/`: Dataset configurations, processed YOLO dataset, audio assets, and event logs.
- `models/`: Custom neural network architecture components (backbone, neck, head).
- `src/`: Core application source code:
  - `api/`: FastAPI backend, WebSocket streaming, REST routers.
  - `inference/`: SAHI pipeline and dual-cadence inference engine.
  - `threat_logic/`: Proximity scoring, IoU calculation, alert dispatch.
  - `xai/`: Grad-CAM generation and overlay rendering.
- `ui/`: React (Vite) frontend for live surveillance monitoring.
- `notebooks/`: EDA and Training experiments (Phase 1 & Phase 2).
- `plan/`: Milestone-based project development plan.
- `docs/`: Technical documentation and implementation guides (EN & FR).
- `scripts/`: Data fetching and automation scripts.

## Documentation
- **Project Roadmap**: [EN](docs/en/project_roadmap.md) | [FR](docs/fr/project_roadmap.md)
- **Current Status**: [EN](docs/en/project_status.md) | [FR](docs/fr/project_status.md)
- **EDA Findings**: [Detailed analysis of dataset geometry](docs/en/EDA_Findings.md)
- **Milestone 2**: [Hybrid Model Implementation](docs/en/milestone_2_implementation.md)
- **Milestone 3**: [Inference Intelligence & XAI](docs/en/milestone-3-inference-intelligence.md)
- **Milestone 4**: [Multimodal Alerting & Dashboard](docs/en/milestone-4-alerting-system.md)

## Quick Start

### Backend (FastAPI)
```bash
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

### Frontend (React)
```bash
cd ui
npm install
npm run dev
```

The React dashboard will be available at `http://localhost:5173`.

## Development Plan
See the [plan/](plan/) directory for detailed milestones and tasks.
