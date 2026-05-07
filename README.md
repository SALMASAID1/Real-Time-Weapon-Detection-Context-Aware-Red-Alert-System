# Real-Time Weapon Detection & Context-Aware Red Alert System

## Overview
This project implements a state-of-the-art weapon detection system using YOLOv11/v12 and Swin Transformers. It is designed for real-time monitoring with context-aware alerting logic (Proximity IoU) and multi-channel notification support (Telegram, Local Audio).

## Key Features
- **YOLOv11/v12 Backbone**: Efficient and accurate object detection.
- **Swin Transformer Neck**: Captures long-range spatial dependencies for improved context.
- **SAHI (Slicing Aided Hyper Inference)**: Enhances detection of small objects in high-resolution streams.
- **Context-Aware Alerting**: Triggers alerts based on threat proximity and confidence.
- **Multi-Channel Alerts**: Notifications via Telegram Bot and local audio (Pygame).

## Project Structure
- `data/`: Dataset configurations and local storage.
- `models/`: Custom neural network architecture components.
- `utils/`: Inference, logic, and alerting utilities.
- `notebooks/`: EDA and Training experiments.
- `plan/`: Milestone-based project development plan.

## Setup
1. Create a virtual environment: `python -m venv venv`
2. Activate it: `source venv/bin/activate`
3. Install dependencies: `pip install -r requirements.txt`

## Development Plan
See the [plan/](plan/) directory for detailed milestones and tasks.
