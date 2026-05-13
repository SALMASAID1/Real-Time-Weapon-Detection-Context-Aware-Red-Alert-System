# Task 1.1: Environment & Workspace Setup

## Description
Set up the development environment and project workspace. This includes initializing a virtual environment, installing dependencies, and configuring version control.

## Details
 1. [x] **Virtual Environment**: Create a Python virtual environment to isolate project dependencies.
   - `python -m venv venv`
   - `source venv/bin/activate`
 2. [x] **Dependencies**: Update `requirements.txt` with core libraries:
   - `ultralytics` (for YOLOv11)
   - `torch`, `torchvision`
   - `opencv-python`
   - `sahi`
   - `python-telegram-bot`
   - `pygame`
 3. [x] **Git Configuration**: Ensure `.gitignore` excludes large datasets, model weights (`.pt`, `.engine`), and virtual environments.

## Learning Resources
- [Ultralytics YOLOv11 Installation Guide](https://docs.ultralytics.com/quickstart/)
- [Python Virtual Environments: A Primer](https://realpython.com/python-virtual-environments-a-primer/)
- [Git Ignore Documentation](https://git-scm.com/docs/gitignore)
