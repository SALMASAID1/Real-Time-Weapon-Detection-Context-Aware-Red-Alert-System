# Task 1.2: Data Acquisition & YAML Configuration

## Description
Configure the dataset structure and YAML files required for YOLOv11 training.

## Details
1. **Dataset Structure**: Organize data into `train`, `val`, and `test` folders inside `data/`.
2. **YAML Configuration**: Create `data/yaml/dataset.yaml` to define:
   - Paths to train/val image sets.
   - Number of classes (`nc`).
   - Class names (e.g., `pistol`, `rifle`, `knife`).
3. **Data Verification**: Ensure that labels are in YOLO format (normalized `x_center`, `y_center`, `width`, `height`).

## Learning Resources
- [YOLOv8/v11 Dataset Format Guide](https://docs.ultralytics.com/datasets/detect/)
- [Roboflow: How to export data for YOLO](https://blog.roboflow.com/how-to-train-yolov8-on-a-custom-dataset/)
- [LabelImg GitHub](https://github.com/HumanSignal/labelImg) (for local annotation)
