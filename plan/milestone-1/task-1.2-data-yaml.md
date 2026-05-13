# Task 1.2: Data Aggregation & Cleaning

## Description
Consolidate ~36,000 images from six disparate sources into a unified YOLO-format dataset, balancing classes and ensuring high generalization.

## Details
 1. [x] **Source Aggregation**: Combine images into `data/processed/yolo_dataset/` from:
   - `dataset_merged` (~23,300 images)
   - `archive` / UGR SOHAS (~720 images)
   - `Simuletic_Weapon_Umbrella_Dataset` (~120 images)
   - `Weapon-detection.v1i.yolov11` (~2,100 images)
   - `hard_negatives_oi` (~5,000 Open Images V7 Confuser samples)
   - `hard_negatives_coco` (~5,000 COCO 2017 Person samples)
 2. [x] **Standardization & Class Mapping**: Convert all annotations to YOLO format and map to the unified schema: `0: Weapon`, `1: Person`, `2: Confuser`.
 3. [x] **Addressing Imbalance**: FiftyOne is used to balance the dataset by adding 10,000 high-variance "Confuser/Person" samples to offset the ~26,000 weapon-heavy samples.
 4. [x] **Hand Detection Strategy**: Confirmed Two-Stream Inference approach. No hand labeling will occur in this dataset; we will use a parallel pre-trained YOLO hand model for proximity logic ($B_h \cap B_w$) calculation.
 5. [x] **Cleaning**: Remove corrupted images and verify that weapons are correctly labeled in low-light samples.

## Learning Resources
- [UGR SOHAS Dataset Info](https://deep-learning-ugr.github.io/SOHAS/)
- [Dataset Merging Strategies for YOLO](https://docs.ultralytics.com/datasets/explorer/)
- [Roboflow: Dataset Management at Scale](https://roboflow.com/formats/yolo-v8-pytorch-txt)
