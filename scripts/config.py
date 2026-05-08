import os
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
YOLO_DATASET_DIR = PROCESSED_DIR / "yolo_dataset"

# Source Specific Raw Paths
DS_MERGED = RAW_DIR / "dataset_merged"
DS_ARCHIVE = RAW_DIR / "archive" / "weapon_detection"
DS_SIMULETIC = RAW_DIR / "Simuletic_Weapon_Umbrella_Dataset"
DS_OI_HARD_NEG = RAW_DIR / "hard_negatives_oi"
DS_COCO_HARD_NEG = RAW_DIR / "hard_negatives_coco"
DS_ADDITIONAL_WEAPONS = RAW_DIR / "additional_weapons_oi"

# Target Class Ontology
# 0: Weapon
# 1: Person
# 2: Confuser
CLASS_MAP = {
    "weapon": 0,
    "person": 1,
    "confuser": 2
}

# Training Split Ratios
SPLIT_RATIO = {
    "train": 0.8,
    "val": 0.1,
    "test": 0.1
}

# FiftyOne Fetch Settings
FO_MAX_SAMPLES = 5000
FO_SEED = 42
