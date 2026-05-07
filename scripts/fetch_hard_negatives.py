import os
import fiftyone as fo
import fiftyone.zoo as foz
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
OI_DIR = RAW_DIR / "hard_negatives_oi"
COCO_DIR = RAW_DIR / "hard_negatives_coco"

def fetch_open_images():
    print("Fetching Open Images V7 Hard Negatives...")
    classes = ["Power tool", "Hammer", "Screwdriver", "Mobile phone", "Umbrella", "Handbag", "Remote control"]
    
    # We load the dataset and then export it to YOLO format
    dataset = foz.load_zoo_dataset(
        "open-images-v7",
        split="train",
        label_types=["detections"],
        classes=classes,
        max_samples=5000,
        dataset_name="oi_confusers_tmp",
        drop_existing_dataset=True
    )
    
    print(f"Exporting Open Images to {OI_DIR}...")
    dataset.export(
        export_dir=str(OI_DIR),
        dataset_type=fo.types.YOLOv5Dataset,
        label_field="ground_truth",
        classes=classes
    )
    fo.delete_dataset("oi_confusers_tmp")

def fetch_coco():
    print("Fetching COCO 2017 Hard Negatives...")
    # COCO has "person" class. "Man", "Woman", "Boy", "Girl" are Open Images classes or not standard COCO.
    # We will use "person".
    classes = ["person"]
    
    dataset = foz.load_zoo_dataset(
        "coco-2017",
        split="train",
        label_types=["detections"],
        classes=classes,
        max_samples=5000,
        dataset_name="coco_confusers_tmp",
        drop_existing_dataset=True
    )
    
    print(f"Exporting COCO to {COCO_DIR}...")
    dataset.export(
        export_dir=str(COCO_DIR),
        dataset_type=fo.types.YOLOv5Dataset,
        label_field="ground_truth",
        classes=classes
    )
    fo.delete_dataset("coco_confusers_tmp")

if __name__ == "__main__":
    fetch_open_images()
    fetch_coco()
    print("External Data Fetching Complete!")
