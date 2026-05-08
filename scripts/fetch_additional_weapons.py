import os
import fiftyone as fo
import fiftyone.zoo as foz
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
ADDITIONAL_WEAPONS_DIR = RAW_DIR / "additional_weapons_oi"

def fetch_additional_weapons():
    print("Fetching Additional Weapons from Open Images V7...")
    classes = ["Handgun", "Shotgun", "Rifle", "Knife"]
    
    # Load the dataset
    dataset = foz.load_zoo_dataset(
        "open-images-v7",
        splits=["train", "validation", "test"],
        label_types=["detections"],
        classes=classes,
        max_samples=5000, 
        dataset_name="oi_weapons_tmp",
        drop_existing_dataset=True
    )
    
    print(f"Exporting Open Images to {ADDITIONAL_WEAPONS_DIR} in YOLO format...")
    dataset.export(
        export_dir=str(ADDITIONAL_WEAPONS_DIR),
        dataset_type=fo.types.YOLOv5Dataset,
        label_field="ground_truth",
        classes=classes
    )
    
    # Clean up the temporary dataset from FiftyOne's internal DB
    fo.delete_dataset("oi_weapons_tmp")
    print("Additional Weapons fetch complete!")

if __name__ == "__main__":
    fetch_additional_weapons()
