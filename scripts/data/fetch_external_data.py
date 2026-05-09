import os
import sys
from pathlib import Path
import fiftyone as fo
import fiftyone.zoo as foz

# Add the parent directory to sys.path to import config
sys.path.append(str(Path(__file__).resolve().parent.parent))
import config

def fetch_from_zoo(dataset_name, splits, classes, max_samples, output_dir, tmp_name):
    """Generic function to fetch and export data from FiftyOne Zoo."""
    print(f"\n--- Fetching {dataset_name} ({', '.join(classes)}) ---")
    
    # Check if we should use multiple splits or a single one
    load_kwargs = {
        "label_types": ["detections"],
        "classes": classes,
        "max_samples": max_samples,
        "dataset_name": tmp_name,
        "drop_existing_dataset": True
    }
    
    if len(splits) > 1:
        load_kwargs["splits"] = splits
    else:
        load_kwargs["split"] = splits[0]

    try:
        # dataset = foz.load_zoo_dataset(dataset_name, **load_kwargs)
        
        # print(f"Exporting to {output_dir} in YOLO format...")
        # dataset.export(
        #     export_dir=str(output_dir),
        #     dataset_type=fo.types.YOLOv5Dataset,
        #     label_field="ground_truth",
        #     classes=classes
        # )

        dataset = foz.load_zoo_dataset(dataset_name, **load_kwargs)

        jpg_ids = [
            sample.id
            for sample in dataset
            if sample.filepath.lower().endswith((".jpg" ,".JPG"))
        ]

        dataset = dataset.select(jpg_ids)

        dataset.export(
            export_dir=str(output_dir),
            dataset_type=fo.types.YOLOv5Dataset,
            label_field="ground_truth",
            classes=classes
        )
        
        fo.delete_dataset(tmp_name)
        print(f"Successfully fetched {dataset.count()} samples.")
    except Exception as e:
        print(f"Error fetching {dataset_name}: {e}")

def main():
    # 1. Fetch Hard Negatives from Open Images
    oi_classes = ["Power tool", "Hammer", "Screwdriver", "Mobile phone", "Umbrella", "Handbag", "Remote control","Wrench","Scissors","Flashlight","Bottle","Toy"]
    fetch_from_zoo(
        dataset_name="open-images-v7",
        splits=["train", "validation", "test"],
        classes=oi_classes,
        max_samples=config.FO_MAX_SAMPLES,
        output_dir=config.DS_OI_HARD_NEG,
        tmp_name="oi_hard_neg_tmp"
    )

    # 2. Fetch Hard Negatives from COCO
    fetch_from_zoo(
        dataset_name="coco-2017",
        splits=["train", "validation"],
        classes=["person"],
        max_samples=config.FO_MAX_SAMPLES,
        output_dir=config.DS_COCO_HARD_NEG,
        tmp_name="coco_hard_neg_tmp"
    )

    # 3. Fetch Additional Weapons from Open Images
    weapon_classes = ["Handgun", "Shotgun", "Rifle", "Knife"]
    fetch_from_zoo(
        dataset_name="open-images-v7",
        splits=["train", "validation", "test"],
        classes=weapon_classes,
        max_samples=config.FO_MAX_SAMPLES,
        output_dir=config.DS_ADDITIONAL_WEAPONS,
        tmp_name="oi_weapons_tmp"
    )

    print("\nExternal Data Fetching Complete!")

if __name__ == "__main__":
    main()
