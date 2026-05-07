import os
import shutil
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
OUTPUT_DIR = BASE_DIR / "data" / "processed" / "yolo_dataset"

# Source Datasets
DS_MERGED = RAW_DIR / "dataset_merged"
DS_ARCHIVE = RAW_DIR / "archive"
DS_SIMULETIC = RAW_DIR / "Simuletic_Weapon_Umbrella_Dataset"
DS_YOLOV11 = RAW_DIR / "Weapon-detection.v1i.yolov11"
DS_OI = RAW_DIR / "hard_negatives_oi"
DS_COCO = RAW_DIR / "hard_negatives_coco"

# Target classes
# 0: Weapon
# 1: Person
# 2: Confuser

def setup_dirs():
    for split in ["train", "val", "test"]:
        for dtype in ["images", "labels"]:
            (OUTPUT_DIR / split / dtype).mkdir(parents=True, exist_ok=True)

def process_label(label_path, out_label_path, class_map, map_all_to=None):
    if not label_path.exists():
        return
    
    valid = True
    with open(label_path, "r") as f:
        lines = f.readlines()
        
    out_lines = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) >= 5:
            cls_id = int(parts[0])
            
            if map_all_to is not None:
                new_cls = map_all_to
            elif cls_id in class_map:
                new_cls = class_map[cls_id]
            else:
                new_cls = 0 # Default fallback
                
            if new_cls not in [0, 1, 2]:
                valid = False
            parts[0] = str(new_cls)
            out_lines.append(" ".join(parts))
    
    if not valid:
        print(f"WARNING: Invalid class detected in {label_path}")

    with open(out_label_path, "w") as f:
        f.write("\n".join(out_lines))
        
    return valid

def copy_data(src_images, src_labels, split, class_map=None, map_all_to=None, prefix=""):
    if not src_images.exists():
        return 0, 0
    
    img_count = 0
    label_count = 0
    for img_name in os.listdir(src_images):
        if not img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue
            
        src_img = src_images / img_name
        base_name = os.path.splitext(img_name)[0]
        src_lbl = src_labels / f"{base_name}.txt"
        
        # New names
        new_img_name = f"{prefix}_{img_name}"
        new_lbl_name = f"{prefix}_{base_name}.txt"
        
        dst_img = OUTPUT_DIR / split / "images" / new_img_name
        dst_lbl = OUTPUT_DIR / split / "labels" / new_lbl_name
        
        shutil.copy(src_img, dst_img)
        img_count += 1
        
        if src_lbl.exists():
            process_label(src_lbl, dst_lbl, class_map, map_all_to)
            label_count += 1
            
    return img_count, label_count

def generate_yaml():
    yaml_content = """path: ../../data/processed/yolo_dataset
train: train/images
val: val/images
test: test/images

nc: 3
names:
  0: Weapon
  1: Person
  2: Confuser
"""
    with open(OUTPUT_DIR / "data.yaml", "w") as f:
        f.write(yaml_content)

def main():
    print("Setting up unified dataset directories...")
    setup_dirs()
    
    total_imgs = 0
    
    # Source 1: dataset_merged
    print("Processing Source 1: dataset_merged...")
    for split in ["train", "val", "test"]:
        imgs, lbls = copy_data(
            DS_MERGED / split / "images",
            DS_MERGED / split / "labels",
            split,
            class_map={0: 0},
            prefix="ds1"
        )
        total_imgs += imgs
        
    # Source 2: archive (classes 0-8 -> 0)
    print("Processing Source 2: archive...")
    for split in ["train", "val", "test"]:
        base_path = DS_ARCHIVE / "weapon_detection" / split
        if not base_path.exists():
            base_path = DS_ARCHIVE / split
        
        imgs, lbls = copy_data(
            base_path / "images",
            base_path / "labels",
            split,
            class_map={i: 0 for i in range(15)}, # Map all to 0
            prefix="ds2"
        )
        total_imgs += imgs
        
    # Source 3: Simuletic (0->1, 1->0, 2->2)
    print("Processing Source 3: Simuletic...")
    imgs, lbls = copy_data(
        DS_SIMULETIC / "images",
        DS_SIMULETIC / "labels",
        "train", # Defaulting to train
        class_map={0: 1, 1: 0, 2: 2},
        prefix="ds3"
    )
    total_imgs += imgs
    
    # Source 4: Weapon-detection.v1i.yolov11 (existing -> 0)
    print("Processing Source 4: Weapon-detection.v1i.yolov11...")
    for split in ["train", "valid", "test"]:
        out_split = "val" if split == "valid" else split
        imgs, lbls = copy_data(
            DS_YOLOV11 / split / "images",
            DS_YOLOV11 / split / "labels",
            out_split,
            class_map={i: 0 for i in range(15)}, # Map any to 0
            prefix="ds4"
        )
        total_imgs += imgs
        
    # Source 5: Open Images V7 Hard Negatives (all -> 2)
    print("Processing Source 5: Open Images V7 Hard Negatives...")
    for split in ["train", "val", "test"]:
        imgs, lbls = copy_data(
            DS_OI / split / "images",
            DS_OI / split / "labels",
            split,
            map_all_to=2,
            prefix="ds5"
        )
        total_imgs += imgs

    # Source 6: COCO 2017 Hard Negatives (all -> 2)
    print("Processing Source 6: COCO 2017 Hard Negatives...")
    for split in ["train", "val", "test"]:
        imgs, lbls = copy_data(
            DS_COCO / split / "images",
            DS_COCO / split / "labels",
            split,
            map_all_to=2,
            prefix="ds6"
        )
        total_imgs += imgs
        
    print("Generating data.yaml...")
    generate_yaml()
    
    print(f"Done! Aggregated {total_imgs} images into {OUTPUT_DIR}")
    print("Verification complete: Ensure all outputs fall into classes 0, 1, 2 via data.yaml schema mapping.")

if __name__ == "__main__":
    main()
