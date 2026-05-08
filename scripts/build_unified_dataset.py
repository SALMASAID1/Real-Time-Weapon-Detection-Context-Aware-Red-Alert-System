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
DS_ADDITIONAL_WEAPONS = RAW_DIR / "additional_weapons_oi"

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

def collect_data(src_images, src_labels, class_map=None, map_all_to=None, prefix=""):
    items = []
    if not src_images.exists():
        return items
    
    for img_name in os.listdir(src_images):
        if not img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue
            
        src_img = src_images / img_name
        base_name = os.path.splitext(img_name)[0]
        src_lbl = src_labels / f"{base_name}.txt"
        
        items.append({
            "src_img": src_img,
            "src_lbl": src_lbl if src_lbl.exists() else None,
            "prefix": prefix,
            "base_name": base_name,
            "img_name": img_name,
            "class_map": class_map,
            "map_all_to": map_all_to
        })
            
    return items

def write_data(items, split):
    img_count = 0
    label_count = 0
    
    for item in items:
        new_img_name = f"{item['prefix']}_{item['img_name']}"
        new_lbl_name = f"{item['prefix']}_{item['base_name']}.txt"
        
        dst_img = OUTPUT_DIR / split / "images" / new_img_name
        dst_lbl = OUTPUT_DIR / split / "labels" / new_lbl_name
        
        shutil.copy(item['src_img'], dst_img)
        img_count += 1
        
        if item['src_lbl']:
            process_label(item['src_lbl'], dst_lbl, item['class_map'], item['map_all_to'])
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
    import random
    print("Setting up unified dataset directories...")
    # Clear output dir if exists to ensure clean run
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    setup_dirs()
    
    global_items = []
    
    # Source 1: dataset_merged
    print("Collecting from Source 1: dataset_merged...")
    for split in ["train", "val", "test"]:
        items = collect_data(
            DS_MERGED / split / "images",
            DS_MERGED / split / "labels",
            class_map={0: 0},
            prefix="ds1"
        )
        global_items.extend(items)
        
    # Source 2: archive (classes 0-8 -> 0)
    print("Collecting from Source 2: archive...")
    for split in ["train", "val", "test"]:
        base_path = DS_ARCHIVE / "weapon_detection" / split
        if not base_path.exists():
            base_path = DS_ARCHIVE / split
        
        items = collect_data(
            base_path / "images",
            base_path / "labels",
            class_map={i: 0 for i in range(15)}, # Map all to 0
            prefix="ds2"
        )
        global_items.extend(items)
        
    # Source 3: Simuletic (0->1, 1->0, 2->2)
    print("Collecting from Source 3: Simuletic...")
    items = collect_data(
        DS_SIMULETIC / "images",
        DS_SIMULETIC / "labels",
        class_map={0: 1, 1: 0, 2: 2},
        prefix="ds3"
    )
    global_items.extend(items)
    
    # Source 4: Weapon-detection.v1i.yolov11 (existing -> 0)
    print("Collecting from Source 4: Weapon-detection.v1i.yolov11...")
    for split in ["train", "valid", "test"]:
        items = collect_data(
            DS_YOLOV11 / split / "images",
            DS_YOLOV11 / split / "labels",
            class_map={i: 0 for i in range(15)}, # Map any to 0
            prefix="ds4"
        )
        global_items.extend(items)
        
    # Source 5: Open Images V7 Hard Negatives (all -> 2)
    print("Collecting from Source 5: Open Images V7 Hard Negatives...")
    for split in ["train", "val", "test"]:
        # FiftyOne YOLO export creates `images/val`, `images/train`, etc.
        img_dir = DS_OI / "images" / split
        lbl_dir = DS_OI / "labels" / split
        if not img_dir.exists():
            img_dir = DS_OI / split / "images"
            lbl_dir = DS_OI / split / "labels"
        if not img_dir.exists():
            if split == "train":
                img_dir = DS_OI / "images"
                lbl_dir = DS_OI / "labels"
            else:
                continue
                
        items = collect_data(
            img_dir,
            lbl_dir,
            map_all_to=2,
            prefix="ds5"
        )
        global_items.extend(items)

    # Source 6: COCO 2017 Hard Negatives (all -> 2)
    print("Collecting from Source 6: COCO 2017 Hard Negatives...")
    for split in ["train", "val", "test"]:
        img_dir = DS_COCO / "images" / split
        lbl_dir = DS_COCO / "labels" / split
        if not img_dir.exists():
            img_dir = DS_COCO / split / "images"
            lbl_dir = DS_COCO / split / "labels"
        if not img_dir.exists():
            if split == "train":
                img_dir = DS_COCO / "images"
                lbl_dir = DS_COCO / "labels"
            else:
                continue
                
        items = collect_data(
            img_dir,
            lbl_dir,
            map_all_to=2,
            prefix="ds6"
        )
        global_items.extend(items)
        
    # Source 7: Additional Weapons from OI (all -> 0)
    print("Collecting from Source 7: Additional Weapons OI...")
    for split in ["train", "val", "test"]:
        img_dir = DS_ADDITIONAL_WEAPONS / "images" / split
        lbl_dir = DS_ADDITIONAL_WEAPONS / "labels" / split
        if not img_dir.exists():
            img_dir = DS_ADDITIONAL_WEAPONS / split / "images"
            lbl_dir = DS_ADDITIONAL_WEAPONS / split / "labels"
        if not img_dir.exists():
            if split == "train":
                img_dir = DS_ADDITIONAL_WEAPONS / "images"
                lbl_dir = DS_ADDITIONAL_WEAPONS / "labels"
            else:
                continue
                
        items = collect_data(
            img_dir,
            lbl_dir,
            map_all_to=0,
            prefix="ds7"
        )
        global_items.extend(items)

    print(f"Total collected images: {len(global_items)}")
    print("Shuffling global dataset...")
    # Fix seed for reproducibility
    random.seed(42)
    random.shuffle(global_items)
    
    # 80/10/10 Split
    total_len = len(global_items)
    train_end = int(total_len * 0.8)
    val_end = int(total_len * 0.9)
    
    train_items = global_items[:train_end]
    val_items = global_items[train_end:val_end]
    test_items = global_items[val_end:]
    
    print(f"Split sizes: Train={len(train_items)}, Val={len(val_items)}, Test={len(test_items)}")
    
    print("Writing Train Split...")
    write_data(train_items, "train")
    
    print("Writing Val Split...")
    write_data(val_items, "val")
    
    print("Writing Test Split...")
    write_data(test_items, "test")
        
    print("Generating data.yaml...")
    generate_yaml()
    
    print(f"Done! Aggregated {len(global_items)} images into {OUTPUT_DIR}")
    print("Verification complete: Ensure all outputs fall into classes 0, 1, 2 via data.yaml schema mapping.")

if __name__ == "__main__":
    main()
