import os
import shutil
import random
import sys
from pathlib import Path

# Add the parent directory to sys.path to import config
sys.path.append(str(Path(__file__).resolve().parent.parent))
import config

def setup_dirs():
    """Create the output directory structure."""
    if config.YOLO_DATASET_DIR.exists():
        print(f"Clearing existing output directory: {config.YOLO_DATASET_DIR}")
        shutil.rmtree(config.YOLO_DATASET_DIR)
        
    for split in ["train", "val", "test"]:
        for dtype in ["images", "labels"]:
            (config.YOLO_DATASET_DIR / split / dtype).mkdir(parents=True, exist_ok=True)

def process_label(label_path, out_label_path, class_map, map_all_to=None):
    """Read, map, and write YOLO labels."""
    if not label_path.exists():
        return False
    
    valid = True
    with open(label_path, "r") as f:
        lines = f.readlines()
        
    out_lines = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) >= 5:
            try:
                cls_id = int(parts[0])
                
                if map_all_to is not None:
                    new_cls = map_all_to
                elif class_map and cls_id in class_map:
                    new_cls = class_map[cls_id]
                else:
                    new_cls = config.CLASS_MAP["weapon"] # Default fallback
                    
                if new_cls not in config.CLASS_MAP.values():
                    valid = False
                parts[0] = str(new_cls)
                out_lines.append(" ".join(parts))
            except ValueError:
                valid = False
    
    if not valid:
        print(f"WARNING: Invalid format or class detected in {label_path}")

    with open(out_label_path, "w") as f:
        f.write("\n".join(out_lines))
        
    return True

def collect_data(src_images, src_labels, class_map=None, map_all_to=None, prefix=""):
    """Gather metadata for images and labels in a source directory."""
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
    """Copy files and process labels for a specific split."""
    img_count = 0
    label_count = 0
    
    for item in items:
        new_img_name = f"{item['prefix']}_{item['img_name']}"
        new_lbl_name = f"{item['prefix']}_{item['base_name']}.txt"
        
        dst_img = config.YOLO_DATASET_DIR / split / "images" / new_img_name
        dst_lbl = config.YOLO_DATASET_DIR / split / "labels" / new_lbl_name
        
        shutil.copy(item['src_img'], dst_img)
        img_count += 1
        
        if item['src_lbl']:
            if process_label(item['src_lbl'], dst_lbl, item['class_map'], item['map_all_to']):
                label_count += 1
            
    return img_count, label_count

def generate_yaml():
    """Create the data.yaml file for YOLOv11."""
    yaml_content = f"""path: {config.YOLO_DATASET_DIR.relative_to(config.BASE_DIR)}
train: train/images
val: val/images
test: test/images

nc: {len(config.CLASS_MAP)}
names:
"""
    sorted_names = sorted(config.CLASS_MAP.items(), key=lambda x: x[1])
    for name, idx in sorted_names:
        yaml_content += f"  {idx}: {name.capitalize()}\n"

    with open(config.YOLO_DATASET_DIR / "data.yaml", "w") as f:
        f.write(yaml_content)

def main():
    print("Initializing Unified Dataset Assembly...")
    setup_dirs()
    
    global_items = []
    
    # Define Source Configurations
    # Format: (Path, ClassMap, MapAllTo, Prefix)
    sources = [
        (config.DS_MERGED, {0: 0}, None, "ds1"),
        (config.DS_ARCHIVE, {i: 0 for i in range(15)}, None, "ds2"),
        (config.DS_SIMULETIC, {0: 1, 1: 0, 2: 2}, None, "ds3"),
        (config.DS_OI_HARD_NEG, None, 2, "ds4"),
        (config.DS_COCO_HARD_NEG, None, 2, "ds5"),
        (config.DS_ADDITIONAL_WEAPONS, None, 0, "ds6")
    ]
    
    for i, (path, cls_map, map_all, prefix) in enumerate(sources):
        print(f"[{i+1}/{len(sources)}] Collecting from: {path.name}...")
        
        if not path.exists():
            print(f"  WARNING: Path does not exist: {path}")
            continue

        current_source_items = []
        
        # Check for subdirectories (train/val/test or images/val etc.)
        possible_splits = ["train", "val", "test", "valid"]
        found_any_split = False

        for split in possible_splits:
            # Type A: path/split/images
            img_dir_a = path / split / "images"
            lbl_dir_a = path / split / "labels"
            
            # Type B: path/images/split
            img_dir_b = path / "images" / split
            lbl_dir_b = path / "labels" / split

            if img_dir_a.exists():
                found_any_split = True
                current_source_items.extend(collect_data(img_dir_a, lbl_dir_a, cls_map, map_all, prefix))
            elif img_dir_b.exists():
                found_any_split = True
                current_source_items.extend(collect_data(img_dir_b, lbl_dir_b, cls_map, map_all, prefix))

        # Fallback for flat structure (no train/val/test subdirs)
        if not found_any_split:
            img_dir = path / "images"
            lbl_dir = path / "labels"
            if not img_dir.exists():
                img_dir = path
                lbl_dir = path
            current_source_items.extend(collect_data(img_dir, lbl_dir, cls_map, map_all, prefix))

        print(f"  Found {len(current_source_items)} images.")
        global_items.extend(current_source_items)

    print(f"\nTotal aggregated images: {len(global_items)}")
    
    if not global_items:
        print("CRITICAL ERROR: No images found! Dataset assembly aborted.")
        return

    print("Applying global shuffle...")
    random.seed(config.FO_SEED)
    random.shuffle(global_items)
    
    # Split logic
    total_len = len(global_items)
    r = config.SPLIT_RATIO
    train_end = int(total_len * r["train"])
    val_end = int(total_len * (r["train"] + r["val"]))
    
    train_items = global_items[:train_end]
    val_items = global_items[train_end:val_end]
    test_items = global_items[val_end:]
    
    print(f"Splits: Train={len(train_items)}, Val={len(val_items)}, Test={len(test_items)}")
    
    splits = [("train", train_items), ("val", val_items), ("test", test_items)]
    
    for name, items in splits:
        print(f"Writing {name} split...")
        write_data(items, name)
        
    print("Generating metadata...")
    generate_yaml()
    
    print(f"\nSUCCESS! Unified dataset assembled at: {config.YOLO_DATASET_DIR}")

if __name__ == "__main__":
    main()
