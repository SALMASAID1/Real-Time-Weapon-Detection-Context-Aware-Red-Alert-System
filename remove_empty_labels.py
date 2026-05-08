import os
import time

dataset_dir = r"c:\Users\pc\Desktop\Real-Time-Weapon-Detection-Context-Aware-Red-Alert-System\data\raw\dataset_merged"
splits = ["train", "val", "test"]

removed_images = 0
removed_labels = 0

print("Starting scan...", flush=True)

for split in splits:
    images_dir = os.path.join(dataset_dir, split, "images")
    labels_dir = os.path.join(dataset_dir, split, "labels")
    
    if not os.path.exists(images_dir) or not os.path.exists(labels_dir):
        print(f"Skipping {split} (does not exist)", flush=True)
        continue
        
    print(f"Processing split: {split}", flush=True)
    
    count = 0
    start_time = time.time()
    
    # Use os.scandir for speed
    with os.scandir(images_dir) as it:
        for entry in it:
            if not entry.is_file():
                continue
                
            count += 1
            if count % 1000 == 0:
                print(f"Processed {count} images in {split} (Time elapsed: {time.time() - start_time:.2f}s)", flush=True)
                
            name, ext = os.path.splitext(entry.name)
            label_path = os.path.join(labels_dir, name + ".txt")
            
            is_empty = True
            if os.path.exists(label_path):
                # Try to check size first, if size is 0 it's empty
                if os.path.getsize(label_path) == 0:
                    is_empty = True
                else:
                    with open(label_path, 'r') as f:
                        content = f.read().strip()
                        if content:
                            is_empty = False
            
            if is_empty:
                # Delete image
                try:
                    os.remove(entry.path)
                    removed_images += 1
                except Exception as e:
                    pass
                    
                # Delete label if it exists
                if os.path.exists(label_path):
                    try:
                        os.remove(label_path)
                        removed_labels += 1
                    except Exception as e:
                        pass

print(f"Removed {removed_images} images with empty or missing labels.", flush=True)
print(f"Removed {removed_labels} empty label files.", flush=True)
