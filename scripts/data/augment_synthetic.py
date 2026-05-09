import os
import random
import shutil
import cv2
import albumentations as A
from tqdm import tqdm
from pathlib import Path
import sys

# Add parent directory to sys.path to import config
sys.path.append(str(Path(__file__).resolve().parent.parent))
import config

def get_augmentation_pipeline():
    """Defines the synthetic weather and low-light pipeline."""
    return A.Compose([
        A.OneOf([
            A.RandomRain(brightness_coefficient=0.9, drop_length=15, p=1),
            A.RandomFog(fog_coef_lower=0.3, fog_coef_upper=0.5, alpha_coef=0.08, p=1),
        ], p=0.7), # 70% chance of rain or fog
        A.RandomBrightnessContrast(brightness_limit=(-0.4, -0.2), contrast_limit=(-0.2, 0.2), p=0.8), # Night/Low-light
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.5), # CCTV sensor noise
        A.MotionBlur(blur_limit=7, p=0.3), # Moving camera/subjects
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

def process_split(split, p=0.2):
    """Applies augmentation to a percentage of the dataset split."""
    img_dir = config.YOLO_DATASET_DIR / split / "images"
    lbl_dir = config.YOLO_DATASET_DIR / split / "labels"
    
    all_images = [f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    num_to_augment = int(len(all_images) * p)
    to_augment = random.sample(all_images, num_to_augment)
    
    print(f"\n--- Augmenting {split} split ({num_to_augment} images) ---")
    
    aug_pipeline = get_augmentation_pipeline()
    
    for img_name in tqdm(to_augment):
        img_path = img_dir / img_name
        lbl_path = lbl_dir / f"{os.path.splitext(img_name)[0]}.txt"
        
        if not lbl_path.exists():
            continue
            
        # Read image
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Read labels
        bboxes = []
        class_labels = []
        with open(lbl_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    class_labels.append(int(parts[0]))
                    bboxes.append([float(x) for x in parts[1:5]])
        
        # Apply Augmentation
        try:
            transformed = aug_pipeline(image=image, bboxes=bboxes, class_labels=class_labels)
            transformed_image = transformed['image']
            transformed_bboxes = transformed['bboxes']
            
            # Save Augmented Image
            new_img_name = f"syn_{img_name}"
            new_img_path = img_dir / new_img_name
            cv2.imwrite(str(new_img_path), cv2.cvtColor(transformed_image, cv2.COLOR_RGB2BGR))
            
            # Save Augmented Label
            new_lbl_name = f"syn_{os.path.splitext(img_name)[0]}.txt"
            new_lbl_path = lbl_dir / new_lbl_name
            with open(new_lbl_path, "w") as f:
                for cls, bbox in zip(class_labels, transformed_bboxes):
                    f.write(f"{cls} {' '.join([f'{x:.6f}' for x in bbox])}\n")
                    
        except Exception as e:
            # Some augmentations might fail if bboxes are too small/invalid
            continue

def main():
    print("Starting Synthetic Injection Pipeline...")
    random.seed(config.FO_SEED)
    
    # 20% for Train, 10% for Test/Val for evaluation
    process_split("train", p=0.2)
    process_split("val", p=0.1)
    process_split("test", p=0.1)
    
    print("\nSUCCESS: Synthetic images and labels generated with 'syn_' prefix.")

if __name__ == "__main__":
    main()
