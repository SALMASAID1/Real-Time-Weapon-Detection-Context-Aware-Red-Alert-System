import fiftyone as fo
import fiftyone.zoo as foz

def count_classes():
    classes = ["Handgun", "Shotgun", "Rifle", "Knife"]
    for split in ["train", "validation", "test"]:
        try:
            dataset = foz.load_zoo_dataset(
                "open-images-v7",
                split=split,
                label_types=["detections"],
                classes=classes,
                max_samples=100,
                dataset_name=f"test_cnt_{split}",
                drop_existing_dataset=True
            )
            print(f"{split}: loaded {dataset.count()}")
            fo.delete_dataset(f"test_cnt_{split}")
        except Exception as e:
            print(f"Error on {split}: {e}")

if __name__ == "__main__":
    count_classes()
