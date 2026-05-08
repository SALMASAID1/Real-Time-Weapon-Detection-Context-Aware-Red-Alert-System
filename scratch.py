import fiftyone.zoo as foz
print("testing")
dataset = foz.load_zoo_dataset(
    "open-images-v7",
    splits=["validation", "test"],
    label_types=["detections"],
    classes=["Handgun"],
    max_samples=10,
    dataset_name="test_tmp",
    drop_existing_dataset=True
)
print(dataset.count())
import fiftyone as fo
fo.delete_dataset("test_tmp")
