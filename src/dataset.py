import itertools
import json
import os
from pathlib import Path
from typing import List, Dict

import cv2
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode

TRAIN_DATASET_NAME = "train"
VAL_DATASET_NAME = "val"
TEST_DATASET_NAME = "test"

DEFAULT_CLASS_NAMES: List[str] = ["Healthy", "Deformed", "Death", "Yolk", "Heart" , "Yolk_Deformed", "Balloon", "Disintegrated", "Undeveloped"]
DEFAULT_MAPPING: Dict[str, int] = {
    "Healthy": 0,
    "Deformed": 1,
    "Death": 2,
    "Yolk": 3,
    "Heart": 4,
    "Yolk_Deformed": 5,
    "Balloon": 6,
    "Disintegrated": 7,
    "Undeveloped": 8,
}
COLLAPSED_MAPPING: Dict[str, int] = {
    "Healthy": 0,
    "Deformed": 1,
    "Death": 2,
    "Yolk": 3,
    "Heart": 1,
    "Yolk_Deformed": 1,
    "Balloon": 1,
    "Disintegrated": 2,
    "Undeveloped": 2,
}
    
    
COLLAPSED_CLASS_NAMES: List[str] = ["Healthy", "Deformed", "Death", "Yolk"]


def register_default_datasets(clear: bool = True):
    if clear:
        DatasetCatalog.clear()
        MetadataCatalog.clear()


    root = Path(__file__).parent / "../dataset"
    
    # Train
    DatasetCatalog.register(TRAIN_DATASET_NAME, lambda: read_dataset(root / "train", DEFAULT_MAPPING))
    MetadataCatalog.get(TRAIN_DATASET_NAME).set(thing_classes=DEFAULT_CLASS_NAMES)

    DatasetCatalog.register(TRAIN_DATASET_NAME + "_collapsed", lambda: read_dataset(root / "train", COLLAPSED_MAPPING))
    MetadataCatalog.get(TRAIN_DATASET_NAME + "_collapsed").set(thing_classes=COLLAPSED_CLASS_NAMES)

    # Val
    DatasetCatalog.register(VAL_DATASET_NAME, lambda: read_dataset(root / "val", DEFAULT_MAPPING))
    MetadataCatalog.get(VAL_DATASET_NAME).set(thing_classes=DEFAULT_CLASS_NAMES)

    DatasetCatalog.register(VAL_DATASET_NAME + "_collapsed", lambda: read_dataset(root / "val", COLLAPSED_MAPPING))
    MetadataCatalog.get(VAL_DATASET_NAME + "_collapsed").set(thing_classes=COLLAPSED_CLASS_NAMES)

    # Test
    DatasetCatalog.register(TEST_DATASET_NAME, lambda: read_dataset(root / "test", DEFAULT_MAPPING))
    MetadataCatalog.get(TEST_DATASET_NAME).set(thing_classes=DEFAULT_CLASS_NAMES)

    DatasetCatalog.register(TEST_DATASET_NAME + "_collapsed", lambda: read_dataset(root / "test", COLLAPSED_MAPPING))
    MetadataCatalog.get(TEST_DATASET_NAME + "_collapsed").set(thing_classes=COLLAPSED_CLASS_NAMES)


def read_dataset(
    root: Path,
    class_name_mapping: Dict[str, int],
):
    records = []

    for i, annotation_path in enumerate(root.glob("*.json")):
        with open(annotation_path) as f:
            annotation_data = json.load(f)

        record = dict()

        record["file_name"] = os.path.join(root, annotation_data["filename"])
        record["image_id"] = i
        record["height"], record["width"] = cv2.imread(record["file_name"]).shape[:2]

        record["annotations"] = []
        assert annotation_data["file_attributes"]["overlap_problematic"] in ["True", "False"]
        overlap_is_problematic = annotation_data["file_attributes"]["overlap_problematic"] == "True"
        if overlap_is_problematic:
           continue

        for region in annotation_data["regions"]:
            shape_attributes = region["shape_attributes"]
            px = shape_attributes["all_points_x"]
            py = shape_attributes["all_points_y"]

            poly = [(x + 0.5, y + 0.5) for x, y in zip(px, py)]
            poly = list(itertools.chain.from_iterable(poly))

            class_name = region["region_attributes"]["class"].strip()
            assert class_name in class_name_mapping, f"Unknown class: {class_name} is not in {class_name_mapping}"
            category_id = class_name_mapping[class_name]

            annotation = {
                "bbox": [min(px), min(py), max(px), max(py)],
                "bbox_mode": BoxMode.XYXY_ABS,
                "segmentation": [poly],
                "category_id": category_id,
                "iscrowd": 0
            }
            record["annotations"].append(annotation)

        records.append(record)

    return records


if __name__ == '__main__':
    register_default_datasets()

    print(len(DatasetCatalog.get("train")))