import itertools
import json
import os
from pathlib import Path
from typing import Optional, Union, List

import cv2
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode

TRAIN_DATASET_NAME = "train"
VAL_DATASET_NAME = "val"
TEST_DATASET_NAME = "test"

DEFAULT_CLASS_NAMES: List[str] = ["Death", "Deformed", "Healthy"]


def register_default_datasets(clear: bool = True):
    if clear:
        DatasetCatalog.clear()
        MetadataCatalog.clear()

    root = Path(__file__).parent / "../dataset"

    DatasetCatalog.register(TRAIN_DATASET_NAME, lambda: json_to_dataset(root / "train", DEFAULT_CLASS_NAMES))
    MetadataCatalog.get(TRAIN_DATASET_NAME).set(thing_classes=DEFAULT_CLASS_NAMES)

    DatasetCatalog.register(VAL_DATASET_NAME, lambda: json_to_dataset(root / "val", DEFAULT_CLASS_NAMES))
    MetadataCatalog.get(VAL_DATASET_NAME).set(thing_classes=DEFAULT_CLASS_NAMES)

    DatasetCatalog.register(TEST_DATASET_NAME, lambda: json_to_dataset(root / "test", DEFAULT_CLASS_NAMES))
    MetadataCatalog.get(TEST_DATASET_NAME).set(thing_classes=DEFAULT_CLASS_NAMES)


def json_to_dataset(
        image_dir_path: Union[str, Path],
        class_names=List[str],
        json_path: Optional[Union[str, Path]] = None,

):
    json_path = json_path if json_path is not None else os.path.join(image_dir_path, "via_region_data.json")

    with open(json_path) as f:
        json_data = json.load(f)

    records = []
    for i, (key, image_data) in enumerate(json_data.items()):
        record = dict()

        record["file_name"] = os.path.join(image_dir_path, image_data["filename"])
        record["image_id"] = i
        record["height"], record["width"] = cv2.imread(record["file_name"]).shape[:2]

        record["annotations"] = []
        for region in image_data["regions"]:
            shape_attributes = region["shape_attributes"]
            px = shape_attributes["all_points_x"]
            py = shape_attributes["all_points_y"]

            poly = [(x, y) for x, y in zip(px, py)]
            poly = list(itertools.chain.from_iterable(poly))

            class_name = region["region_attributes"]["class"].strip()
            assert class_name in class_names, f"Unknown class: {class_name} is not in {class_names}"
            category_id = class_names.index(class_name)

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
