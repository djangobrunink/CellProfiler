import glob
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import NamedTuple, Dict, List, Optional
from sklearn.model_selection import train_test_split
import numpy as np
from tqdm import trange

Item = NamedTuple("Item", [('class_counts', Dict[str, int]), ('annotation_path', Path), ("image_path", Path), ("n_overlaps", int)])

def _copy(self, target):
    import shutil
    assert self.is_file()
    shutil.copy(str(self), str(target))

Path.copy = _copy

class Dataset:
    def __init__(self, items: List[Item] = None):
        self.items = items if items is not None else list()

    def distribution(self, class_names: Optional[List[str]] = None):
        class_names = [] if class_names is None else class_names
        res = {k:0 for k in class_names}
        total = 0
        for item in self.items:
            for class_name, count in item.class_counts.items():
                if class_name not in res:
                    res[class_name] = 0

                res[class_name] += count
                total += count

        return {k:v / total for k,v in res.items()}

    def n_overlaps(self):
        return sum(map(lambda item: item.n_overlaps, self.items))

    def n_overlaps_distrubition(self):
        return self.n_overlaps() / len(self.items)

    def counts(self, class_names: Optional[List[str]] = None):
        class_names = [] if class_names is None else class_names
        res = {k:0 for k in class_names}
        for item in self.items:
            for class_name, count in item.class_counts.items():
                if class_name not in res:
                    res[class_name] = 0

                res[class_name] += count

        return {k:v for k,v in res.items()}

def main():
    class_names = ["Healthy", "Deformed", "Death", "Heart" , "Yolk", "Balloon", "Disintegrated", "Undeveloped"]
    data = []
    total = []


    for annotation_path in glob.glob("./**/*.json"):
        annotation_path = Path(annotation_path)

        with open(annotation_path) as f:
            annotation = json.load(f)
            image_path = annotation_path.parents[0] / annotation["filename"]

            class_counts = {k: 0 for k in class_names}
            n_overlaps = int(annotation["file_attributes"]["n_overlaps"])

            for region in annotation["regions"]:
                class_name = region["region_attributes"]["class"].strip()
                assert class_name in class_names, f"Unknown class_name={class_name} in {annotation['filename']}"
                class_counts[class_name] += 1
                total.append(class_name)

            data.append(Item(
                class_counts=class_counts,
                annotation_path=annotation_path,
                image_path=image_path,
                n_overlaps=n_overlaps,
            ))


    total_count = Counter(total)
    print("total")
    print(total_count)

    print( {k: v / sum(total_count.values()) for k, v in total_count.items()})

    best_dataset = None
    for _ in trange(30000):
        train, test = train_test_split(data, test_size=0.30)
        val, test = train_test_split(test, test_size=0.5)

        train_ds = Dataset(train)
        val_ds = Dataset(val)
        test_ds = Dataset(test)
        datasets = [train_ds, val_ds, test_ds]
        if best_dataset is None:
            best_dataset = datasets

        best_dataset = min(best_dataset, datasets, key=score)


    print(score(best_dataset))
    train_ds, val_ds, test_ds = best_dataset
    print("train_ds", train_ds.distribution())
    print("train_ds", train_ds.counts())
    print("train_ds", train_ds.n_overlaps_distrubition(), train_ds.n_overlaps())

    print("val_ds", val_ds.distribution())
    print("val_ds", val_ds.counts())
    print("val_ds", val_ds.n_overlaps_distrubition(), val_ds.n_overlaps())

    print("test_ds", test_ds.distribution())
    print("test_ds", test_ds.counts())
    print("test_ds", test_ds.n_overlaps_distrubition(), test_ds.n_overlaps())


    output_folders = [Path() / "train2", Path() / "val2", Path() / "test2"]

    for output_folder, ds in zip(output_folders, [train_ds, val_ds, test_ds]):
        os.makedirs(output_folder)
        for item in ds.items:
            item.annotation_path.copy(output_folder / item.annotation_path.name)
            item.image_path.copy(output_folder / item.image_path.name)




def score(datasets: List[Dataset]):
    result = defaultdict(lambda: - np.inf)

    for i, dataset in enumerate(datasets):
        for other_dataset in datasets[i:]:
            distribution = dataset.distribution()
            other_distribution = other_dataset.distribution()
            for k in distribution:
                diff = abs(distribution[k] - other_distribution[k])
                if diff > result[k]:
                    result[k] = diff

            overlap_diff = abs(dataset.n_overlaps_distrubition() - other_dataset.n_overlaps_distrubition())
            if overlap_diff > result["overlap"]:
                result["overlap"] = overlap_diff


    return sum(dict(result).values())



if __name__ == '__main__':
    main()
