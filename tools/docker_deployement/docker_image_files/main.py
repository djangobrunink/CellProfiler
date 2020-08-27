import mimetypes
import os
from pathlib import Path

from detectron2.config import CfgNode as CN
from detectron2.engine import DefaultPredictor
from detectron2.data import detection_utils as utils
import glob
import pickle


def main():
    cfg = CN(CN.load_yaml_with_base(f"config.yaml"))
    cfg.MODEL.WEIGHTS = f"model_final.pth"

    predictor = DefaultPredictor(cfg)

    for image_path in get_all_images("images"):
        image = utils.read_image(image_path, format="BGR")
        prediction = predictor(image)["instances"]

        pred_boxes = prediction.pred_boxes.tensor.numpy()
        scores = prediction.scores.numpy()
        pred_masks = prediction.pred_masks.numpy()
        pred_classes = prediction.pred_classes.numpy()
        num_instances = len(scores)
        file_name = str(Path(image_path).name)
        file_name_without_extension, _ = os.path.splitext(file_name)

        prediction_data = {
            "file_name": file_name,
            "num_instances": num_instances,
            "pred_classes": pred_classes,
            "pred_masks": pred_masks,
            "scores": scores,
            "pred_boxes": pred_boxes,
        }

        with open(f"images/{file_name_without_extension}.p", "wb") as f:
            pickle.dump(prediction_data, f)



def get_all_images(path):
    image_extensions = set(k for k,v in mimetypes.types_map.items() if v.startswith('image/'))

    paths = []
    for image_extension in image_extensions:
        for file_path in glob.glob(f"{path}/*{image_extension}"):
            paths.append(file_path)

    return paths


if __name__ == '__main__':
    main()