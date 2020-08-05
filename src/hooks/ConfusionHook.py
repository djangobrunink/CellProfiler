import io

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn
import torch
from detectron2.data import MetadataCatalog, build_detection_test_loader, DatasetMapper, DatasetCatalog
from detectron2.data import transforms as T
from detectron2.engine import HookBase

from src.visualization import create_confusion_matrix


class ConfusionHook(HookBase):
    def __init__(
            self,
            data_loader,
            n,
            threshold: float = 0.75,
    ) -> None:
        self.data_loader = data_loader
        self.n = n
        self.threshold = threshold

        assert 0 < self.threshold < 1

    def after_step(self):
        next_iter = self.trainer.iter + 1
        is_final = next_iter == self.trainer.max_iter
        eval_period = self.trainer.cfg.TEST.EVAL_PERIOD * 2
        is_eval_period = eval_period > 0 and next_iter % eval_period == 0

        if is_eval_period or is_final:
            print("@@@@", is_eval_period, eval_period, next_iter, "@@@@")
            self._preform()

    def _preform(self):
        self.trainer.model.eval()
        meta_data = MetadataCatalog.get(self.trainer.cfg.DATASETS.TEST[0])
        loader = iter(self.data_loader)

        ground_truths = []
        predictions = []

        with torch.no_grad():
            for i in range(self.n):
                inputs = next(loader)
                ground_truths.append(inputs[0])

                outputs = self.trainer.model(inputs)[0]["instances"].to("cpu")
                outputs = outputs[outputs.scores > self.threshold]
                predictions.append(outputs)

        m, labels = create_confusion_matrix(ground_truths, predictions, meta_data)
        df = pd.DataFrame(
            m,
            index=list(map(lambda x: "Pred " + x, labels)),
            columns=list(map(lambda x: "GT " + x, labels))
        )

        title = f"Confusion matrix {self.trainer.cfg.DATASETS.TEST[0]}"
        plt.figure(figsize=(7, 7))
        sn.heatmap(df, annot=True)

        plt.title(title)
        buf = io.BytesIO()
        plt.savefig(buf, format='jpg')
        buf.seek(0)
        image = cv2.imdecode(np.frombuffer(buf.getvalue(), np.uint8), -1)
        image = image.transpose(2, 0, 1)

        self.trainer.storage.put_image(title, image)
        self.trainer.model.train()
        plt.close()

    @classmethod
    def create(cls, cfg, *, threshold=0.75):
        mapper = DatasetMapper(
            is_train=True,
            augmentations=[
                T.ResizeShortestEdge(
                    short_edge_length=cfg.INPUT.MIN_SIZE_TEST,
                    max_size=cfg.INPUT.MAX_SIZE_TEST,
                    sample_style=cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING)
            ],
            image_format=cfg.INPUT.FORMAT,
            use_instance_mask=cfg.MODEL.MASK_ON,
            instance_mask_format=cfg.INPUT.MASK_FORMAT,
            use_keypoint=cfg.MODEL.KEYPOINT_ON,
            recompute_boxes=True,
        )

        data_loader = build_detection_test_loader(
            cfg,
            cfg.DATASETS.TEST[0],
            mapper=mapper,
        )
        n = len(DatasetCatalog.get(cfg.DATASETS.TEST[0]))
        return ConfusionHook(data_loader, n=n, threshold=threshold)
