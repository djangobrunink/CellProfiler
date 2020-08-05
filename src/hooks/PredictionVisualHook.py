from pathlib import Path

import detectron2.utils.comm as comm
import torch
from detectron2.data import MetadataCatalog, build_detection_test_loader, DatasetCatalog
from detectron2.data import detection_utils as utils
from detectron2.engine import HookBase
from detectron2.utils.visualizer import Visualizer


class PredictionVisualHook(HookBase):
    def __init__(
            self,
            data_loader,
            n,
            threshold: float = 0.5,
            scale: float = 0.5,
    ) -> None:
        self.data_loader = data_loader
        self.threshold = threshold
        self.n = n
        self.scale = scale

        assert 0 < self.threshold < 1

    def after_step(self):
        next_iter = self.trainer.iter + 1
        is_final = next_iter == self.trainer.max_iter
        eval_period = self.trainer.cfg.TEST.EVAL_PERIOD * 2
        is_eval_period = eval_period > 0 and next_iter % eval_period == 0

        if is_eval_period or is_final:
            self._preform()

    def _preform(self):
        self.trainer.model.eval()
        meta_data = MetadataCatalog.get(self.trainer.cfg.DATASETS.TEST[0])
        loader = iter(self.data_loader)

        i = 0
        with torch.no_grad():
            for i in range(self.n):
                inputs = next(loader)
                outputs = self.trainer.model(inputs)[0]["instances"].to("cpu")
                outputs = outputs[outputs.scores > self.threshold]

                image = utils.read_image(inputs[0]["file_name"])
                v = Visualizer(image, meta_data, scale=self.scale)
                out = v.draw_instance_predictions(outputs)
                predicted_image = out.get_image().transpose(2, 0, 1)

                path = Path(inputs[0]["file_name"])
                name = path.parents[0].name + "/" + path.name
                self.trainer.storage.put_image(name, predicted_image)

            comm.synchronize()

        self.trainer.model.train()

    @classmethod
    def create(cls, cfg, n=None):
        data_loader = build_detection_test_loader(
            cfg,
            cfg.DATASETS.TEST[0],
        )
        if n is None:
            n = len(DatasetCatalog.get(cfg.DATASETS.TEST[0]))
        return PredictionVisualHook(data_loader, n)
