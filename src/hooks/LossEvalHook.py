import datetime
import logging
import time

import detectron2.utils.comm as comm
import numpy as np
import torch
from detectron2.data import DatasetMapper, build_detection_test_loader
from detectron2.data import transforms as T
from detectron2.engine import HookBase
from detectron2.utils.logger import log_every_n_seconds, setup_logger

setup_logger()


class LossEvalHook(HookBase):
    def __init__(
            self,
            data_loader,
    ) -> None:
        self._data_loader = data_loader

    def after_step(self):
        next_iter = self.trainer.iter + 1
        is_final = next_iter == self.trainer.max_iter
        is_eval_period = self.trainer.cfg.TEST.EVAL_PERIOD > 0 and next_iter % self.trainer.cfg.TEST.EVAL_PERIOD == 0

        if is_eval_period or is_final:
            print("@@@ Evaluation @@@", self.trainer.iter)
            self._do_loss_eval()

    def _do_loss_eval(self):
        # Copying inference_on_dataset from evaluator.py
        total = len(self._data_loader)
        num_warmup = min(5, total - 1)

        start_time = time.perf_counter()
        total_compute_time = 0
        losses = []

        for idx, inputs in enumerate(self._data_loader):
            if idx == num_warmup:
                start_time = time.perf_counter()
                total_compute_time = 0

            start_compute_time = time.perf_counter()

            if torch.cuda.is_available():
                torch.cuda.synchronize()

            total_compute_time += time.perf_counter() - start_compute_time
            iters_after_start = idx + 1 - num_warmup * int(idx >= num_warmup)
            seconds_per_img = total_compute_time / iters_after_start

            if idx >= num_warmup * 2 or seconds_per_img > 5:
                total_seconds_per_img = (time.perf_counter() - start_time) / iters_after_start
                eta = datetime.timedelta(seconds=int(total_seconds_per_img * (total - idx - 1)))
                log_every_n_seconds(
                    logging.INFO,
                    "Loss on Validation  done {}/{}. {:.4f} s / img. ETA={}".format(
                        idx + 1, total, seconds_per_img, str(eta)
                    ),
                    n=5,
                )

            loss_batch = self._get_loss(inputs)
            losses.append(loss_batch)

        mean_loss = np.mean(losses)
        self.trainer.storage.put_scalar('validation_loss', mean_loss)

        comm.synchronize()

        return losses

    def _get_loss(self, data):
        # How loss is calculated on train_loop
        with torch.no_grad():
            metrics_dict = self.trainer.model(data)
            metrics_dict = {
                k: v.detach().cpu().item() if isinstance(v, torch.Tensor) else float(v)
                for k, v in metrics_dict.items()
            }
            total_losses_reduced = sum(loss for loss in metrics_dict.values())
            return total_losses_reduced

    @classmethod
    def create(cls, cfg):
        mapper = DatasetMapper(
            is_train=True,
            augmentations=[
                T.ResizeShortestEdge(short_edge_length=cfg.INPUT.MIN_SIZE_TEST, max_size=cfg.INPUT.MAX_SIZE_TEST,
                                     sample_style=cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING)],
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

        return LossEvalHook(data_loader)
