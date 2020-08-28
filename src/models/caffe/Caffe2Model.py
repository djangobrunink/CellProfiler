import os
from typing import Union

import numpy as np
import torch
import torch.nn as nn
from caffe2.proto import caffe2_pb2
from caffe2.python import core

from .ProtobufDetectionModel import ProtobufDetectionModel
from .structures.instances import Instances


class Caffe2Model(nn.Module):
    """
    A wrapper around the traced model in caffe2's pb format.
    """

    def __init__(self, predict_net, init_net):
        super().__init__()
        self.eval()  # always in eval mode
        self._predict_net = predict_net
        self._init_net = init_net
        self._predictor = None

    @property
    def predict_net(self):
        """
        Returns:
            core.Net: the underlying caffe2 predict net
        """
        return self._predict_net

    @property
    def init_net(self):
        """
        Returns:
            core.Net: the underlying caffe2 init net
        """
        return self._init_net

    @property
    def predictor(self):
        if self._predictor is None:
            self._predictor = ProtobufDetectionModel(self._predict_net, self._init_net)

        return self._predictor

    @staticmethod
    def load_protobuf(dir):
        """
        Args:
            dir (str): a directory used to save Caffe2Model with
                :meth:`save_protobuf`.
                The files "model.pb" and "model_init.pb" are needed.

        Returns:
            Caffe2Model: the caffe2 model loaded from this directory.
        """

        predict_net = caffe2_pb2.NetDef()
        with open(os.path.join(dir, "model.pb"), "rb") as f:
            predict_net.ParseFromString(f.read())

        init_net = caffe2_pb2.NetDef()
        with open(os.path.join(dir, "model_init.pb"), "rb") as f:
            init_net.ParseFromString(f.read())

        return Caffe2Model(predict_net, init_net)

    def __call__(self, image: Union[torch.Tensor, np.ndarray]) -> Instances:
        """
        Args:
            image (Union[torch.Tensor, np.ndarray])): A BGR image with shape (3,H,W).

        Returns:
            Instances: Containing the predicted values.
        """
        assert len(image.shape) == 3 and image.shape[0] == 3, \
            f"Expected image to have shape (3,H,W) but has shape {image.shape}"

        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image.copy())

        _, height, width = image.shape
        inputs = [{
            "image": image,
            "height": height,
            "width": width,
        }]

        results = self.predictor(inputs)
        assert len(results) == 1, "Expected only 1 resutl"
        results = results[0]["instances"].to("cpu")

        return results
