import numpy as np
import torch
from caffe2.python import core

from .ProtobufModel import ProtobufModel
from .utils import get_pb_arg_vali, get_pb_arg_vals, infer_device_type, assemble_rcnn_outputs_by_name, \
    detector_postprocess, convert_batched_inputs_to_c2_format


class ProtobufDetectionModel(torch.nn.Module):
    """
    A class works just like a pytorch meta arch in terms of inference, but running
    caffe2 model under the hood.
    """

    def __init__(self, predict_net, init_net, *, convert_outputs=None):
        """
        Args:
            predict_net, init_net (core.Net): caffe2 nets
            convert_outptus (callable): a function that converts caffe2
                outputs to the same format of the original pytorch model.
                By default, use the one defined in the caffe2 meta_arch.
        """
        super().__init__()
        self.protobuf_model = ProtobufModel(predict_net, init_net)
        self.size_divisibility = get_pb_arg_vali(predict_net, "size_divisibility", 0)
        self.device = get_pb_arg_vals(predict_net, "device", b"cpu").decode("ascii")

    def _convert_outputs(self, batched_inputs, c2_inputs, c2_results):
        image_sizes = [[int(im[0]), int(im[1])] for im in c2_inputs["im_info"]]
        results = assemble_rcnn_outputs_by_name(image_sizes, c2_results)

        return self._postprocess(results, batched_inputs, image_sizes)

    def _postprocess(self, instances, batched_inputs, image_sizes):
        """
        Rescale the output instances to the target size.
        """
        # note: private function; subject to changes
        processed_results = []
        for results_per_image, input_per_image, image_size in zip(
                instances, batched_inputs, image_sizes
        ):
            height = input_per_image.get("height", image_size[0])
            width = input_per_image.get("width", image_size[1])
            r = detector_postprocess(results_per_image, height, width)
            processed_results.append({"instances": r})
        return processed_results

    def _infer_output_devices(self, inputs_dict):
        def _get_device_type(torch_tensor):
            assert torch_tensor.device.type in ["cpu", "cuda"]
            assert torch_tensor.device.index == 0
            return torch_tensor.device.type

        predict_net = self.protobuf_model.net.Proto()
        input_device_types = {
            (name, 0): _get_device_type(tensor) for name, tensor in inputs_dict.items()
        }
        device_type_map = infer_device_type(
            predict_net, known_status=input_device_types, device_name_style="pytorch"
        )
        ssa, versions = core.get_ssa(predict_net)
        versioned_outputs = [(name, versions[name]) for name in predict_net.external_output]
        output_devices = [device_type_map[outp] for outp in versioned_outputs]
        return output_devices

    def _convert_inputs(self, batched_inputs):
        # currently all models convert inputs in the same way
        data, im_info = convert_batched_inputs_to_c2_format(
            batched_inputs, self.size_divisibility, self.device
        )
        return {"data": data, "im_info": im_info}

    def forward(self, batched_inputs):
        c2_inputs = self._convert_inputs(batched_inputs)
        c2_results = self.protobuf_model(c2_inputs)

        if any(t.device.type != "cpu" for _, t in c2_inputs.items()):
            output_devices = self._infer_output_devices(c2_inputs)
        else:
            output_devices = ["cpu" for _ in self.protobuf_model.net.Proto().external_output]

        def _cast_caffe2_blob_to_torch_tensor(blob, device):
            return torch.Tensor(blob).to(device) if isinstance(blob, np.ndarray) else None

        c2_results = {
            name: _cast_caffe2_blob_to_torch_tensor(c2_results[name], device)
            for name, device in zip(self.protobuf_model.net.Proto().external_output, output_devices)
        }

        return self._convert_outputs(batched_inputs, c2_inputs, c2_results)
