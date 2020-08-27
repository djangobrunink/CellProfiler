import copy
from contextlib import contextmanager
from functools import wraps
from typing import Any, Tuple, Dict, Callable

import numpy as np
import torch
from caffe2.proto import caffe2_pb2
from caffe2.python import core, workspace
from torch.nn import functional as F

from .structures.boxes import Boxes
from .structures.image_list import ImageList
from .structures.instances import Instances
from .structures.rotated_boxes import RotatedBoxes


class ScopedWS(object):
    def __init__(self, ws_name, is_reset, is_cleanup=False):
        self.ws_name = ws_name
        self.is_reset = is_reset
        self.is_cleanup = is_cleanup
        self.org_ws = ""

    def __enter__(self):
        self.org_ws = workspace.CurrentWorkspace()
        if self.ws_name is not None:
            workspace.SwitchWorkspace(self.ws_name, True)
        if self.is_reset:
            workspace.ResetWorkspace()

        return workspace

    def __exit__(self, *args):
        if self.is_cleanup:
            workspace.ResetWorkspace()
        if self.ws_name is not None:
            workspace.SwitchWorkspace(self.org_ws)


def get_pb_arg_vals(pb, arg_name, default_val):
    arg = get_pb_arg(pb, arg_name)
    return arg.s if arg is not None else default_val


def get_pb_arg_vali(pb, arg_name, default_val):
    arg = get_pb_arg(pb, arg_name)
    return arg.i if arg is not None else default_val


def get_pb_arg(pb, arg_name):
    for x in pb.arg:
        if x.name == arg_name:
            return x
    return None


def detector_postprocess(results, output_height, output_width, mask_threshold=0.5):
    """
    Resize the output instances.
    The input images are often resized when entering an object detector.
    As a result, we often need the outputs of the detector in a different
    resolution from its inputs.
    This function will resize the raw outputs of an R-CNN detector
    to produce outputs according to the desired output resolution.
    Args:
        results (Instances): the raw outputs from the detector.
            `results.image_size` contains the input image resolution the detector sees.
            This object might be modified in-place.
        output_height, output_width: the desired output resolution.
    Returns:
        Instances: the resized output from the model, based on the output resolution
    """

    # Converts integer tensors to float temporaries
    #   to ensure true division is performed when
    #   computing scale_x and scale_y.
    if isinstance(output_width, torch.Tensor):
        output_width_tmp = output_width.float()
    else:
        output_width_tmp = output_width

    if isinstance(output_height, torch.Tensor):
        output_height_tmp = output_height.float()
    else:
        output_height_tmp = output_height

    scale_x, scale_y = (
        output_width_tmp / results.image_size[1],
        output_height_tmp / results.image_size[0],
    )
    results = Instances((output_height, output_width), **results.get_fields())

    if results.has("pred_boxes"):
        output_boxes = results.pred_boxes
    elif results.has("proposal_boxes"):
        output_boxes = results.proposal_boxes

    output_boxes.scale(scale_x, scale_y)
    output_boxes.clip(results.image_size)

    results = results[output_boxes.nonempty()]

    if results.has("pred_masks"):
        results.pred_masks = retry_if_cuda_oom(paste_masks_in_image)(
            results.pred_masks[:, 0, :, :],  # N, 1, M, M
            results.pred_boxes,
            results.image_size,
            threshold=mask_threshold,
        )

    if results.has("pred_keypoints"):
        results.pred_keypoints[:, :, 0] *= scale_x
        results.pred_keypoints[:, :, 1] *= scale_y

    return results


def retry_if_cuda_oom(func):
    """
    Makes a function retry itself after encountering
    pytorch's CUDA OOM error.
    It will first retry after calling `torch.cuda.empty_cache()`.
    If that still fails, it will then retry by trying to convert inputs to CPUs.
    In this case, it expects the function to dispatch to CPU implementation.
    The return values may become CPU tensors as well and it's user's
    responsibility to convert it back to CUDA tensor if needed.
    Args:
        func: a stateless callable that takes tensor-like objects as arguments
    Returns:
        a callable which retries `func` if OOM is encountered.
    Examples:
    ::
        output = retry_if_cuda_oom(some_torch_function)(input1, input2)
        # output may be on CPU even if inputs are on GPU
    Note:
        1. When converting inputs to CPU, it will only look at each argument and check
           if it has `.device` and `.to` for conversion. Nested structures of tensors
           are not supported.
        2. Since the function might be called more than once, it has to be
           stateless.
    """

    def maybe_to_cpu(x):
        try:
            like_gpu_tensor = x.device.type == "cuda" and hasattr(x, "to")
        except AttributeError:
            like_gpu_tensor = False
        if like_gpu_tensor:
            return x.to(device="cpu")
        else:
            return x

    @wraps(func)
    def wrapped(*args, **kwargs):
        with _ignore_torch_cuda_oom():
            return func(*args, **kwargs)

        # Clear cache and retry
        torch.cuda.empty_cache()
        with _ignore_torch_cuda_oom():
            return func(*args, **kwargs)

        # Try on CPU. This slows down the code significantly, therefore print a notice.
        logger = logging.getLogger(__name__)
        logger.info("Attempting to copy inputs of {} to CPU due to CUDA OOM".format(str(func)))
        new_args = (maybe_to_cpu(x) for x in args)
        new_kwargs = {k: maybe_to_cpu(v) for k, v in kwargs.items()}
        return func(*new_args, **new_kwargs)

    return wrapped


@contextmanager
def _ignore_torch_cuda_oom():
    """
    A context which ignores CUDA OOM exception from pytorch.
    """
    try:
        yield
    except RuntimeError as e:
        # NOTE: the string may change?
        if "CUDA out of memory. " in str(e):
            pass
        else:
            raise


BYTES_PER_FLOAT = 4
# TODO: This memory limit may be too much or too little. It would be better to
# determine it based on available resources.
GPU_MEM_LIMIT = 1024 ** 3  # 1 GB memory limit


def paste_masks_in_image(masks, boxes, image_shape, threshold=0.5):
    """
    Paste a set of masks that are of a fixed resolution (e.g., 28 x 28) into an image.
    The location, height, and width for pasting each mask is determined by their
    corresponding bounding boxes in boxes.
    Note:
        This is a complicated but more accurate implementation. In actual deployment, it is
        often enough to use a faster but less accurate implementation.
        See :func:`paste_mask_in_image_old` in this file for an alternative implementation.
    Args:
        masks (tensor): Tensor of shape (Bimg, Hmask, Wmask), where Bimg is the number of
            detected object instances in the image and Hmask, Wmask are the mask width and mask
            height of the predicted mask (e.g., Hmask = Wmask = 28). Values are in [0, 1].
        boxes (Boxes or Tensor): A Boxes of length Bimg or Tensor of shape (Bimg, 4).
            boxes[i] and masks[i] correspond to the same object instance.
        image_shape (tuple): height, width
        threshold (float): A threshold in [0, 1] for converting the (soft) masks to
            binary masks.
    Returns:
        img_masks (Tensor): A tensor of shape (Bimg, Himage, Wimage), where Bimg is the
        number of detected object instances and Himage, Wimage are the image width
        and height. img_masks[i] is a binary mask for object instance i.
    """

    assert masks.shape[-1] == masks.shape[-2], "Only square mask predictions are supported"
    N = len(masks)
    if N == 0:
        return masks.new_empty((0,) + image_shape, dtype=torch.uint8)
    if not isinstance(boxes, torch.Tensor):
        boxes = boxes.tensor
    device = boxes.device
    assert len(boxes) == N, boxes.shape

    img_h, img_w = image_shape

    # The actual implementation split the input into chunks,
    # and paste them chunk by chunk.
    if device.type == "cpu":
        # CPU is most efficient when they are pasted one by one with skip_empty=True
        # so that it performs minimal number of operations.
        num_chunks = N
    else:
        # GPU benefits from parallelism for larger chunks, but may have memory issue
        # int(img_h) because shape may be tensors in tracing
        num_chunks = int(np.ceil(N * int(img_h) * int(img_w) * BYTES_PER_FLOAT / GPU_MEM_LIMIT))
        assert (
                num_chunks <= N
        ), "Default GPU_MEM_LIMIT in mask_ops.py is too small; try increasing it"
    chunks = torch.chunk(torch.arange(N, device=device), num_chunks)

    img_masks = torch.zeros(
        N, img_h, img_w, device=device, dtype=torch.bool if threshold >= 0 else torch.uint8
    )
    for inds in chunks:
        masks_chunk, spatial_inds = _do_paste_mask(
            masks[inds, None, :, :], boxes[inds], img_h, img_w, skip_empty=device.type == "cpu"
        )

        if threshold >= 0:
            masks_chunk = (masks_chunk >= threshold).to(dtype=torch.bool)
        else:
            # for visualization and debugging
            masks_chunk = (masks_chunk * 255).to(dtype=torch.uint8)

        img_masks[(inds,) + spatial_inds] = masks_chunk
    return img_masks


def _do_paste_mask(masks, boxes, img_h, img_w, skip_empty=True):
    """
    Args:
        masks: N, 1, H, W
        boxes: N, 4
        img_h, img_w (int):
        skip_empty (bool): only paste masks within the region that
            tightly bound all boxes, and returns the results this region only.
            An important optimization for CPU.
    Returns:
        if skip_empty == False, a mask of shape (N, img_h, img_w)
        if skip_empty == True, a mask of shape (N, h', w'), and the slice
            object for the corresponding region.
    """
    # On GPU, paste all masks together (up to chunk size)
    # by using the entire image to sample the masks
    # Compared to pasting them one by one,
    # this has more operations but is faster on COCO-scale dataset.
    device = masks.device
    if skip_empty:
        x0_int, y0_int = torch.clamp(boxes.min(dim=0).values.floor()[:2] - 1, min=0).to(
            dtype=torch.int32
        )
        x1_int = torch.clamp(boxes[:, 2].max().ceil() + 1, max=img_w).to(dtype=torch.int32)
        y1_int = torch.clamp(boxes[:, 3].max().ceil() + 1, max=img_h).to(dtype=torch.int32)
    else:
        x0_int, y0_int = 0, 0
        x1_int, y1_int = img_w, img_h
    x0, y0, x1, y1 = torch.split(boxes, 1, dim=1)  # each is Nx1

    N = masks.shape[0]

    img_y = torch.arange(y0_int, y1_int, device=device, dtype=torch.float32) + 0.5
    img_x = torch.arange(x0_int, x1_int, device=device, dtype=torch.float32) + 0.5
    img_y = (img_y - y0) / (y1 - y0) * 2 - 1
    img_x = (img_x - x0) / (x1 - x0) * 2 - 1
    # img_x, img_y have shapes (N, w), (N, h)

    gx = img_x[:, None, :].expand(N, img_y.size(1), img_x.size(1))
    gy = img_y[:, :, None].expand(N, img_y.size(1), img_x.size(1))
    grid = torch.stack([gx, gy], dim=3)

    img_masks = F.grid_sample(masks.to(dtype=torch.float32), grid, align_corners=False)

    if skip_empty:
        return img_masks[:, 0], (slice(y0_int, y1_int), slice(x0_int, x1_int))
    else:
        return img_masks[:, 0], ()


def infer_device_type(
        predict_net: caffe2_pb2.NetDef,
        known_status: Dict[Tuple[str, int], Any],
        device_name_style: str = "caffe2",
) -> Dict[Tuple[str, int], str]:
    """ Return the device type ("cpu" or "gpu"/"cuda") of each (versioned) blob """

    assert device_name_style in ["caffe2", "pytorch"]
    _CPU_STR = "cpu"
    _GPU_STR = "gpu" if device_name_style == "caffe2" else "cuda"

    def _copy_cpu_to_gpu_updater(op, input_types, output_types):
        if input_types[0] == _GPU_STR or output_types[0] == _CPU_STR:
            _updater_raise(op, input_types, output_types)
        return ([_CPU_STR], [_GPU_STR])

    def _copy_gpu_to_cpu_updater(op, input_types, output_types):
        if input_types[0] == _CPU_STR or output_types[0] == _GPU_STR:
            _updater_raise(op, input_types, output_types)
        return ([_GPU_STR], [_CPU_STR])

    def _other_ops_updater(op, input_types, output_types):
        non_none_types = [x for x in input_types + output_types if x is not None]
        if len(non_none_types) > 0:
            the_type = non_none_types[0]
            if not all(x == the_type for x in non_none_types):
                _updater_raise(op, input_types, output_types)
        else:
            the_type = None
        return ([the_type for _ in op.input], [the_type for _ in op.output])

    def _device_updater(op, *args, **kwargs):
        return {
            "CopyCPUToGPU": _copy_cpu_to_gpu_updater,
            "CopyGPUToCPU": _copy_gpu_to_cpu_updater,
        }.get(op.type, _other_ops_updater)(op, *args, **kwargs)

    return _generic_status_identifier(predict_net, _device_updater, known_status)


def _updater_raise(op, input_types, output_types):
    raise RuntimeError(
        "Failed to apply updater for op {} given input_types {} and"
        " output_types {}".format(op, input_types, output_types)
    )


def _generic_status_identifier(
        predict_net: caffe2_pb2.NetDef,
        status_updater: Callable,
        known_status: Dict[Tuple[str, int], Any],
) -> Dict[Tuple[str, int], Any]:
    """
    Statically infer the status of each blob, the status can be such as device type
        (CPU/GPU), layout (NCHW/NHWC), data type (float32/int8), etc. "Blob" here
        is versioned blob (Tuple[str, int]) in the format compatible with ssa.
    Inputs:
        predict_net: the caffe2 network
        status_updater: a callable, given an op and the status of its input/output,
            it returns the updated status of input/output. `None` is used for
            representing unknown status.
        known_status: a dict containing known status, used as initialization.
    Outputs:
        A dict mapping from versioned blob to its status
    """
    ssa, versions = core.get_ssa(predict_net)
    versioned_ext_input = [(b, 0) for b in predict_net.external_input]
    versioned_ext_output = [(b, versions[b]) for b in predict_net.external_output]
    all_versioned_blobs = set().union(*[set(x[0] + x[1]) for x in ssa])

    allowed_vbs = all_versioned_blobs.union(versioned_ext_input).union(versioned_ext_output)
    assert all(k in allowed_vbs for k in known_status)
    assert all(v is not None for v in known_status.values())
    _known_status = copy.deepcopy(known_status)

    def _check_and_update(key, value):
        assert value is not None
        if key in _known_status:
            if not _known_status[key] == value:
                raise RuntimeError(
                    "Confilict status for {}, existing status {}, new status {}".format(
                        key, _known_status[key], value
                    )
                )
        _known_status[key] = value

    def _update_i(op, ssa_i):
        versioned_inputs = ssa_i[0]
        versioned_outputs = ssa_i[1]

        inputs_status = [_known_status.get(b, None) for b in versioned_inputs]
        outputs_status = [_known_status.get(b, None) for b in versioned_outputs]

        new_inputs_status, new_outputs_status = status_updater(op, inputs_status, outputs_status)

        for versioned_blob, status in zip(
                versioned_inputs + versioned_outputs, new_inputs_status + new_outputs_status
        ):
            if status is not None:
                _check_and_update(versioned_blob, status)

    for op, ssa_i in zip(predict_net.op, ssa):
        _update_i(op, ssa_i)
    for op, ssa_i in zip(reversed(predict_net.op), reversed(ssa)):
        _update_i(op, ssa_i)

    # NOTE: This strictly checks all the blob from predict_net must be assgined
    # a known status. However sometimes it's impossible (eg. having deadend op),
    # we may relax this constraint if
    for k in all_versioned_blobs:
        if k not in _known_status:
            raise NotImplementedError(
                "Can not infer the status for {}. Currently only support the case where"
                " a single forward and backward pass can identify status for all blobs.".format(k)
            )

    return _known_status


def assemble_rcnn_outputs_by_name(image_sizes, tensor_outputs, force_mask_on=False):
    """
    A function to assemble caffe2 model's outputs (i.e. Dict[str, Tensor])
    to detectron2's format (i.e. list of Instances instance).
    This only works when the model follows the Caffe2 detectron's naming convention.
    Args:
        image_sizes (List[List[int, int]]): [H, W] of every image.
        tensor_outputs (Dict[str, Tensor]): external_output to its tensor.
        force_mask_on (Bool): if true, the it make sure there'll be pred_masks even
            if the mask is not found from tensor_outputs (usually due to model crash)
    """

    results = [Instances(image_size) for image_size in image_sizes]

    batch_splits = tensor_outputs.get("batch_splits", None)
    if batch_splits:
        raise NotImplementedError()
    assert len(image_sizes) == 1
    result = results[0]

    bbox_nms = tensor_outputs["bbox_nms"]
    score_nms = tensor_outputs["score_nms"]
    class_nms = tensor_outputs["class_nms"]
    # Detection will always success because Conv support 0-batch
    assert bbox_nms is not None
    assert score_nms is not None
    assert class_nms is not None
    if bbox_nms.shape[1] == 5:
        result.pred_boxes = RotatedBoxes(bbox_nms)
    else:
        result.pred_boxes = Boxes(bbox_nms)
    result.scores = score_nms
    result.pred_classes = class_nms.to(torch.int64)

    mask_fcn_probs = tensor_outputs.get("mask_fcn_probs", None)
    if mask_fcn_probs is not None:
        # finish the mask pred
        mask_probs_pred = mask_fcn_probs
        num_masks = mask_probs_pred.shape[0]
        class_pred = result.pred_classes
        indices = torch.arange(num_masks, device=class_pred.device)
        mask_probs_pred = mask_probs_pred[indices, class_pred][:, None]
        result.pred_masks = mask_probs_pred
    elif force_mask_on:
        # NOTE: there's no way to know the height/width of mask here, it won't be
        # used anyway when batch size is 0, so just set them to 0.
        result.pred_masks = torch.zeros([0, 1, 0, 0], dtype=torch.uint8)

    keypoints_out = tensor_outputs.get("keypoints_out", None)
    kps_score = tensor_outputs.get("kps_score", None)
    if keypoints_out is not None:
        # keypoints_out: [N, 4, #kypoints], where 4 is in order of (x, y, score, prob)
        keypoints_tensor = keypoints_out
        # NOTE: it's possible that prob is not calculated if "should_output_softmax"
        # is set to False in HeatmapMaxKeypoint, so just using raw score, seems
        # it doesn't affect mAP. TODO: check more carefully.
        keypoint_xyp = keypoints_tensor.transpose(1, 2)[:, :, [0, 1, 2]]
        result.pred_keypoints = keypoint_xyp

    return results


def convert_batched_inputs_to_c2_format(batched_inputs, size_divisibility, device):
    """
    See get_caffe2_inputs() below.
    """
    assert all(isinstance(x, dict) for x in batched_inputs)
    assert all(x["image"].dim() == 3 for x in batched_inputs)

    images = [x["image"] for x in batched_inputs]
    images = ImageList.from_tensors(images, size_divisibility)

    im_info = []
    for input_per_image, image_size in zip(batched_inputs, images.image_sizes):
        target_height = input_per_image.get("height", image_size[0])
        target_width = input_per_image.get("width", image_size[1])  # noqa
        # NOTE: The scale inside im_info is kept as convention and for providing
        # post-processing information if further processing is needed. For
        # current Caffe2 model definitions that don't include post-processing inside
        # the model, this number is not used.
        # NOTE: There can be a slight difference between width and height
        # scales, using a single number can results in numerical difference
        # compared with D2's post-processing.
        scale = target_height / image_size[0]
        im_info.append([image_size[0], image_size[1], scale])
    im_info = torch.Tensor(im_info)

    return images.tensor.to(device), im_info.to(device)
