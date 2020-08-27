import inspect
import sys
from abc import ABCMeta, abstractmethod
from typing import Tuple
import pprint
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from fvcore.transforms.transform import NoOpTransform, Transform


class Augmentation(metaclass=ABCMeta):
    """
    Augmentation defines policies/strategies to generate :class:`Transform` from data.
    It is often used for pre-processing of input data. A policy typically contains
    randomness, but it can also choose to deterministically generate a :class:`Transform`.
    A "policy" that generates a :class:`Transform` may, in the most general case,
    need arbitrary information from input data in order to determine what transforms
    to apply. Therefore, each :class:`Augmentation` instance defines the arguments
    needed by its :meth:`get_transform` method with the :attr:`input_args` attribute.
    When called with the positional arguments defined by the :attr:`input_args`,
    the :meth:`get_transform` method executes the policy.
    Examples:
    ::
        # if a policy needs to know both image and semantic segmentation
        assert aug.input_args == ("image", "sem_seg")
        tfm: Transform = aug.get_transform(image, sem_seg)
        new_image = tfm.apply_image(image)
    To implement a custom :class:`Augmentation`, define its :attr:`input_args` and
    implement :meth:`get_transform`.
    Note that :class:`Augmentation` defines the policies to create a :class:`Transform`,
    but not how to apply the actual transform to those data.
    """

    input_args: Tuple[str] = ("image",)
    """
    Attribute of class instances that defines the argument(s) needed by
    :meth:`get_transform`. Default to only "image", because most policies only
    require knowing the image in order to determine the transform.
    Users can freely define arbitrary new args and their types in custom
    :class:`Augmentation`. In detectron2 we use the following convention:
    * image: (H,W) or (H,W,C) ndarray of type uint8 in range [0, 255], or
      floating point in range [0, 1] or [0, 255].
    * boxes: (N,4) ndarray of float32. It represents the instance bounding boxes
      of N instances. Each is in XYXY format in unit of absolute coordinates.
    * sem_seg: (H,W) ndarray of type uint8. Each element is an integer label of pixel.
    We do not specify convention for other types and do not include builtin
    :class:`Augmentation` that uses other types in detectron2.
    """

    def _init(self, params=None):
        if params:
            for k, v in params.items():
                if k != "self" and not k.startswith("_"):
                    setattr(self, k, v)

    # NOTE: in the future, can allow it to return list[Augmentation],
    # to delegate augmentation to others
    @abstractmethod
    def get_transform(self, *args):
        """
        Execute the policy to use input data to create transform(s).
        Args:
            arguments must follow what's defined in :attr:`input_args`.
        Returns:
            Return a :class:`Transform` instance, which is the transform to apply to inputs.
        """
        pass

    def _rand_range(self, low=1.0, high=None, size=None):
        """
        Uniform float random number between low and high.
        """
        if high is None:
            low, high = 0, low
        if size is None:
            size = []
        return np.random.uniform(low, high, size)

    def __repr__(self):
        """
        Produce something like:
        "MyAugmentation(field1={self.field1}, field2={self.field2})"
        """
        try:
            sig = inspect.signature(self.__init__)
            classname = type(self).__name__
            argstr = []
            for name, param in sig.parameters.items():
                assert (
                    param.kind != param.VAR_POSITIONAL and param.kind != param.VAR_KEYWORD
                ), "The default __repr__ doesn't support *args or **kwargs"
                assert hasattr(self, name), (
                    "Attribute {} not found! "
                    "Default __repr__ only works if attributes match the constructor.".format(name)
                )
                attr = getattr(self, name)
                default = param.default
                if default is attr:
                    continue
                argstr.append("{}={}".format(name, pprint.pformat(attr)))
            return "{}({})".format(classname, ", ".join(argstr))
        except AssertionError:
            return super().__repr__()

    __str__ =  __repr__


class ResizeShortestEdge(Augmentation):
    """
    Scale the shorter edge to the given size, with a limit of `max_size` on the longer edge.
    If `max_size` is reached, then downscale so that the longer edge does not exceed max_size.
    """

    def __init__(
        self, short_edge_length, max_size=sys.maxsize, sample_style="range", interp=Image.BILINEAR
    ):
        """
        Args:
            short_edge_length (list[int]): If ``sample_style=="range"``,
                a [min, max] interval from which to sample the shortest edge length.
                If ``sample_style=="choice"``, a list of shortest edge lengths to sample from.
            max_size (int): maximum allowed longest edge length.
            sample_style (str): either "range" or "choice".
        """
        super().__init__()
        assert sample_style in ["range", "choice"], sample_style

        self.is_range = sample_style == "range"
        if isinstance(short_edge_length, int):
            short_edge_length = (short_edge_length, short_edge_length)
        if self.is_range:
            assert len(short_edge_length) == 2, (
                "short_edge_length must be two values using 'range' sample style."
                f" Got {short_edge_length}!"
            )
        self._init(locals())

    def get_transform(self, img):
        h, w = img.shape[:2]
        if self.is_range:
            size = np.random.randint(self.short_edge_length[0], self.short_edge_length[1] + 1)
        else:
            size = np.random.choice(self.short_edge_length)
        if size == 0:
            return NoOpTransform()

        scale = size * 1.0 / min(h, w)
        if h < w:
            newh, neww = size, scale * w
        else:
            newh, neww = scale * h, size
        if max(newh, neww) > self.max_size:
            scale = self.max_size * 1.0 / max(newh, neww)
            newh = newh * scale
            neww = neww * scale
        neww = int(neww + 0.5)
        newh = int(newh + 0.5)
        return ResizeTransform(h, w, newh, neww, self.interp)


class ResizeTransform(Transform):
    """
    Resize the image to a target size.
    """

    def __init__(self, h, w, new_h, new_w, interp=None):
        """
        Args:
            h, w (int): original image size
            new_h, new_w (int): new image size
            interp: PIL interpolation methods, defaults to bilinear.
        """
        # TODO decide on PIL vs opencv
        super().__init__()
        if interp is None:
            interp = Image.BILINEAR
        self._set_attributes(locals())

    def apply_image(self, img, interp=None):
        assert img.shape[:2] == (self.h, self.w)
        assert len(img.shape) <= 4

        if img.dtype == np.uint8:
            pil_image = Image.fromarray(img)
            interp_method = interp if interp is not None else self.interp
            pil_image = pil_image.resize((self.new_w, self.new_h), interp_method)
            ret = np.asarray(pil_image)
            ret = ret.copy()
        else:
            # PIL only supports uint8
            img = torch.from_numpy(img)
            shape = list(img.shape)
            shape_4d = shape[:2] + [1] * (4 - len(shape)) + shape[2:]
            img = img.view(shape_4d).permute(2, 3, 0, 1)  # hw(c) -> nchw
            _PIL_RESIZE_TO_INTERPOLATE_MODE = {Image.BILINEAR: "bilinear", Image.BICUBIC: "bicubic"}
            mode = _PIL_RESIZE_TO_INTERPOLATE_MODE[self.interp]
            img = F.interpolate(img, (self.new_h, self.new_w), mode=mode, align_corners=False)
            shape[:2] = (self.new_h, self.new_w)
            ret = img.permute(2, 3, 0, 1).view(shape).numpy()  # nchw -> hw(c)

        return ret

    def apply_coords(self, coords):
        coords[:, 0] = coords[:, 0] * (self.new_w * 1.0 / self.w)
        coords[:, 1] = coords[:, 1] * (self.new_h * 1.0 / self.h)
        return coords

    def apply_segmentation(self, segmentation):
        segmentation = self.apply_image(segmentation, interp=Image.NEAREST)
        return segmentation

    def inverse(self):
        return ResizeTransform(self.new_h, self.new_w, self.h, self.w, self.interp)