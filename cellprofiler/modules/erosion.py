"""
Erosion
=======

**Erosion** shrinks bright shapes in an image. See `this tutorial`_ for more information.

|

============ ============ ===============
Supports 2D? Supports 3D? Respects masks?
============ ============ ===============
YES          YES          NO
============ ============ ===============

.. _this tutorial: https://scikit-image.org/docs/stable/auto_examples/applications/plot_morphology.html#erosion

"""

import numpy
import skimage.morphology
from cellprofiler_core.module import ImageProcessing
from cellprofiler_core.setting import StructuringElement

from cellprofiler.modules._help import HELP_FOR_STREL


class Erosion(ImageProcessing):
    category = "Advanced"

    module_name = "Erosion"

    variable_revision_number = 1

    def create_settings(self):
        super(Erosion, self).create_settings()

        self.structuring_element = StructuringElement(
            allow_planewise=True, doc=HELP_FOR_STREL
        )

    def settings(self):
        __settings__ = super(Erosion, self).settings()

        return __settings__ + [self.structuring_element]

    def visible_settings(self):
        __settings__ = super(Erosion, self).settings()

        return __settings__ + [self.structuring_element]

    def run(self, workspace):

        x = workspace.image_set.get_image(self.x_name.value)

        is_strel_2d = self.structuring_element.value.ndim == 2

        is_img_2d = x.pixel_data.ndim == 2

        if is_strel_2d and not is_img_2d:

            self.function = planewise_morphology_erosion

        elif not is_strel_2d and is_img_2d:

            raise NotImplementedError(
                "A 3D structuring element cannot be applied to a 2D image."
            )

        else:

            self.function = skimage.morphology.erosion

        super(Erosion, self).run(workspace)


def planewise_morphology_erosion(x_data, structuring_element):

    y_data = numpy.zeros_like(x_data)

    for index, plane in enumerate(x_data):

        y_data[index] = skimage.morphology.erosion(plane, structuring_element)

    return y_data
