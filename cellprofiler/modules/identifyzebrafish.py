# coding=utf-8

import math

import centrosome.cpmorphology
import centrosome.outline
import centrosome.propagate
import centrosome.threshold
import numpy
import scipy.ndimage
import scipy.sparse
import skimage.morphology
import torch
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.datasets import coco
import matplotlib.pyplot as plt
import imageio
import pprint
from src.models.caffe.Caffe2Model import Caffe2Model
import src.utils as utils

import matplotlib.patches as patches    
import cellprofiler.gui.help
import cellprofiler.gui.help.content
import cellprofiler_core.object
import cellprofiler_core.setting
from cellprofiler.modules import _help, threshold
import cellprofiler_core.module.image_segmentation

__doc__ = ""


# Settings text which is referenced in various places in the help
EXCLUDE_OVERLAP_TEXT = "Exclude regions with overlapping zebrafish?"
MANAGE_OVERLAP_TEXT = "How to deal with overlapping regions?"
GENERATE_IMAGE_FOR = "For which types of zebrafish do you want to generate an image?"

# Settings choices
OVERLAP_ASSIGN = "Assign region to one class (in order: healthy > deformed > dead)"
OVERLAP_EXCLUDE_REGION = "Exclude the region that has overlap."
OVERLAP_EXCLUDE_OBJECTS = "Exclude all objects that contain overlapping regions."
OVERLAP_ALL = [OVERLAP_ASSIGN, OVERLAP_EXCLUDE_REGION, OVERLAP_EXCLUDE_OBJECTS]

GENERATE_HEALTHY = "ZebrafishHealthy"
GENERATE_DEFORMED = "ZebrafishDeformed"
GENERATE_DEAD = "ZebrafishDead"
GENERATE_ALL = [GENERATE_HEALTHY, GENERATE_DEFORMED, GENERATE_DEAD]

IMAGES_MAX_TOTAL_COUNT = 3

start_up = True

class IdentifyZebrafish(
    cellprofiler_core.module.ImageProcessing
):
    variable_revision_number = 1

    category = "Image Processing"

    module_name = "IdentifyZebrafish"

    def __init__(self):
        super(IdentifyZebrafish, self).__init__()

    def volumetric(self):
        return False

    def create_settings(self):
        super(IdentifyZebrafish, self).create_settings()

        self.x_name.text = "Select the input image"
        self.x_name.doc = "Select the image that you want to use to identify objects."

        self.y_name.text = "Name the zebrafish objects to be identified"
        self.y_name.doc = "Enter the name that you want to call the objects identified by this module."

        self.manage_overlap = cellprofiler_core.setting.Choice(
            MANAGE_OVERLAP_TEXT, 
            OVERLAP_ALL,
            doc="",
        )

        self.separator = cellprofiler_core.setting.Divider(
            line=True
        )

        self.total_images = []

        self.total_images_count = cellprofiler_core.setting.HiddenCount(
            self.total_images, 
            "Total image count"
        )

        self.add_button = cellprofiler_core.setting.DoSomething(
            "",
            "Add a new image",
            self.add_image
        )

        self.add_image(can_remove=False, name=GENERATE_HEALTHY)
        self.add_image(can_remove=True, name=GENERATE_DEFORMED)
        self.add_image(can_remove=True, name=GENERATE_DEAD)

        

    def add_image(self, can_remove=True, name=GENERATE_DEFORMED):
        group = cellprofiler_core.setting.SettingsGroup()

        group.append("separator", cellprofiler_core.setting.Divider(line=True))

        if can_remove:
            group.append(
                "generate_image_for",
                cellprofiler_core.setting.Choice(
                    text=GENERATE_IMAGE_FOR,
                    choices=[GENERATE_DEFORMED, GENERATE_DEAD],
                    value=name,
                    doc="",
                )
            )

        else:
            group.append(
                "generate_image_for",
                cellprofiler_core.setting.Choice(
                    text=GENERATE_IMAGE_FOR,
                    choices=[GENERATE_HEALTHY],
                    value=GENERATE_HEALTHY,
                    doc="",
                )
            )

        group.append(
            "output_image_name",
            cellprofiler_core.setting.ImageNameProvider(
                "Name the output image",
                name,
                doc="",
            )
        )

        if can_remove:
            group.append(
                "remove_button",
                cellprofiler_core.setting.RemoveSettingButton(
                    "",
                    "Remove",
                    self.total_images,
                    group
                )
            )
        
        self.total_images.append(group)


    def settings(self):
        settings = super(IdentifyZebrafish, self).settings()

        settings += [
            self.x_name,
            self.manage_overlap,
            self.total_images_count,
        ]   

        for additional in self.total_images:
            settings += [
                additional.generate_image_for,
                additional.output_image_name,
            ]

        return settings
    

    def visible_settings(self):
        visible_settings = []

        visible_settings += [
            self.x_name,
            self.manage_overlap,
        ]

        for total in self.total_images:
            visible_settings += total.visible_settings()
        
        if self.total_images_count.value < IMAGES_MAX_TOTAL_COUNT:
            visible_settings += [self.add_button]

        return visible_settings

    def get_model(self):
        WEIGHT_FOLDER_PATH = r'weights/caffe2/unlabeled_fish_cpu'
        model = Caffe2Model.load_protobuf(WEIGHT_FOLDER_PATH) #TODO change pretrained=True to our own set of weights.
        return model

    def run(self, workspace):
        #TODO raise Error when Deformed or Dead is selected twice. 
        THRESHOLD = 0.5
        image_name = self.x_name.value
        image = workspace.image_set.get_image(image_name).get_image()
        image = numpy.array((image * 255), dtype=numpy.uint8)
        print(image.shape)
        assert image.ndim in (2, 3), "Image should be grayscale or RGB"
        if image.ndim == 2:
            image = numpy.stack((image,)*3, axis=0)
        print(numpy.min(image))
        print(numpy.max(image))
        print(image.dtype)
        print(image.shape)
        workspace.display_data.statistics = []

        # Convert image to torch image.
        #torch_image = torch.from_numpy(image)
        #torch_image = torch_image.unsqueeze(0).unsqueeze(0)
        
        #pass image through DL model
        model = self.get_model()
        output = model(image)

        # Initialize output images and convert to numpy
        healthy_mask = numpy.zeros(image.shape)
        deformed_mask = numpy.zeros(image.shape)
        dead_mask = numpy.zeros(image.shape)
        orig_image = image

        print(output)


        for i, (numpy_mask, label) in enumerate(zip(output.pred_masks, output.pred_classes)):
            numpy_mask = numpy.where(numpy_mask > THRESHOLD, i + 1, 0)
            if label == 0:
                # Add it to 'Healthy'
                healthy_mask += numpy_mask
                
            if label == 1:
                # Add it to 'Deformed'
                deformed_mask += numpy_mask
    
            if label == 2:
                # Add it to 'Dead'
                dead_mask += numpy_mask

        # Discard objects that overlap if required.
        if self.manage_overlap == OVERLAP_ASSIGN:
            # If object is present in healthy_mask it is removed from deformed and dead. If an object is not in healthy_mask but is in deformed_mask, it is removed from dead_mask.
            deformed_mask = numpy.where((healthy_mask != 0) & (deformed_mask != 0), 0, deformed_mask)
            dead_mask = numpy.where(((healthy_mask != 0) & (dead_mask != 0)) | ((deformed_mask != 0) & (dead_mask != 0)), 0, dead_mask)

        if self.manage_overlap == OVERLAP_EXCLUDE_REGION or self.manage_overlap == OVERLAP_EXCLUDE_OBJECTS:
            # If region is present in any two or more masks it is removed from all.
            healthy_deformed_overlap = ((healthy_mask != 0) & (deformed_mask != 0)) 
            healthy_dead_overlap = ((healthy_mask != 0) & (dead_mask != 0))
            deformed_dead_overlap = ((deformed_mask != 0) & (dead_mask != 0))
            
            condition = ((healthy_deformed_overlap) | (healthy_dead_overlap) | (deformed_dead_overlap))
            
            if self.manage_overlap == OVERLAP_EXCLUDE_REGION:
                healthy_mask = numpy.where(condition, 0, healthy_mask)
                deformed_mask = numpy.where(condition, 0, deformed_mask)
                dead_mask = numpy.where(condition, 0, dead_mask)
            
            if self.manage_overlap == OVERLAP_EXCLUDE_OBJECTS:
                masks = [healthy_mask, deformed_mask, dead_mask]
                for i, mask in enumerate(masks):
                    overlap_objects = numpy.unique(mask[numpy.where(condition)])
                    for overlap in overlap_objects:
                        masks[i] = mask[mask == overlap] = 0

        # Determine the amount of accepted zebrafish. '-1' is to exclude 0's 
        healthy_unique = len(numpy.unique(healthy_mask)) - 1
        deformed_unique = len(numpy.unique(deformed_mask)) - 1
        dead_unique = len(numpy.unique(dead_mask)) - 1
        object_count = healthy_unique + deformed_unique + dead_unique
        
        print("object_count", object_count)

        statistics = workspace.display_data.statistics
        statistics.append(["# of accepted objects", "%d" % object_count])
        workspace.display_data.image = orig_image
        workspace.display_data.healthy = healthy_mask
        workspace.display_data.deformed = deformed_mask
        workspace.display_data.dead = dead_mask

        masks = [healthy_mask, deformed_mask, dead_mask]

        for (total, mask) in zip(self.total_images, masks):
            self.create_output(
                workspace,
                image_name,
                total.output_image_name.value,
                mask,
            )

        
    def create_output(self, workspace, input_image_name, output_name, int_output_image):
        input_image = workspace.image_set.get_image(input_image_name)
        output_image = cellprofiler_core.image.Image(
            image=int_output_image,
            parent_image=input_image,
            dimensions=input_image.dimensions
        )

        workspace.image_set.add(output_name, output_image)


    def display(self, workspace, figure):
        if self.show_window:
            figure.set_subplots((2, 2))
            orig_ax = figure.subplot(0, 0)
            healthy_ax = figure.subplot(0, 1, sharexy=orig_ax)
            deformed_ax = figure.subplot(1, 0, sharexy=orig_ax)
            dead_ax = figure.subplot(1, 1, sharexy=orig_ax)

            title = "Input image, cycle #%d" % (workspace.measurements.image_number,)
            image = workspace.display_data.image
            healthy = workspace.display_data.healthy
            deformed = workspace.display_data.deformed
            dead = workspace.display_data.dead

            figure.subplot_imshow_grayscale(0, 0, image, title)
            figure.subplot_imshow(0, 1, healthy, "Healthy")
            figure.subplot_imshow(1, 0, deformed, "Deformed")
            figure.subplot_imshow(1, 1, dead, "Dead")

    def get_measurement_columns(self, pipeline):
        columns = []
        columns += super(IdentifyZebrafish, self).get_measurement_columns(
            pipeline
        )

        return columns

    def get_categories(self, pipeline, object_name):
        categories = []
        categories += super(IdentifyZebrafish, self).get_categories(
            pipeline, object_name
        )

        return categories

    def get_measurements(self, pipeline, object_name, category):
        measurements = []
        measurements += super(IdentifyZebrafish, self).get_measurements(
            pipeline, object_name, category
        )

        return measurements

    def get_measurement_objects(self, pipeline, object_name, category, measurement):
        return []
