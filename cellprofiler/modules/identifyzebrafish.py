# coding=utf-8

import numpy
from src.models.caffe.Caffe2Model import Caffe2Model
from itertools import combinations
import src.utils as utils

import os.path
import cv2
from configparser import SafeConfigParser

from cellprofiler_core.module import ImageProcessing
from cellprofiler_core.image import Image
from cellprofiler_core.setting import Binary, Divider, HiddenCount, SettingsGroup
from cellprofiler_core.setting.choice import Choice
from cellprofiler_core.setting.do_something import DoSomething, RemoveSettingButton
from cellprofiler_core.setting.text import ImageName

import pickle
__doc__ = ""


# Settings text which is referenced in various places in the help
X_NAME_TEXT = "Select the input image."
Y_NAME_TEXT = "Name the zebrafish to be identified."
MANAGE_INTER_OVERLAP_TEXT = "How to handle overlapping regions between different types of zebrafish?"
MANAGE_INTRA_OVERLAP_TEXT = "How to handle overlapping regions between the same types of zebrafish?"
MANAGE_REQUIRE_CONNECTION_TEXT = "Discard predicted sections that are not connected to the main instance?"
MANAGE_YOLK_TEXT = "Discard yolk from the predicted mask?"
INPUT_GROUP_COUNT_TEXT = "Input group count."
GENERATE_IMAGE_FOR_TEXT = "For which type of zebrafish do you want to generate an image?"
ADD_BUTTON_TEXT = "Add a new image"

# Doc
X_NAME_DOC = "Select the image that you want to use to identify zebrafish."
Y_NAME_DOC = "Enter the name that you want to call the zebrafish identified by this module."
MANAGE_INTRA_OVERLAP_DOC = ""
MANAGE_INTER_OVERLAP_DOC = ""
MANAGE_REQUIRE_CONNECTION_DOC = ""
MANAGE_YOLK_DOC = ""

# Settings choices
INTER_OVERLAP_ALLOW = "Allow overlap between different types of zebrafish."
INTER_OVERLAP_ASSIGN = "Assign region to one instance in order of selection."
INTER_OVERLAP_EXCLUDE_REGION = "Exclude the region that has overlap from all instances."
INTER_OVERLAP_EXCLUDE_INSTANCE = "Exclude all instances that contain overlapping regions."
INTER_OVERLAP_ALL = [
    INTER_OVERLAP_ALLOW,
    INTER_OVERLAP_ASSIGN,
    INTER_OVERLAP_EXCLUDE_REGION,
    INTER_OVERLAP_EXCLUDE_INSTANCE
]

INTRA_OVERLAP_ASSIGN = "Assign region to one instance in order of selection."
INTRA_OVERLAP_EXCLUDE_REGION = "Exclude the region that has overlap from all instances."
INTRA_OVERLAP_EXCLUDE_INSTANCES = "Exclude all instances that contain overlapping regions."
INTRA_OVERLAP_ALL = [
    INTRA_OVERLAP_ASSIGN,
    INTRA_OVERLAP_EXCLUDE_REGION,
    INTRA_OVERLAP_EXCLUDE_INSTANCES
]

# Constants
THRESHOLD = 0.6

# Paths
WEIGHT_FOLDER_PATH = r'weights/caffe2/large_fish'
MODEL_FILE_PATH = r'model/model.txt'
OUTPUT_FOLDER_PATH_TEST = r'outputs/output.txt'
LABELS_FILE_PATH = r'labels/additional_labels.cfg'

# Error messages
CONFIG_FILE_NOT_READ_TEXT = """\
Config file with additional labels was not read. Please make sure it exists 
as *{LABELS_FILE_PATH}* in the CellProfiler installation directory.
Only ZebrafishHealthy will be available.""".format(
    **{
        "LABELS_FILE_PATH": LABELS_FILE_PATH
    }
)

TESTMODE = True

NAME_HEALTHY = "ZebrafishHealthy"
NAME_ALL = [
    NAME_HEALTHY,
]

class ConfigFileNotReadError(Exception): pass

class SameClassError(Exception): pass

def get_additional_labels(names, path):
    parser = SafeConfigParser()
    parser.read(path)

    for key, entry in parser.items('labels'):
        names.append(entry)
    return names

try:
    NAME_ALL = get_additional_labels(NAME_ALL, LABELS_FILE_PATH)
except ConfigFileNotReadError:
    print(CONFIG_FILE_NOT_READ_TEXT)

MAX_MASK_COUNT = len(NAME_ALL)

class IdentifyZebrafish(ImageProcessing):
    variable_revision_number = 1

    category = "Image Processing"

    module_name = "IdentifyZebrafish"

    def __init__(self):
        super(IdentifyZebrafish, self).__init__()

    def volumetric(self):
        return False

    def non_max_suppression(self, pred_boxes, overlapThresh):
        """
        Non-max suppression is used to ensure that bounding boxes that are predicted multiple times
        are only displayed once.
        """
        boxes = numpy.zeros(shape=(len(pred_boxes), 4))
        for i, box in enumerate(pred_boxes):
            boxes[i] = box.numpy()

        if len(boxes) == 0:
            return []
        
        pick = []
        
        x1 = boxes[:,0]
        y1 = boxes[:,1]
        x2 = boxes[:,2]
        y2 = boxes[:,3]
        
        area = (x2 - x1 + 1) * (y2 - y1 + 1)
        idxs = numpy.argsort(y2)
        
        while len(idxs) > 0:
            
            last = len(idxs) - 1
            i = idxs[last]
            pick.append(i)
            
            xx1 = numpy.maximum(x1[i], x1[idxs[:last]])
            yy1 = numpy.maximum(y1[i], y1[idxs[:last]])
            xx2 = numpy.minimum(x2[i], x2[idxs[:last]])
            yy2 = numpy.minimum(y2[i], y2[idxs[:last]])
            
            w = numpy.maximum(0, xx2 - xx1 + 1)
            h = numpy.maximum(0, yy2 - yy1 + 1)
            
            overlap = (w * h) / area[idxs[:last]]
            
            idxs = numpy.delete(idxs, numpy.concatenate(([last],
                numpy.where(overlap > overlapThresh)[0])))
        
        return pick

    def create_settings(self):
        super(IdentifyZebrafish, self).create_settings()

        self.x_name.text = X_NAME_TEXT
        self.x_name.doc = X_NAME_DOC

        self.y_name.text = Y_NAME_TEXT
        self.y_name.doc = Y_NAME_DOC

        self.manage_intra_overlap = Choice(
            MANAGE_INTRA_OVERLAP_TEXT,
            INTRA_OVERLAP_ALL,
            doc=MANAGE_INTRA_OVERLAP_DOC,
        )

        self.manage_inter_overlap = Choice(
            MANAGE_INTER_OVERLAP_TEXT,
            INTER_OVERLAP_ALL,
            doc=MANAGE_INTER_OVERLAP_DOC,
        )

        self.require_connection = Binary(
            MANAGE_REQUIRE_CONNECTION_TEXT,
            False,
            doc=MANAGE_REQUIRE_CONNECTION_DOC,
        )

        self.discard_yolk = Binary(
            MANAGE_YOLK_TEXT,
            False,
            doc=MANAGE_YOLK_DOC,
        )

        self.separator = Divider(
            line=True
        )

        self.input_groups = []

        self.input_group_count = HiddenCount(
            self.input_groups,
            INPUT_GROUP_COUNT_TEXT
        )

        self.add_button = DoSomething(
            "",
            ADD_BUTTON_TEXT,
            self.add_image
        )   

    def cleanup_mask(self, mask):
        connection_map = cv2.connectedComponents(mask)[1]
        unique_values, sizes = numpy.unique(connection_map, return_counts=True)
        unique_values = unique_values[1:]
        sizes = sizes[1:]
        if len(unique_values) > 2:
            new_mask = numpy.zeros(mask.shape)
            max_index = numpy.argmax(sizes)
            non_max_indices = unique_values[numpy.arange(len(unique_values))!=max_index]
            for index in non_max_indices:
                new_mask = numpy.where(connection_map == index, 0, mask)
            return new_mask
        return mask

    def add_image(self, can_remove=True, name='--Please choose from the list--'):
        group = SettingsGroup()

        group.append("separator", Divider(line=True))

        if self.input_group_count.value == 0:
            can_remove = False
            name = NAME_HEALTHY

        if can_remove:
            group.append(
                "generate_image_for",
                Choice(
                    text=GENERATE_IMAGE_FOR_TEXT,
                    choices=NAME_ALL[1:],
                    value=name,
                    doc="",
                )
            )

        else:
            group.append(
                "generate_image_for",
                Choice(
                    text=GENERATE_IMAGE_FOR_TEXT,
                    choices=[NAME_HEALTHY],
                    value=NAME_HEALTHY,
                    doc="",
                )
            )

        group.append(
            "output_image_name",
            ImageName(
                "Name the output image",
                "ZebrafishNew",
                doc="",
            )
        )

        if can_remove:
            group.append(
                "remove_button",
                RemoveSettingButton(
                    "",
                    "Remove",
                    self.input_groups,
                    group
                )
            )

        self.input_groups.append(group)

    def settings(self):
        settings = super(IdentifyZebrafish, self).settings()

        settings += [
            self.x_name,
            self.manage_intra_overlap,
            self.manage_inter_overlap,
            self.input_group_count,
            self.require_connection,
            self.discard_yolk,
        ]

        for group in self.input_groups:
            settings += [
                group.generate_image_for,
                group.output_image_name,
            ]

        return settings

    def visible_settings(self):
        visible_settings = [
            self.x_name,
            self.manage_intra_overlap,            
            self.manage_inter_overlap,
            self.require_connection,
            self.discard_yolk,
        ]

        for total in self.input_groups:
            visible_settings += total.visible_settings()

        if self.input_group_count.value < MAX_MASK_COUNT:
            visible_settings += [self.add_button]

        return visible_settings

    def prepare_settings(self, setting_values):
        try:
            # Reset setting_values[5] to the correct value
            setting_count = int(setting_values[5])
            # print("current allowed", setting_count)
            # setting_values[5] = str(len(NAME_ALL))
            # setting_count = len(NAME_ALL)
            # print("adapted allowed", setting_count)
            # print("actual allowed:", len(NAME_ALL))
            
            if len(self.input_groups) > setting_count:
                del self.input_groups[setting_count:]
            else:
                for _, name in zip(
                    range(len(self.input_groups), setting_count), 
                    NAME_ALL
                ):
                    can_remove = False if name == NAME_HEALTHY else True
                    self.add_image(can_remove=can_remove, name=name)
        except ValueError:
            logging.warning(
                'Additional image setting count was "%s" which is not an integer.',
                setting_values[5],
                exc_info=True,
            )
            pass

    def get_classes_to_predict(self, workspace):
        classes_to_predict = []
        class_names = []
        for i, group in enumerate(self.input_groups):
            name = group.generate_image_for.value
            if name not in class_names:
                classes_to_predict.append(i)
                class_names.append(name)
            else:
                raise SameClassError(
                    "Selected the same class twice. Please make sure classes are only selected once.")

        return classes_to_predict, class_names

    def get_model(self):
        print("Generating model ...")
        model = Caffe2Model.load_protobuf(WEIGHT_FOLDER_PATH)
        print("Generated model ...")
        return model

    def post_processing(self, masks, classes_to_predict):
        """ 
        More post-processing can be added here. 'masks' is a list of length 
        len(NAME_ALL) containing a (HxW) mask of of every class that was selected
        by the user. If a class was not selected 'mask is None' evaluates to True.
        """       
        masks = self.handle_inter_class_overlap(masks, classes_to_predict)
        return masks
        
    def handle_intra_class_overlap(self, old_mask, added_mask):
        masks_overlap = (old_mask != 0) & (added_mask != 0)
        both_masks = old_mask + added_mask
        if not masks_overlap.any():
            return both_masks

        if self.manage_intra_overlap == INTRA_OVERLAP_ASSIGN:
            new_mask = numpy.where(masks_overlap, old_mask, both_masks)

        elif self.manage_intra_overlap == INTRA_OVERLAP_EXCLUDE_REGION:
            new_mask = numpy.where(masks_overlap, 0, both_masks)

        elif self.manage_intra_overlap == INTRA_OVERLAP_EXCLUDE_INSTANCES:
            overlap_instances = (
                list(
                    numpy.unique(old_mask[numpy.where(masks_overlap)])) +
                list(
                    numpy.unique(added_mask[numpy.where(masks_overlap)]))
            )
            for instance in overlap_instances:
                old_mask[old_mask == instance] = -1
                added_mask[added_mask == instance] = -1
            new_mask = old_mask + added_mask

        else:
            raise NotImplementedError(
                "Choose a between-the-same-types overlap option from the provided list."
            )

        return new_mask

    def handle_inter_class_overlap(self, masks, classes_to_predict):
        # If overlap between different classes should be allowed, the masks should not be modified.
        if self.manage_inter_overlap == INTER_OVERLAP_ALLOW:
              return masks
        
        new_masks = masks.copy()

        combis = list(combinations(classes_to_predict, 2))
        for combi in combis:
            mask0, mask1 = masks[combi[0]], masks[combi[1]]
            if mask0 is None or mask1 is None:
                continue

            masks_overlap = (mask0 != 0) & (mask1 != 0)
            if not masks_overlap.any():
                continue  

            # Keep the value mask of the first class, discard the other.
            if self.manage_inter_overlap == INTER_OVERLAP_ASSIGN:
                mask0 = numpy.where(masks_overlap, mask0, mask0)
                mask1 = numpy.where(masks_overlap, 0,     mask1)

            # Discard the value of both masks
            elif self.manage_inter_overlap == INTER_OVERLAP_EXCLUDE_REGION:
                #condition = overlap_masks.any(axis=0)
                mask0 = numpy.where(masks_overlap, 0, mask0)
                mask1 = numpy.where(masks_overlap, 0, mask1)

            # Find the instance label of the instance that overlaps and set all occurances to 0.
            elif self.manage_inter_overlap == INTER_OVERLAP_EXCLUDE_INSTANCE:
                overlap_instance_mask0 = numpy.unique(
                    mask0[numpy.where(masks_overlap)]
                )
                for instance in overlap_instance_mask0:
                    mask0[mask0 == instance] = 0

                overlap_objects_mask1 = numpy.unique(
                    mask1[numpy.where(masks_overlap)]
                )
                for instance in overlap_objects_mask1:
                    mask1[mask1 == instance] = 0
            else:
                raise NotImplementedError(
                    "Choose a between-different-types overlap option from the provided list."
                )

            new_masks[combi[0]] = numpy.where(
                (new_masks[combi[0]] == 0) | (mask0 == 0), 0, mask0)
            new_masks[combi[1]] = numpy.where(
                (new_masks[combi[1]] == 0) | (mask1 == 0), 0, mask1)

        return new_masks

    def convert(self, orig_image):
        """
        The module expects the input image to be an RGB image. This is converted to BGR before
        providing it to the DL model. Any future transformations that need to be done before 
        passing an image through the DL model can be added here.
        """
        r, g, b = cv2.split(orig_image)
        bgr_image = cv2.merge((b, g, r))
        return bgr_image

    def generate_output(self, input_):
        """
        Passing the images through the DL model.
        """
        if TESTMODE:
            if os.path.exists(OUTPUT_FOLDER_PATH_TEST):
                with open(OUTPUT_FOLDER_PATH_TEST, 'rb') as f:
                    output = pickle.load(f)
            else:
                model = self.get_model()
                output = model(input_)
                with open(OUTPUT_FOLDER_PATH_TEST, 'wb') as f:
                    pickle.dump(output, f)
        else:
            print("Analysing images ...")
            model = self.get_model()
            output = model(input_)
            print("Analysis done ...")
        return output

    def get_empty_masks(self, classes_to_predict, parent_image_pixels_shape):
        """
        Initialize masks. If a class is not selected by the user, None is appended, otherwise a numpy array 
        with the size of the image is filled with zeros. 
        """
        masks = []
        for potential_class in range(MAX_MASK_COUNT):
            if potential_class in classes_to_predict:
                masks.append(numpy.zeros(parent_image_pixels_shape))
            else:
                masks.append(None)
        return masks

    def run(self, workspace):
        workspace.display_data.statistics = []
        statistics = workspace.display_data.statistics
        image_name = self.x_name.value
        parent_image = workspace.image_set.get_image(image_name)
        parent_image_pixels = parent_image.pixel_data

        input_ = self.convert(parent_image_pixels)
        
        output = self.generate_output(input_)
        
        # Run NMS to select best boxes
        best_box_indices = self.non_max_suppression(output.pred_boxes, THRESHOLD)
        
        # Prepare list that will contain predicted masks
        classes_to_predict, class_names = self.get_classes_to_predict(
            workspace)
        
        # Create emtpy masks to fill with data
        masks = self.get_empty_masks(classes_to_predict, (parent_image_pixels.shape[0], parent_image_pixels.shape[1]))
        
        # Fill the empty masks with the instance_masks predicted by the DL model.
        masks = self.populate_masks(masks, output, classes_to_predict, best_box_indices)
            
        # Post processing
        masks = self.post_processing(masks, classes_to_predict)
       
        # Determine the amount of accepted zebrafish. '-1' is to exclude 0's
        instance_count = self.calc_number_of_accepted_instances(masks)
        statistics.append(["# of accepted objects", "%d" % instance_count])

        # Write output to make it visible for other modules
        for i, (mask, name) in enumerate(zip(masks, class_names)):
            if i not in classes_to_predict or mask is None:
                continue
            if mask.max() != 0:
                # Scale the mask to fit between 0.5 and 1 to ensure it is processed properly by ConvertImageToObjects.
                mask = mask.astype("float16")
                mask *= (1/(255 * 2))
                mask += 0.5
            self.provide_to_workspace(
                workspace=workspace,
                parent_image=parent_image,
                mask=mask,
                mask_name=name
            )

        # Write display information
        if self.show_window:
            workspace.display_data.parent_image_pixels = parent_image_pixels
            workspace.display_data.images = []
            workspace.display_data.image_names = []
            for (mask, name) in zip(masks, class_names):
                workspace.display_data.images.append(mask)
                workspace.display_data.image_names.append(name)

    def provide_to_workspace(self, workspace, parent_image, mask, mask_name):
        """
        Provides the images generated by the module to the workspace of CP.
        """
        output_mask = Image(
            image=mask,
            parent_image=parent_image,
            dimensions=parent_image.dimensions
        )

        workspace.image_set.add(mask_name, output_mask)

    def calc_number_of_accepted_instances(self, masks):
        instances = []
        for mask in masks:
            if mask is None:
                continue
            instances.extend(numpy.unique(mask))
        instances[:] = [entry for entry in instances if entry > 0]
        return len(instances)

    def display(self, workspace, figure):
        if self.show_window:
            width = -(-(self.input_group_count.value + 1) // 2)
            figure.set_subplots((width, 2))
            images = workspace.display_data.images

            orig_ax = figure.subplot(0, 0)
            for i, _ in enumerate(images):
                figure.subplot((i + 1) // 2, (i + 1) % 2, sharexy=orig_ax)

            image_names = workspace.display_data.image_names
            parent_image_pixels = workspace.display_data.parent_image_pixels

            figure.subplot_imshow_grayscale(0, 0, parent_image_pixels, "Input Image")
            for i, (image, name) in enumerate(zip(images, image_names)):
                figure.subplot_imshow((i + 1) // 2, (i + 1) % 2, image, name)

    def populate_masks(self, masks, output, classes_to_predict, best_box_indices):
        """
        Checks if the results of the DL model have class predictions that are selected and assign predictions to masks accordingly.
        Ocurrances of overlap are labeled with -1 and are set back to 0 after all instance_masks are added.
        """ 
        yolk_mask = numpy.zeros(masks[0].shape)
        for i, (instance_mask, label) in enumerate(zip(output.pred_masks, output.pred_classes)):
            if i not in best_box_indices:
                continue
            instance_mask = numpy.array(instance_mask, dtype=numpy.uint8)
            
            if self.require_connection:
                instance_mask = self.cleanup_mask(instance_mask)
            
            if label == 99:
                yolk_mask = numpy.where(instance_mask, 1, yolk_mask)

            elif label in classes_to_predict:    
                old_mask = masks[label]
                added_mask = instance_mask * (i + 1)
                new_mask = self.handle_intra_class_overlap(old_mask, added_mask)
                masks[label] = new_mask

        # Remove the negative numbers (indicating where overlap occured) from the masks
        for i, mask in enumerate(masks):
            if mask is None:
                continue
            mask[mask < 0] = 0
            masks[i] = mask

            if self.discard_yolk:
                mask[yolk_mask == 1] = 0

        

        return masks

    # def get_measurement_columns(self, pipeline):
    #     columns = []
    #     columns += super(IdentifyZebrafish, self).get_measurement_columns(
    #         pipeline
    #     )

    #     return columns

    # def get_categories(self, pipeline, object_name):
    #     categories = []
    #     categories += super(IdentifyZebrafish, self).get_categories(
    #         pipeline, object_name
    #     )

    #     return categories

    # def get_measurements(self, pipeline, object_name, category):
    #     measurements = []
    #     measurements += super(IdentifyZebrafish, self).get_measurements(
    #         pipeline, object_name, category
    #     )

    #     return measurements

    # def get_measurement_objects(self, pipeline, object_name, category, measurement):
    #     return []