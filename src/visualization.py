from typing import Any, Callable, List, NamedTuple

import torch
import matplotlib.pyplot as plt
from detectron2.structures.boxes import matched_boxlist_iou
from detectron2.utils.visualizer import Visualizer
from detectron2.structures import polygons_to_bitmask
from detectron2.data import detection_utils as utils
from sklearn.metrics import confusion_matrix
from pathlib import Path

Match = NamedTuple('Match',
                   [('gt_index', int), ('pred_index', int), ("IoU", float), ("gt_class", int), ("pred_class", int)])


def create_confusion_matrix(ground_truths, predictions, meta_dataset, threshold_score=0.5, IoU_threshold=0.5):
    labels = ["Background"] + meta_dataset.thing_classes
    all_y_pred, all_y_true = evaluate_confusion_on_multiple(ground_truths, predictions, threshold_score, IoU_threshold)

    return confusion_matrix(all_y_pred, all_y_true, labels=list(range(-1, len(labels) - 1))), labels, all_y_pred, all_y_true


def evaluate_confusion_on_multiple(ground_truths, predictions, threshold_score=0.5, IoU_threshold=0.5):
    all_y_pred = []
    all_y_true = []

    for ground_truths, pred_instances in zip(ground_truths, predictions):
        gt_instances = ground_truths["instances"]
        height = ground_truths["height"]
        width = ground_truths["width"]

        y_pred, y_true = evaluate_confusion(gt_instances, pred_instances, height, width, threshold_score, IoU_threshold)

        all_y_pred += y_pred
        all_y_true += y_true

    return all_y_pred, all_y_true


def evaluate_confusion(gt_instances, pred_instances, height, width, threshold_score, IoU_threshold):
    n_gt_instances = len(gt_instances)
    n_pred_instances = len(pred_instances)

    matches = []

    for i in range(n_gt_instances):
        gt_instance = gt_instances[i]
        #gt_mask = torch.from_numpy(polygons_to_bitmask(gt_instance.gt_masks.polygons[0], height, width))
        gt_box = gt_instance.gt_boxes

        gt_class = gt_instance.gt_classes.cpu().numpy()[0]
        for j in range(n_pred_instances):
            pred_instance = pred_instances[j]
            #pred_mask = pred_instance.pred_masks.squeeze()
            pred_box = pred_instance.pred_boxes

            pred_score = pred_instance.scores[0].cpu().numpy()
            #IoU = _mask_IoU(gt_mask, pred_mask)
            IoU = float(matched_boxlist_iou(gt_box, pred_box)[0].cpu().numpy())
            pred_class = pred_instance.pred_classes.cpu().numpy()[0]

            if pred_score >= threshold_score and IoU >= IoU_threshold:
                matches.append(Match(gt_index=i, pred_index=j, IoU=IoU, gt_class=gt_class, pred_class=pred_class))

    matches = _filter_matches_by_key(matches, lambda m: m.gt_index)
    matches = _filter_matches_by_key(matches, lambda m: m.pred_index)

    matched_gts = list(map(lambda x: x.gt_index, matches))
    matched_preds = list(map(lambda x: x.pred_index, matches))

    unmatched_gts = list(filter(lambda x: x not in matched_gts, range(n_gt_instances)))
    unmatched_pred = list(filter(lambda x: x not in matched_preds, range(n_pred_instances)))

    y_pred = []
    y_true = []

    for match in matches:
        y_pred.append(match.pred_class)
        y_true.append(match.gt_class)

    for gt_idx in unmatched_gts:
        gt_instance = gt_instances[gt_idx]
        gt_class = gt_instance.gt_classes.cpu().numpy()[0]
        y_pred.append(-1)
        y_true.append(gt_class)

    for gt_idx in unmatched_pred:
        pred_instance = pred_instances[gt_idx]
        pred_class = pred_instance.pred_classes.cpu().numpy()[0]
        y_pred.append(pred_class)
        y_true.append(-1)

    return y_pred, y_true


def _mask_IoU(one, other):
    assert one.shape == other.shape, f"{one.shape} != {other.shape}"
    s = one.float() + other.float()
    union = float((s >= 1).sum().float())
    interscation = float((s >= 2).sum())

    return interscation / union


def _filter_matches_by_key(matches: List[Match], key: Callable[[Match], Any]):
    matches_per_key = dict()
    for match in matches:
        k = key(match)
        if k not in matches_per_key or match.IoU > matches_per_key[k].IoU:
            matches_per_key[k] = match

    return list(matches_per_key.values())


def plot_prediction_and_ground_truth(item, prediction, meta_dataset):
    image_bgr = utils.read_image(item["file_name"], format="BGR")
    
    visualizer = Visualizer(image_bgr , metadata=meta_dataset, scale=0.5)
    out_prediction = visualizer.draw_instance_predictions(prediction)
    
    visualizer = Visualizer(image_bgr , metadata=meta_dataset, scale=0.5)
    out_gt = visualizer.draw_dataset_dict(item)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(25, 25))
    fig.suptitle(Path(item["file_name"]).name)
    ax1.imshow(out_prediction.get_image())

    
    ax2.imshow(out_gt.get_image())
    plt.show()