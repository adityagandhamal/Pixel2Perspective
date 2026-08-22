
# Multi-Source RGB-IR Uncertainty Quantification (MSRIQ), from the Pixel2Perspective paper (Eq. 2):
#
#   IoU_MSRIQ = |P_RGB ∩ P_IR| / |P_RGB ∪ P_IR|
#
# Unlike standard mIoU, this does NOT compare a prediction against ground truth -- it compares a
# model's prediction on the RGB frame against its own prediction on the co-registered IR frame of
# the same scene. A low MSRIQ means the model's predictions disagree strongly between modalities,
# indicating high cross-modal uncertainty (see Fig. 2 of the paper). Averaged per class, the same
# way standard mIoU is, using one modality's prediction as a stand-in "reference" for the other.

import numpy as np


def confusion_matrix(pred_a, pred_b, num_classes, ignore_index=255):
    """
    Build a (num_classes x num_classes) confusion matrix between two predicted label maps of the
    same shape. `pred_b` plays the role ground truth normally plays in a standard confusion matrix.
    """
    pred_a = np.asarray(pred_a).reshape(-1)
    pred_b = np.asarray(pred_b).reshape(-1)
    assert pred_a.shape == pred_b.shape, f'{pred_a.shape} != {pred_b.shape}'

    valid = (pred_a != ignore_index) & (pred_b != ignore_index)
    pred_a, pred_b = pred_a[valid], pred_b[valid]

    conf = np.bincount(
        num_classes * pred_b.astype(np.int64) + pred_a.astype(np.int64),
        minlength=num_classes ** 2,
    ).reshape(num_classes, num_classes)
    return conf


def msriq_from_confusion_matrix(conf_matrix, classes_of_interest=None):
    """
    Computes MSRIQ (mean per-class IoU between two prediction sources) and per-class IoU from an
    accumulated confusion matrix, analogous to standard mIoU.
    """
    num_classes = conf_matrix.shape[0]
    tp = conf_matrix.diagonal().astype(float)
    pos_b = conf_matrix.sum(axis=1).astype(float)  # rows: pred_b (reference)
    pos_a = conf_matrix.sum(axis=0).astype(float)  # cols: pred_a
    union = pos_a + pos_b - tp

    iou = np.full(num_classes, np.nan, dtype=float)
    valid = union > 0
    iou[valid] = tp[valid] / union[valid]

    if classes_of_interest is not None:
        coi = np.zeros(num_classes, dtype=bool)
        coi[list(classes_of_interest)] = True
        valid = valid & coi

    msriq = float(np.mean(iou[valid])) if valid.any() else float('nan')
    return msriq, iou


class MSRIQAccumulator:
    """
    Accumulates a confusion matrix between paired RGB/IR predictions across a dataset.

    Usage:
        acc = MSRIQAccumulator(num_classes=9, classes_of_interest=range(1, 9))
        for pred_rgb, pred_ir in paired_predictions:
            acc.update(pred_rgb, pred_ir)
        msriq, per_class_iou = acc.evaluate()
    """

    def __init__(self, num_classes, ignore_index=255, classes_of_interest=None):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.classes_of_interest = classes_of_interest
        self._conf_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)

    def update(self, pred_rgb, pred_ir):
        self._conf_matrix += confusion_matrix(
            pred_rgb, pred_ir, self.num_classes, ignore_index=self.ignore_index
        )

    def evaluate(self):
        return msriq_from_confusion_matrix(self._conf_matrix, self.classes_of_interest)
