
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import SimpleITK as sitk
import warnings

class NotComputableMetricWarning(RuntimeWarning):
    """Warning class to raise if a metric cannot be computed."""

class ConfusionMatrix:
    """Represents a confusion matrix (or error matrix)."""

    def __init__(self, prediction, label):
        """Initializes a new instance of the ConfusionMatrix class."""

        # true positive (tp): we predict a label of 1 (positive), and the true label is 1
        self.tp = np.sum(np.logical_and(prediction == 1, label == 1))
        # true negative (tn): we predict a label of 0 (negative), and the true label is 0
        self.tn = np.sum(np.logical_and(prediction == 0, label == 0))
        # false positive (fp): we predict a label of 1 (positive), but the true label is 0
        self.fp = np.sum(np.logical_and(prediction == 1, label == 0))
        # false negative (fn): we predict a label of 0 (negative), but the true label is 1
        self.fn = np.sum(np.logical_and(prediction == 0, label == 1))

        self.n = prediction.size


def VolumeSimilarity(confusion_matrix):
    """Calculates the volume similarity."""

    tp = confusion_matrix.tp
    fp = confusion_matrix.fp
    fn = confusion_matrix.fn

    return 1 - abs(fn - fp) / (2 * tp + fn + fp)


def DiceCoefficient(confusion_matrix):
    """Calculates the Dice coefficient."""

    if (confusion_matrix.tp == 0) and \
            ((confusion_matrix.tp + confusion_matrix.fp + confusion_matrix.fn) == 0):
        return 1.

    return 2 * confusion_matrix.tp / \
           (2 * confusion_matrix.tp + confusion_matrix.fp + confusion_matrix.fn)


def AverageDistance(ground_truth, segmentation):
    """Calculates the average (Hausdorff) distance."""

    if np.count_nonzero(sitk.GetArrayFromImage(ground_truth)) == 0:
        warnings.warn('Unable to compute average distance due to empty label mask, returning inf',
                      NotComputableMetricWarning)
        return float('inf')
    if np.count_nonzero(sitk.GetArrayFromImage(segmentation)) == 0:
        warnings.warn('Unable to compute average distance due to empty segmentation mask, returning inf',
                      NotComputableMetricWarning)
        return float('inf')

    distance_filter = sitk.HausdorffDistanceImageFilter()
    distance_filter.Execute(ground_truth, segmentation)
    return distance_filter.GetAverageHausdorffDistance()





