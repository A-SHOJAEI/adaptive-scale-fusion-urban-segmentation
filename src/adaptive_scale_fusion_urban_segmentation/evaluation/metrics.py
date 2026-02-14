"""Evaluation metrics for semantic segmentation."""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F


logger = logging.getLogger(__name__)


def compute_iou(
    pred: torch.Tensor,
    target: torch.Tensor,
    num_classes: int,
    ignore_index: int = 255,
) -> torch.Tensor:
    """Compute Intersection over Union (IoU) per class.

    Args:
        pred: Predicted class indices (B, H, W).
        target: Ground truth class indices (B, H, W).
        num_classes: Number of classes.
        ignore_index: Index to ignore in computation.

    Returns:
        IoU scores per class (num_classes,).
    """
    ious = []

    for cls in range(num_classes):
        pred_mask = (pred == cls)
        target_mask = (target == cls)

        # Ignore specified index
        valid_mask = (target != ignore_index)
        pred_mask = pred_mask & valid_mask
        target_mask = target_mask & valid_mask

        intersection = (pred_mask & target_mask).sum().float()
        union = (pred_mask | target_mask).sum().float()

        if union == 0:
            iou = torch.tensor(float('nan'))
        else:
            iou = intersection / union

        ious.append(iou)

    return torch.tensor(ious)


def compute_dice(
    pred: torch.Tensor,
    target: torch.Tensor,
    num_classes: int,
    ignore_index: int = 255,
) -> torch.Tensor:
    """Compute Dice coefficient per class.

    Args:
        pred: Predicted class indices (B, H, W).
        target: Ground truth class indices (B, H, W).
        num_classes: Number of classes.
        ignore_index: Index to ignore in computation.

    Returns:
        Dice scores per class (num_classes,).
    """
    dice_scores = []

    for cls in range(num_classes):
        pred_mask = (pred == cls)
        target_mask = (target == cls)

        # Ignore specified index
        valid_mask = (target != ignore_index)
        pred_mask = pred_mask & valid_mask
        target_mask = target_mask & valid_mask

        intersection = (pred_mask & target_mask).sum().float()
        pred_sum = pred_mask.sum().float()
        target_sum = target_mask.sum().float()

        denominator = pred_sum + target_sum

        if denominator == 0:
            dice = torch.tensor(float('nan'))
        else:
            dice = (2.0 * intersection) / denominator

        dice_scores.append(dice)

    return torch.tensor(dice_scores)


class SegmentationMetrics:
    """Comprehensive metrics for semantic segmentation evaluation."""

    def __init__(
        self,
        num_classes: int,
        class_names: Optional[List[str]] = None,
        ignore_index: int = 255,
        small_classes: Optional[List[int]] = None,
        large_classes: Optional[List[int]] = None,
    ):
        """Initialize segmentation metrics.

        Args:
            num_classes: Number of classes.
            class_names: Optional list of class names.
            ignore_index: Index to ignore in computation.
            small_classes: Indices of small object classes.
            large_classes: Indices of large object classes.
        """
        self.num_classes = num_classes
        self.class_names = class_names or [f"class_{i}" for i in range(num_classes)]
        self.ignore_index = ignore_index
        self.small_classes = small_classes or []
        self.large_classes = large_classes or []

        self.reset()

    def reset(self) -> None:
        """Reset all accumulated metrics."""
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)
        self.total_samples = 0

    def update(self, pred: torch.Tensor, target: torch.Tensor) -> None:
        """Update metrics with new predictions.

        Args:
            pred: Predicted class indices (B, H, W) or logits (B, C, H, W).
            target: Ground truth class indices (B, H, W).
        """
        # Convert logits to predictions if needed
        if pred.dim() == 4:
            pred = pred.argmax(dim=1)

        pred = pred.cpu().numpy().flatten()
        target = target.cpu().numpy().flatten()

        # Filter out ignore index
        valid_mask = (target != self.ignore_index)
        pred = pred[valid_mask]
        target = target[valid_mask]

        # Update confusion matrix
        for p, t in zip(pred, target):
            if 0 <= t < self.num_classes and 0 <= p < self.num_classes:
                self.confusion_matrix[t, p] += 1

        self.total_samples += len(pred)

    def compute_metrics(self) -> Dict[str, float]:
        """Compute all evaluation metrics.

        Returns:
            Dictionary of metric names to values.
        """
        metrics = {}

        # Per-class IoU
        iou_per_class = []
        for i in range(self.num_classes):
            tp = self.confusion_matrix[i, i]
            fp = self.confusion_matrix[:, i].sum() - tp
            fn = self.confusion_matrix[i, :].sum() - tp

            if tp + fp + fn == 0:
                iou = float('nan')
            else:
                iou = tp / (tp + fp + fn)

            iou_per_class.append(iou)
            metrics[f'iou_{self.class_names[i]}'] = iou

        # Mean IoU (ignoring NaN values)
        valid_ious = [iou for iou in iou_per_class if not np.isnan(iou)]
        metrics['mean_iou'] = np.mean(valid_ious) if valid_ious else 0.0

        # Small object IoU
        if self.small_classes:
            small_ious = [iou_per_class[i] for i in self.small_classes if not np.isnan(iou_per_class[i])]
            metrics['small_object_iou'] = np.mean(small_ious) if small_ious else 0.0

        # Large object IoU
        if self.large_classes:
            large_ious = [iou_per_class[i] for i in self.large_classes if not np.isnan(iou_per_class[i])]
            metrics['large_object_iou'] = np.mean(large_ious) if large_ious else 0.0

        # Pixel accuracy
        correct = np.diag(self.confusion_matrix).sum()
        total = self.confusion_matrix.sum()
        metrics['pixel_accuracy'] = correct / total if total > 0 else 0.0

        # Mean accuracy (per-class accuracy)
        class_accuracies = []
        for i in range(self.num_classes):
            tp = self.confusion_matrix[i, i]
            total_cls = self.confusion_matrix[i, :].sum()
            if total_cls > 0:
                class_accuracies.append(tp / total_cls)

        metrics['mean_accuracy'] = np.mean(class_accuracies) if class_accuracies else 0.0

        # Frequency weighted IoU
        weights = self.confusion_matrix.sum(axis=1) / self.confusion_matrix.sum()
        weighted_ious = []
        for i, iou in enumerate(iou_per_class):
            if not np.isnan(iou):
                weighted_ious.append(weights[i] * iou)

        metrics['freq_weighted_iou'] = np.sum(weighted_ious) if weighted_ious else 0.0

        return metrics

    def get_confusion_matrix(self) -> np.ndarray:
        """Get the confusion matrix.

        Returns:
            Confusion matrix (num_classes, num_classes).
        """
        return self.confusion_matrix

    def print_summary(self) -> None:
        """Print a summary of the metrics."""
        metrics = self.compute_metrics()

        logger.info("=" * 80)
        logger.info("Segmentation Metrics Summary")
        logger.info("=" * 80)
        logger.info(f"Mean IoU: {metrics['mean_iou']:.4f}")
        logger.info(f"Pixel Accuracy: {metrics['pixel_accuracy']:.4f}")
        logger.info(f"Mean Accuracy: {metrics['mean_accuracy']:.4f}")
        logger.info(f"Frequency Weighted IoU: {metrics['freq_weighted_iou']:.4f}")

        if 'small_object_iou' in metrics:
            logger.info(f"Small Object IoU: {metrics['small_object_iou']:.4f}")

        if 'large_object_iou' in metrics:
            logger.info(f"Large Object IoU: {metrics['large_object_iou']:.4f}")

        logger.info("-" * 80)
        logger.info("Per-class IoU:")
        for i, name in enumerate(self.class_names):
            iou = metrics.get(f'iou_{name}', 0.0)
            if not np.isnan(iou):
                logger.info(f"  {name:20s}: {iou:.4f}")

        logger.info("=" * 80)
