"""Results analysis and visualization utilities."""

import logging
import json
from pathlib import Path
from typing import Dict, Any, List, Optional

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import torch


logger = logging.getLogger(__name__)


class ResultsAnalyzer:
    """Analyze and visualize segmentation results."""

    def __init__(
        self,
        num_classes: int,
        class_names: Optional[List[str]] = None,
        output_dir: str = './results',
    ):
        """Initialize results analyzer.

        Args:
            num_classes: Number of segmentation classes.
            class_names: Optional list of class names.
            output_dir: Directory to save analysis results.
        """
        self.num_classes = num_classes
        self.class_names = class_names or [f"class_{i}" for i in range(num_classes)]
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def save_metrics(
        self,
        metrics: Dict[str, float],
        filename: str = 'metrics.json',
    ) -> None:
        """Save metrics to JSON file.

        Args:
            metrics: Dictionary of metrics.
            filename: Output filename.
        """
        output_path = self.output_dir / filename

        # Convert numpy types to Python types for JSON serialization
        serializable_metrics = {}
        for key, value in metrics.items():
            if isinstance(value, (np.float32, np.float64)):
                serializable_metrics[key] = float(value)
            elif isinstance(value, (np.int32, np.int64)):
                serializable_metrics[key] = int(value)
            elif np.isnan(value):
                serializable_metrics[key] = None
            else:
                serializable_metrics[key] = value

        with open(output_path, 'w') as f:
            json.dump(serializable_metrics, f, indent=2)

        logger.info(f"Saved metrics to {output_path}")

    def plot_confusion_matrix(
        self,
        confusion_matrix: np.ndarray,
        filename: str = 'confusion_matrix.png',
        normalize: bool = True,
    ) -> None:
        """Plot and save confusion matrix.

        Args:
            confusion_matrix: Confusion matrix (num_classes, num_classes).
            filename: Output filename.
            normalize: Whether to normalize the matrix.
        """
        if normalize:
            # Normalize by row (true labels)
            row_sums = confusion_matrix.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1  # Avoid division by zero
            cm = confusion_matrix.astype(float) / row_sums
            fmt = '.2f'
        else:
            cm = confusion_matrix
            fmt = 'd'

        plt.figure(figsize=(12, 10))
        plt.imshow(cm, interpolation='nearest', cmap='Blues')
        plt.title('Confusion Matrix' + (' (Normalized)' if normalize else ''))
        plt.colorbar()

        tick_marks = np.arange(self.num_classes)
        plt.xticks(tick_marks, self.class_names, rotation=45, ha='right')
        plt.yticks(tick_marks, self.class_names)

        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()

        output_path = self.output_dir / filename
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        logger.info(f"Saved confusion matrix to {output_path}")

    def plot_per_class_metrics(
        self,
        metrics: Dict[str, float],
        filename: str = 'per_class_iou.png',
    ) -> None:
        """Plot per-class IoU scores.

        Args:
            metrics: Dictionary containing per-class IoU metrics.
            filename: Output filename.
        """
        # Extract per-class IoU
        ious = []
        names = []
        for name in self.class_names:
            key = f'iou_{name}'
            if key in metrics and not np.isnan(metrics[key]):
                ious.append(metrics[key])
                names.append(name)

        if not ious:
            logger.warning("No valid IoU scores to plot")
            return

        # Sort by IoU value
        sorted_indices = np.argsort(ious)
        ious = [ious[i] for i in sorted_indices]
        names = [names[i] for i in sorted_indices]

        plt.figure(figsize=(12, 6))
        bars = plt.barh(range(len(names)), ious)

        # Color bars based on performance
        for i, bar in enumerate(bars):
            if ious[i] >= 0.7:
                bar.set_color('green')
            elif ious[i] >= 0.5:
                bar.set_color('orange')
            else:
                bar.set_color('red')

        plt.yticks(range(len(names)), names)
        plt.xlabel('IoU Score')
        plt.title('Per-Class IoU Performance')
        plt.xlim(0, 1.0)
        plt.grid(axis='x', alpha=0.3)
        plt.tight_layout()

        output_path = self.output_dir / filename
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        logger.info(f"Saved per-class IoU plot to {output_path}")

    def plot_training_curves(
        self,
        history: Dict[str, List[float]],
        filename: str = 'training_curves.png',
    ) -> None:
        """Plot training and validation loss curves.

        Args:
            history: Training history dictionary.
            filename: Output filename.
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Loss curves
        axes[0].plot(history['train_losses'], label='Train Loss', linewidth=2)
        axes[0].plot(history['val_losses'], label='Val Loss', linewidth=2)
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training and Validation Loss')
        axes[0].legend()
        axes[0].grid(alpha=0.3)

        # Learning rate schedule
        if 'learning_rates' in history:
            axes[1].plot(history['learning_rates'], linewidth=2, color='green')
            axes[1].set_xlabel('Epoch')
            axes[1].set_ylabel('Learning Rate')
            axes[1].set_title('Learning Rate Schedule')
            axes[1].set_yscale('log')
            axes[1].grid(alpha=0.3)

        plt.tight_layout()

        output_path = self.output_dir / filename
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        logger.info(f"Saved training curves to {output_path}")

    def visualize_predictions(
        self,
        images: torch.Tensor,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        num_samples: int = 4,
        filename: str = 'predictions.png',
    ) -> None:
        """Visualize model predictions alongside ground truth.

        Args:
            images: Input images (B, 3, H, W).
            predictions: Predicted segmentation masks (B, H, W).
            targets: Ground truth masks (B, H, W).
            num_samples: Number of samples to visualize.
            filename: Output filename.
        """
        num_samples = min(num_samples, images.size(0))

        fig, axes = plt.subplots(num_samples, 3, figsize=(15, 5 * num_samples))
        if num_samples == 1:
            axes = axes.reshape(1, -1)

        for i in range(num_samples):
            # Denormalize image for visualization
            img = images[i].cpu().permute(1, 2, 0).numpy()
            img = (img - img.min()) / (img.max() - img.min() + 1e-8)

            # Image
            axes[i, 0].imshow(img)
            axes[i, 0].set_title('Input Image')
            axes[i, 0].axis('off')

            # Ground truth
            axes[i, 1].imshow(targets[i].cpu().numpy(), cmap='tab20')
            axes[i, 1].set_title('Ground Truth')
            axes[i, 1].axis('off')

            # Prediction
            axes[i, 2].imshow(predictions[i].cpu().numpy(), cmap='tab20')
            axes[i, 2].set_title('Prediction')
            axes[i, 2].axis('off')

        plt.tight_layout()

        output_path = self.output_dir / filename
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        logger.info(f"Saved prediction visualizations to {output_path}")

    def generate_report(
        self,
        metrics: Dict[str, float],
        confusion_matrix: Optional[np.ndarray] = None,
        history: Optional[Dict[str, List[float]]] = None,
    ) -> None:
        """Generate comprehensive evaluation report.

        Args:
            metrics: Dictionary of evaluation metrics.
            confusion_matrix: Optional confusion matrix.
            history: Optional training history.
        """
        # Save metrics
        self.save_metrics(metrics, 'evaluation_metrics.json')

        # Plot per-class metrics
        self.plot_per_class_metrics(metrics, 'per_class_iou.png')

        # Plot confusion matrix
        if confusion_matrix is not None:
            self.plot_confusion_matrix(confusion_matrix, 'confusion_matrix.png')

        # Plot training curves
        if history is not None:
            self.plot_training_curves(history, 'training_curves.png')

        logger.info(f"Generated comprehensive report in {self.output_dir}")
