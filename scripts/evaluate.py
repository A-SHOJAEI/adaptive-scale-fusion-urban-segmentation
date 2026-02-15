#!/usr/bin/env python
"""Evaluation script for adaptive scale fusion urban segmentation."""

import sys
import argparse
import logging
import time
from pathlib import Path

# Add project root and src/ to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
import numpy as np

from adaptive_scale_fusion_urban_segmentation.utils.config import (
    load_config,
    set_random_seeds,
    get_device,
)
from adaptive_scale_fusion_urban_segmentation.data.loader import (
    get_dataloaders,
    CityscapesDataset,
)
from adaptive_scale_fusion_urban_segmentation.models.model import AdaptiveScaleFusionSegmenter
from adaptive_scale_fusion_urban_segmentation.evaluation.metrics import SegmentationMetrics
from adaptive_scale_fusion_urban_segmentation.evaluation.analysis import ResultsAnalyzer


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Evaluate adaptive scale fusion segmentation model'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to model checkpoint',
    )
    parser.add_argument(
        '--config',
        type=str,
        default='configs/default.yaml',
        help='Path to configuration file',
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./results',
        help='Directory to save evaluation results',
    )
    parser.add_argument(
        '--split',
        type=str,
        default='test',
        choices=['val', 'test'],
        help='Dataset split to evaluate on',
    )
    parser.add_argument(
        '--visualize',
        action='store_true',
        help='Generate prediction visualizations',
    )

    return parser.parse_args()


@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    metrics: SegmentationMetrics,
    device: torch.device,
    visualize: bool = False,
    analyzer: ResultsAnalyzer = None,
) -> tuple:
    """Evaluate model on a dataset.

    Args:
        model: Model to evaluate.
        dataloader: Data loader for evaluation.
        metrics: Metrics calculator.
        device: Device to run evaluation on.
        visualize: Whether to generate visualizations.
        analyzer: Results analyzer for visualization.

    Returns:
        Tuple of (metrics_dict, inference_time).
    """
    model.eval()
    metrics.reset()

    total_time = 0.0
    num_batches = 0

    # Collect samples for visualization
    vis_images = []
    vis_preds = []
    vis_targets = []

    for batch_idx, batch in enumerate(dataloader):
        images = batch['image'].to(device)
        targets = batch['mask'].to(device)

        # Measure inference time
        if device.type == 'cuda':
            torch.cuda.synchronize()

        start_time = time.time()
        outputs = model(images, return_aux=False)
        predictions = outputs['main'].argmax(dim=1)

        if device.type == 'cuda':
            torch.cuda.synchronize()

        elapsed = time.time() - start_time
        total_time += elapsed
        num_batches += 1

        # Update metrics
        metrics.update(predictions, targets)

        # Collect samples for visualization
        if visualize and batch_idx < 2:
            vis_images.append(images[:4].cpu())
            vis_preds.append(predictions[:4].cpu())
            vis_targets.append(targets[:4].cpu())

        if (batch_idx + 1) % 10 == 0:
            logger.info(f"Processed {batch_idx + 1}/{len(dataloader)} batches")

    # Compute metrics
    metrics_dict = metrics.compute_metrics()

    # Compute inference speed
    avg_time_per_batch = total_time / num_batches
    batch_size = dataloader.batch_size
    fps = batch_size / avg_time_per_batch
    metrics_dict['fps_inference'] = fps
    metrics_dict['avg_inference_time'] = avg_time_per_batch

    # Generate visualizations
    if visualize and analyzer is not None and vis_images:
        images_cat = torch.cat(vis_images, dim=0)
        preds_cat = torch.cat(vis_preds, dim=0)
        targets_cat = torch.cat(vis_targets, dim=0)

        analyzer.visualize_predictions(
            images_cat,
            preds_cat,
            targets_cat,
            num_samples=min(8, len(images_cat)),
        )

    return metrics_dict, total_time


def main():
    """Main evaluation function."""
    args = parse_args()

    # Load configuration
    logger.info(f"Loading configuration from {args.config}")
    config = load_config(args.config)

    # Set random seeds
    seed = config.get('seed', 42)
    set_random_seeds(seed)

    # Get device
    device = get_device(config)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Create dataloaders
        logger.info("Creating dataloaders...")
        train_loader, val_loader, test_loader = get_dataloaders(
            config,
            synthetic=True,
        )

        # Select appropriate loader
        if args.split == 'val':
            eval_loader = val_loader
        else:
            eval_loader = test_loader

        logger.info(f"Evaluating on {args.split} set with {len(eval_loader)} batches")

        # Create model
        logger.info("Creating model...")
        model_config = config.get('model', {})
        model = AdaptiveScaleFusionSegmenter(
            num_classes=model_config.get('num_classes', 19),
            encoder_name=model_config.get('encoder_name', 'resnet50'),
            pretrained=False,  # Will load from checkpoint
            num_scales=model_config.get('num_scales', 4),
            use_scale_attention=model_config.get('use_scale_attention', True),
            use_aux_head=model_config.get('use_aux_head', True),
        )
        model = model.to(device)

        # Load checkpoint
        logger.info(f"Loading checkpoint from {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)

        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            logger.info(f"Loaded model from epoch {checkpoint.get('epoch', 'unknown')}")
        else:
            model.load_state_dict(checkpoint)

        # Define class categories
        small_classes = [5, 6, 7, 11, 12, 17, 18]  # poles, lights, signs, person, rider, motorcycle, bicycle
        large_classes = [0, 2, 8, 10]  # road, building, vegetation, sky

        # Create metrics calculator
        metrics = SegmentationMetrics(
            num_classes=model_config.get('num_classes', 19),
            class_names=CityscapesDataset.CLASSES,
            small_classes=small_classes,
            large_classes=large_classes,
        )

        # Create results analyzer
        analyzer = ResultsAnalyzer(
            num_classes=model_config.get('num_classes', 19),
            class_names=CityscapesDataset.CLASSES,
            output_dir=output_dir,
        )

        # Evaluate model
        logger.info("Evaluating model...")
        metrics_dict, total_time = evaluate_model(
            model=model,
            dataloader=eval_loader,
            metrics=metrics,
            device=device,
            visualize=args.visualize,
            analyzer=analyzer,
        )

        # Print summary
        logger.info("=" * 80)
        logger.info("Evaluation Results")
        logger.info("=" * 80)
        logger.info(f"Mean IoU: {metrics_dict['mean_iou']:.4f}")
        logger.info(f"Pixel Accuracy: {metrics_dict['pixel_accuracy']:.4f}")
        logger.info(f"Mean Accuracy: {metrics_dict['mean_accuracy']:.4f}")
        logger.info(f"Small Object IoU: {metrics_dict.get('small_object_iou', 0.0):.4f}")
        logger.info(f"Large Object IoU: {metrics_dict.get('large_object_iou', 0.0):.4f}")
        logger.info(f"FPS (Inference): {metrics_dict['fps_inference']:.2f}")
        logger.info(f"Avg Inference Time: {metrics_dict['avg_inference_time']:.4f}s")
        logger.info("=" * 80)

        # Save metrics
        analyzer.save_metrics(metrics_dict, 'evaluation_metrics.json')

        # Generate analysis plots
        logger.info("Generating analysis plots...")
        analyzer.plot_per_class_metrics(metrics_dict)
        analyzer.plot_confusion_matrix(metrics.get_confusion_matrix())

        # Print per-class results
        metrics.print_summary()

        logger.info(f"Evaluation results saved to {output_dir}")

    except Exception as e:
        logger.error(f"Evaluation failed with error: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
