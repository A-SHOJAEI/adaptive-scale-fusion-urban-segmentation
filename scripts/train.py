#!/usr/bin/env python
"""Training script for adaptive scale fusion urban segmentation."""

import sys
import argparse
import logging
from pathlib import Path

# Add project root and src/ to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
import torch.nn as nn
from torch.optim import AdamW, SGD

from adaptive_scale_fusion_urban_segmentation.utils.config import (
    load_config,
    set_random_seeds,
    get_device,
    save_config,
)
from adaptive_scale_fusion_urban_segmentation.data.loader import get_dataloaders
from adaptive_scale_fusion_urban_segmentation.models.model import AdaptiveScaleFusionSegmenter
from adaptive_scale_fusion_urban_segmentation.models.components import MultiScaleLoss
from adaptive_scale_fusion_urban_segmentation.training.trainer import SegmentationTrainer
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
        description='Train adaptive scale fusion segmentation model'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='configs/default.yaml',
        help='Path to configuration file',
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        default=None,
        help='Path to checkpoint to resume training',
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./results',
        help='Directory to save results',
    )

    return parser.parse_args()


def create_optimizer(model: nn.Module, config: dict) -> torch.optim.Optimizer:
    """Create optimizer based on configuration.

    Args:
        model: Model to optimize.
        config: Configuration dictionary.

    Returns:
        Configured optimizer.
    """
    optimizer_name = config.get('optimizer', 'adamw').lower()
    lr = config.get('learning_rate', 0.001)
    weight_decay = config.get('weight_decay', 0.0001)

    if optimizer_name == 'adamw':
        optimizer = AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )
    elif optimizer_name == 'sgd':
        optimizer = SGD(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            momentum=0.9,
        )
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

    logger.info(f"Created {optimizer_name} optimizer with lr={lr}")
    return optimizer


def create_scheduler(
    optimizer: torch.optim.Optimizer,
    config: dict,
) -> torch.optim.lr_scheduler._LRScheduler:
    """Create learning rate scheduler.

    Args:
        optimizer: Optimizer to schedule.
        config: Configuration dictionary.

    Returns:
        Learning rate scheduler.
    """
    scheduler_config = config.get('scheduler', {})
    scheduler_type = scheduler_config.get('type', 'cosine').lower()
    epochs = config.get('epochs', 100)

    if scheduler_type == 'cosine':
        warmup_epochs = scheduler_config.get('warmup_epochs', 5)
        min_lr = scheduler_config.get('min_lr', 0.00001)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=epochs - warmup_epochs,
            eta_min=min_lr,
        )
    elif scheduler_type == 'step':
        step_size = scheduler_config.get('step_size', 30)
        gamma = scheduler_config.get('gamma', 0.1)

        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=step_size,
            gamma=gamma,
        )
    elif scheduler_type == 'plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.1,
            patience=5,
            verbose=True,
        )
    else:
        logger.warning(f"Unknown scheduler type: {scheduler_type}, using cosine")
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=epochs,
            eta_min=0.00001,
        )

    logger.info(f"Created {scheduler_type} learning rate scheduler")
    return scheduler


def main():
    """Main training function."""
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

    # Save configuration to output directory
    save_config(config, output_dir / 'config.yaml')

    try:
        # Create dataloaders
        logger.info("Creating dataloaders...")
        train_loader, val_loader, test_loader = get_dataloaders(
            config,
            synthetic=True,  # Use synthetic data for demonstration
        )
        logger.info(
            f"Created dataloaders: {len(train_loader)} train, "
            f"{len(val_loader)} val, {len(test_loader)} test batches"
        )

        # Create model
        logger.info("Creating model...")
        model_config = config.get('model', {})
        model = AdaptiveScaleFusionSegmenter(
            num_classes=model_config.get('num_classes', 19),
            encoder_name=model_config.get('encoder_name', 'resnet50'),
            pretrained=model_config.get('pretrained', True),
            num_scales=model_config.get('num_scales', 4),
            use_scale_attention=model_config.get('use_scale_attention', True),
            use_aux_head=model_config.get('use_aux_head', True),
        )
        model = model.to(device)

        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(
            f"Model parameters: {total_params:,} total, {trainable_params:,} trainable"
        )

        # Create loss function
        logger.info("Creating loss function...")
        loss_config = config.get('loss', {})
        criterion = MultiScaleLoss(
            num_classes=model_config.get('num_classes', 19),
            ignore_index=loss_config.get('ignore_index', 255),
            aux_weight=loss_config.get('aux_weight', 0.4),
            scale_weight=loss_config.get('scale_weight', 0.2),
        )

        # Create optimizer and scheduler
        optimizer = create_optimizer(model, config)
        scheduler = create_scheduler(optimizer, config)

        # Create trainer
        logger.info("Creating trainer...")
        trainer = SegmentationTrainer(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            config=config,
        )

        # Load checkpoint if provided
        if args.checkpoint:
            logger.info(f"Loading checkpoint from {args.checkpoint}")
            trainer.load_checkpoint(args.checkpoint)

        # Initialize MLflow (if enabled)
        if config.get('use_mlflow', False):
            try:
                import mlflow
                mlflow.start_run()
                logger.info("Started MLflow run")
            except Exception as e:
                logger.warning(f"Failed to start MLflow: {e}")

        # Train model
        logger.info("Starting training...")
        history = trainer.train(train_loader, val_loader)

        # Save training history
        import json
        history_path = output_dir / 'training_history.json'
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)
        logger.info(f"Saved training history to {history_path}")

        # Create visualizations
        logger.info("Creating training visualizations...")
        analyzer = ResultsAnalyzer(
            num_classes=model_config.get('num_classes', 19),
            output_dir=output_dir,
        )
        analyzer.plot_training_curves(history)

        # End MLflow run
        if config.get('use_mlflow', False):
            try:
                import mlflow
                mlflow.end_run()
            except Exception as e:
                logger.warning(f"Failed to end MLflow run: {e}")

        logger.info("Training completed successfully!")
        logger.info(f"Results saved to {output_dir}")
        logger.info(f"Best model saved to {trainer.checkpoint_dir / 'best_model.pth'}")

    except Exception as e:
        logger.error(f"Training failed with error: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
