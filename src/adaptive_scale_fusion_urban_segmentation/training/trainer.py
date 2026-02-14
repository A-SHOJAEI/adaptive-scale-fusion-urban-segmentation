"""Training loop with learning rate scheduling, early stopping, and checkpointing."""

import logging
import time
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
import numpy as np


logger = logging.getLogger(__name__)


class SegmentationTrainer:
    """Trainer for semantic segmentation with advanced training features."""

    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
        device: torch.device,
        config: Dict[str, Any],
    ):
        """Initialize segmentation trainer.

        Args:
            model: Segmentation model to train.
            criterion: Loss function.
            optimizer: Optimizer.
            scheduler: Learning rate scheduler.
            device: Device to train on.
            config: Configuration dictionary.
        """
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.config = config

        # Training configuration
        self.epochs = config.get('epochs', 100)
        self.grad_clip = config.get('grad_clip', 1.0)
        self.use_amp = config.get('use_amp', True)
        self.log_interval = config.get('log_interval', 10)

        # Early stopping
        self.patience = config.get('patience', 10)
        self.min_delta = config.get('min_delta', 0.001)
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.best_epoch = 0

        # Checkpointing
        self.checkpoint_dir = Path(config.get('checkpoint_dir', './checkpoints'))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Mixed precision training
        self.scaler = GradScaler('cuda') if self.use_amp else None

        # Metrics tracking
        self.train_losses = []
        self.val_losses = []
        self.learning_rates = []

        # MLflow tracking (optional)
        self.use_mlflow = config.get('use_mlflow', False)
        if self.use_mlflow:
            try:
                import mlflow
                self.mlflow = mlflow
                self.mlflow_enabled = True
                logger.info("MLflow tracking enabled")
            except ImportError:
                logger.warning("MLflow not available, disabling tracking")
                self.mlflow_enabled = False
        else:
            self.mlflow_enabled = False

        logger.info(f"Initialized trainer: epochs={self.epochs}, patience={self.patience}")

    def train_epoch(
        self,
        train_loader: DataLoader,
        epoch: int,
    ) -> Tuple[float, Dict[str, float]]:
        """Train for one epoch.

        Args:
            train_loader: Training data loader.
            epoch: Current epoch number.

        Returns:
            Tuple of (average_loss, metrics_dict).
        """
        self.model.train()
        epoch_losses = []
        epoch_metrics = {'seg_loss': 0.0, 'aux_loss': 0.0, 'scale_loss': 0.0}

        start_time = time.time()

        for batch_idx, batch in enumerate(train_loader):
            images = batch['image'].to(self.device)
            masks = batch['mask'].to(self.device)

            # Forward pass with mixed precision
            if self.use_amp:
                with autocast('cuda'):
                    outputs = self.model(images, return_aux=True)
                    loss, loss_dict = self.criterion(outputs, masks)
            else:
                outputs = self.model(images, return_aux=True)
                loss, loss_dict = self.criterion(outputs, masks)

            # Backward pass
            self.optimizer.zero_grad()

            if self.use_amp:
                self.scaler.scale(loss).backward()
                if self.grad_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.grad_clip,
                    )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                if self.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.grad_clip,
                    )
                self.optimizer.step()

            # Track metrics
            epoch_losses.append(loss.item())
            for key, value in loss_dict.items():
                if key in epoch_metrics:
                    epoch_metrics[key] += value

            # Logging
            if (batch_idx + 1) % self.log_interval == 0:
                avg_loss = np.mean(epoch_losses[-self.log_interval:])
                logger.info(
                    f"Epoch [{epoch}/{self.epochs}] "
                    f"Batch [{batch_idx + 1}/{len(train_loader)}] "
                    f"Loss: {avg_loss:.4f}"
                )

        # Average metrics over epoch
        avg_loss = np.mean(epoch_losses)
        for key in epoch_metrics:
            epoch_metrics[key] /= len(train_loader)

        epoch_time = time.time() - start_time
        logger.info(
            f"Epoch [{epoch}/{self.epochs}] completed in {epoch_time:.2f}s - "
            f"Train Loss: {avg_loss:.4f}"
        )

        return avg_loss, epoch_metrics

    @torch.no_grad()
    def validate(
        self,
        val_loader: DataLoader,
        epoch: int,
    ) -> Tuple[float, Dict[str, float]]:
        """Validate the model.

        Args:
            val_loader: Validation data loader.
            epoch: Current epoch number.

        Returns:
            Tuple of (average_loss, metrics_dict).
        """
        self.model.eval()
        val_losses = []
        val_metrics = {'seg_loss': 0.0, 'aux_loss': 0.0, 'scale_loss': 0.0}

        for batch in val_loader:
            images = batch['image'].to(self.device)
            masks = batch['mask'].to(self.device)

            if self.use_amp:
                with autocast('cuda'):
                    outputs = self.model(images, return_aux=True)
                    loss, loss_dict = self.criterion(outputs, masks)
            else:
                outputs = self.model(images, return_aux=True)
                loss, loss_dict = self.criterion(outputs, masks)

            val_losses.append(loss.item())
            for key, value in loss_dict.items():
                if key in val_metrics:
                    val_metrics[key] += value

        avg_loss = np.mean(val_losses)
        for key in val_metrics:
            val_metrics[key] /= len(val_loader)

        logger.info(f"Epoch [{epoch}/{self.epochs}] - Val Loss: {avg_loss:.4f}")

        return avg_loss, val_metrics

    def check_early_stopping(self, val_loss: float, epoch: int) -> bool:
        """Check if early stopping criteria is met.

        Args:
            val_loss: Current validation loss.
            epoch: Current epoch number.

        Returns:
            True if training should stop, False otherwise.
        """
        if val_loss < self.best_val_loss - self.min_delta:
            self.best_val_loss = val_loss
            self.best_epoch = epoch
            self.patience_counter = 0
            return False
        else:
            self.patience_counter += 1
            logger.info(
                f"Early stopping counter: {self.patience_counter}/{self.patience}"
            )
            return self.patience_counter >= self.patience

    def save_checkpoint(
        self,
        epoch: int,
        val_loss: float,
        is_best: bool = False,
    ) -> None:
        """Save model checkpoint.

        Args:
            epoch: Current epoch number.
            val_loss: Validation loss.
            is_best: Whether this is the best model so far.
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
            'config': self.config,
        }

        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()

        # Save latest checkpoint
        checkpoint_path = self.checkpoint_dir / 'latest_checkpoint.pth'
        torch.save(checkpoint, checkpoint_path)

        # Save best model
        if is_best:
            best_path = self.checkpoint_dir / 'best_model.pth'
            torch.save(checkpoint, best_path)
            logger.info(f"Saved best model at epoch {epoch} with val_loss={val_loss:.4f}")

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
    ) -> Dict[str, Any]:
        """Full training loop with early stopping and checkpointing.

        Args:
            train_loader: Training data loader.
            val_loader: Validation data loader.

        Returns:
            Training history dictionary.
        """
        logger.info("Starting training...")

        if self.mlflow_enabled:
            try:
                self.mlflow.log_params({
                    'epochs': self.epochs,
                    'batch_size': self.config.get('batch_size'),
                    'learning_rate': self.config.get('learning_rate'),
                    'optimizer': self.config.get('optimizer'),
                })
            except Exception as e:
                logger.warning(f"MLflow logging failed: {e}")

        for epoch in range(1, self.epochs + 1):
            # Training phase
            train_loss, train_metrics = self.train_epoch(train_loader, epoch)
            self.train_losses.append(train_loss)

            # Validation phase
            val_loss, val_metrics = self.validate(val_loader, epoch)
            self.val_losses.append(val_loss)

            # Learning rate scheduling
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

            current_lr = self.optimizer.param_groups[0]['lr']
            self.learning_rates.append(current_lr)

            # MLflow logging
            if self.mlflow_enabled:
                try:
                    self.mlflow.log_metrics({
                        'train_loss': train_loss,
                        'val_loss': val_loss,
                        'learning_rate': current_lr,
                        **{f'train_{k}': v for k, v in train_metrics.items()},
                        **{f'val_{k}': v for k, v in val_metrics.items()},
                    }, step=epoch)
                except Exception as e:
                    logger.warning(f"MLflow logging failed: {e}")

            # Checkpointing
            is_best = val_loss < self.best_val_loss
            self.save_checkpoint(epoch, val_loss, is_best)

            # Early stopping check
            if self.check_early_stopping(val_loss, epoch):
                logger.info(
                    f"Early stopping triggered at epoch {epoch}. "
                    f"Best epoch was {self.best_epoch} with val_loss={self.best_val_loss:.4f}"
                )
                break

        logger.info("Training completed!")

        history = {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'learning_rates': self.learning_rates,
            'best_epoch': self.best_epoch,
            'best_val_loss': self.best_val_loss,
        }

        return history

    def load_checkpoint(self, checkpoint_path: str) -> None:
        """Load model from checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file.
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        logger.info(f"Loaded checkpoint from {checkpoint_path}")
