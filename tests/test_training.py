"""Tests for training pipeline."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from adaptive_scale_fusion_urban_segmentation.training.trainer import SegmentationTrainer
from adaptive_scale_fusion_urban_segmentation.models.model import AdaptiveScaleFusionSegmenter
from adaptive_scale_fusion_urban_segmentation.models.components import MultiScaleLoss
from adaptive_scale_fusion_urban_segmentation.evaluation.metrics import (
    SegmentationMetrics,
    compute_iou,
    compute_dice,
)


class TestSegmentationTrainer:
    """Tests for segmentation trainer."""

    @pytest.fixture
    def simple_model(self, num_classes):
        """Create a simple model for testing."""
        return AdaptiveScaleFusionSegmenter(
            num_classes=num_classes,
            encoder_name='resnet18',
            pretrained=False,
            use_scale_attention=True,
        )

    @pytest.fixture
    def criterion(self, num_classes):
        """Create loss function."""
        return MultiScaleLoss(
            num_classes=num_classes,
            aux_weight=0.4,
            scale_weight=0.2,
        )

    @pytest.fixture
    def optimizer(self, simple_model):
        """Create optimizer."""
        return torch.optim.AdamW(simple_model.parameters(), lr=0.001)

    @pytest.fixture
    def scheduler(self, optimizer):
        """Create learning rate scheduler."""
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

    @pytest.fixture
    def dummy_dataloader(self, batch_size, num_classes, image_size):
        """Create a dummy dataloader for testing."""
        images = torch.randn(10, 3, image_size[0], image_size[1])
        masks = torch.randint(0, num_classes, (10, image_size[0], image_size[1]))

        dataset = TensorDataset(images, masks)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        # Convert to dict format
        class DictDataLoader:
            def __init__(self, base_loader):
                self.base_loader = base_loader

            def __iter__(self):
                for images, masks in self.base_loader:
                    yield {'image': images, 'mask': masks}

            def __len__(self):
                return len(self.base_loader)

        return DictDataLoader(loader)

    def test_trainer_initialization(
        self,
        simple_model,
        criterion,
        optimizer,
        scheduler,
        device,
        config,
    ):
        """Test trainer initialization."""
        trainer = SegmentationTrainer(
            model=simple_model,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            config=config,
        )
        assert trainer is not None

    def test_train_epoch(
        self,
        simple_model,
        criterion,
        optimizer,
        scheduler,
        device,
        config,
        dummy_dataloader,
    ):
        """Test training for one epoch."""
        trainer = SegmentationTrainer(
            model=simple_model,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            config=config,
        )

        avg_loss, metrics = trainer.train_epoch(dummy_dataloader, epoch=1)

        assert isinstance(avg_loss, float)
        assert avg_loss >= 0
        assert 'seg_loss' in metrics

    def test_validate(
        self,
        simple_model,
        criterion,
        optimizer,
        scheduler,
        device,
        config,
        dummy_dataloader,
    ):
        """Test validation."""
        trainer = SegmentationTrainer(
            model=simple_model,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            config=config,
        )

        avg_loss, metrics = trainer.validate(dummy_dataloader, epoch=1)

        assert isinstance(avg_loss, float)
        assert avg_loss >= 0

    def test_early_stopping(
        self,
        simple_model,
        criterion,
        optimizer,
        scheduler,
        device,
        config,
    ):
        """Test early stopping mechanism."""
        trainer = SegmentationTrainer(
            model=simple_model,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            config=config,
        )

        # Should not stop on improvement
        should_stop = trainer.check_early_stopping(val_loss=1.0, epoch=1)
        assert not should_stop

        # Should stop after patience epochs without improvement
        for i in range(config['patience'] + 1):
            should_stop = trainer.check_early_stopping(val_loss=2.0, epoch=i + 2)

        assert should_stop


class TestSegmentationMetrics:
    """Tests for segmentation metrics."""

    def test_metrics_initialization(self, num_classes):
        """Test metrics initialization."""
        metrics = SegmentationMetrics(
            num_classes=num_classes,
        )
        assert metrics is not None

    def test_metrics_update(self, num_classes, batch_size, image_size):
        """Test metrics update."""
        metrics = SegmentationMetrics(num_classes=num_classes)

        pred = torch.randint(0, num_classes, (batch_size, image_size[0], image_size[1]))
        target = torch.randint(0, num_classes, (batch_size, image_size[0], image_size[1]))

        metrics.update(pred, target)
        assert metrics.total_samples > 0

    def test_metrics_compute(self, num_classes, batch_size, image_size):
        """Test metrics computation."""
        metrics = SegmentationMetrics(num_classes=num_classes)

        pred = torch.randint(0, num_classes, (batch_size, image_size[0], image_size[1]))
        target = torch.randint(0, num_classes, (batch_size, image_size[0], image_size[1]))

        metrics.update(pred, target)
        results = metrics.compute_metrics()

        assert 'mean_iou' in results
        assert 'pixel_accuracy' in results
        assert 'mean_accuracy' in results

    def test_perfect_prediction(self, num_classes, batch_size, image_size):
        """Test metrics with perfect predictions."""
        metrics = SegmentationMetrics(num_classes=num_classes)

        pred = torch.randint(0, num_classes, (batch_size, image_size[0], image_size[1]))
        target = pred.clone()

        metrics.update(pred, target)
        results = metrics.compute_metrics()

        assert results['pixel_accuracy'] == 1.0
        assert results['mean_iou'] == 1.0

    def test_metrics_with_ignore_index(self, num_classes, batch_size, image_size):
        """Test metrics with ignore index."""
        metrics = SegmentationMetrics(
            num_classes=num_classes,
            ignore_index=255,
        )

        pred = torch.randint(0, num_classes, (batch_size, image_size[0], image_size[1]))
        target = torch.randint(0, num_classes, (batch_size, image_size[0], image_size[1]))

        # Set some pixels to ignore index
        target[0, :10, :10] = 255

        metrics.update(pred, target)
        results = metrics.compute_metrics()

        assert 'mean_iou' in results


class TestMetricFunctions:
    """Tests for metric functions."""

    def test_compute_iou(self, num_classes, batch_size, image_size):
        """Test IoU computation."""
        pred = torch.randint(0, num_classes, (batch_size, image_size[0], image_size[1]))
        target = torch.randint(0, num_classes, (batch_size, image_size[0], image_size[1]))

        ious = compute_iou(pred, target, num_classes)

        assert ious.shape == (num_classes,)

    def test_compute_dice(self, num_classes, batch_size, image_size):
        """Test Dice coefficient computation."""
        pred = torch.randint(0, num_classes, (batch_size, image_size[0], image_size[1]))
        target = torch.randint(0, num_classes, (batch_size, image_size[0], image_size[1]))

        dice_scores = compute_dice(pred, target, num_classes)

        assert dice_scores.shape == (num_classes,)

    def test_iou_perfect_match(self, num_classes):
        """Test IoU with perfect match."""
        pred = torch.zeros((1, 100, 100), dtype=torch.long)
        target = torch.zeros((1, 100, 100), dtype=torch.long)

        ious = compute_iou(pred, target, num_classes)

        # Class 0 should have IoU of 1.0
        assert ious[0] == 1.0

    def test_iou_no_overlap(self, num_classes):
        """Test IoU with no overlap."""
        pred = torch.zeros((1, 100, 100), dtype=torch.long)
        target = torch.ones((1, 100, 100), dtype=torch.long)

        ious = compute_iou(pred, target, num_classes)

        # Class 0 and 1 should have IoU of 0.0
        assert ious[0] == 0.0
        assert ious[1] == 0.0
