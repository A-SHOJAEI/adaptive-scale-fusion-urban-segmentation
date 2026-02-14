"""Tests for model architecture and components."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest
import torch

from adaptive_scale_fusion_urban_segmentation.models.model import AdaptiveScaleFusionSegmenter
from adaptive_scale_fusion_urban_segmentation.models.components import (
    ScaleAttentionModule,
    MultiScaleLoss,
    AuxiliaryScalePredictionHead,
    FocalLoss,
)


class TestAdaptiveScaleFusionSegmenter:
    """Tests for the main segmentation model."""

    def test_model_initialization(self, num_classes):
        """Test model can be initialized."""
        model = AdaptiveScaleFusionSegmenter(
            num_classes=num_classes,
            encoder_name='resnet18',
            pretrained=False,
        )
        assert model is not None

    def test_model_forward_pass(self, num_classes, batch_size, image_size):
        """Test model forward pass."""
        model = AdaptiveScaleFusionSegmenter(
            num_classes=num_classes,
            encoder_name='resnet18',
            pretrained=False,
        )
        model.eval()

        images = torch.randn(batch_size, 3, image_size[0], image_size[1])

        with torch.no_grad():
            outputs = model(images, return_aux=False)

        assert 'main' in outputs
        assert outputs['main'].shape == (batch_size, num_classes, image_size[0], image_size[1])

    def test_model_forward_with_aux(self, num_classes, batch_size, image_size):
        """Test model forward pass with auxiliary outputs."""
        model = AdaptiveScaleFusionSegmenter(
            num_classes=num_classes,
            encoder_name='resnet18',
            pretrained=False,
            use_aux_head=True,
        )
        model.train()

        images = torch.randn(batch_size, 3, image_size[0], image_size[1])
        outputs = model(images, return_aux=True)

        assert 'main' in outputs
        assert 'aux' in outputs
        assert 'scale_pred' in outputs

    def test_model_without_scale_attention(self, num_classes, batch_size, image_size):
        """Test model without scale attention module."""
        model = AdaptiveScaleFusionSegmenter(
            num_classes=num_classes,
            encoder_name='resnet18',
            pretrained=False,
            use_scale_attention=False,
        )
        model.eval()

        images = torch.randn(batch_size, 3, image_size[0], image_size[1])

        with torch.no_grad():
            outputs = model(images, return_aux=False)

        assert outputs['main'].shape == (batch_size, num_classes, image_size[0], image_size[1])

    def test_get_attention_weights(self, num_classes, batch_size, image_size):
        """Test getting scale attention weights."""
        model = AdaptiveScaleFusionSegmenter(
            num_classes=num_classes,
            encoder_name='resnet18',
            pretrained=False,
            use_scale_attention=True,
        )
        model.eval()

        images = torch.randn(batch_size, 3, image_size[0], image_size[1])
        weights = model.get_attention_weights(images)

        assert weights is not None
        assert weights.shape[0] == batch_size
        assert weights.shape[1] == 4  # num_scales

    def test_model_parameter_count(self, num_classes):
        """Test model has reasonable number of parameters."""
        model = AdaptiveScaleFusionSegmenter(
            num_classes=num_classes,
            encoder_name='resnet18',
            pretrained=False,
        )

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        assert total_params > 0
        assert trainable_params > 0
        assert trainable_params == total_params


class TestScaleAttentionModule:
    """Tests for scale attention module."""

    def test_attention_initialization(self):
        """Test attention module initialization."""
        attention = ScaleAttentionModule(
            in_channels=256,
            num_scales=4,
        )
        assert attention is not None

    def test_attention_forward(self, batch_size):
        """Test attention module forward pass."""
        attention = ScaleAttentionModule(
            in_channels=256,
            num_scales=4,
        )

        features = [
            torch.randn(batch_size, 256, 128, 256),
            torch.randn(batch_size, 256, 64, 128),
            torch.randn(batch_size, 256, 32, 64),
            torch.randn(batch_size, 256, 16, 32),
        ]

        fused, weights = attention(features)

        assert fused.shape == (batch_size, 256, 128, 256)
        assert weights.shape == (batch_size, 4)

    def test_attention_weights_sum(self, batch_size):
        """Test attention weights sum to approximately 1."""
        attention = ScaleAttentionModule(
            in_channels=256,
            num_scales=4,
        )

        features = [
            torch.randn(batch_size, 256, 64, 128),
            torch.randn(batch_size, 256, 32, 64),
            torch.randn(batch_size, 256, 16, 32),
            torch.randn(batch_size, 256, 8, 16),
        ]

        _, weights = attention(features)
        weights_sum = weights.sum(dim=1)

        assert torch.allclose(weights_sum, torch.ones(batch_size), atol=1e-6)


class TestMultiScaleLoss:
    """Tests for multi-scale loss function."""

    def test_loss_initialization(self, num_classes):
        """Test loss function initialization."""
        criterion = MultiScaleLoss(
            num_classes=num_classes,
            aux_weight=0.4,
            scale_weight=0.2,
        )
        assert criterion is not None

    def test_loss_forward(self, num_classes, batch_size, image_size):
        """Test loss function forward pass."""
        criterion = MultiScaleLoss(
            num_classes=num_classes,
            aux_weight=0.4,
            scale_weight=0.2,
        )

        outputs = {
            'main': torch.randn(batch_size, num_classes, image_size[0], image_size[1]),
        }
        targets = torch.randint(0, num_classes, (batch_size, image_size[0], image_size[1]))

        loss, loss_dict = criterion(outputs, targets)

        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0  # Scalar
        assert 'seg_loss' in loss_dict
        assert 'total_loss' in loss_dict

    def test_loss_with_aux(self, num_classes, batch_size, image_size):
        """Test loss with auxiliary outputs."""
        criterion = MultiScaleLoss(
            num_classes=num_classes,
            aux_weight=0.4,
            scale_weight=0.2,
        )

        outputs = {
            'main': torch.randn(batch_size, num_classes, image_size[0], image_size[1]),
            'aux': torch.randn(batch_size, num_classes, image_size[0], image_size[1]),
            'scale_pred': torch.randn(batch_size, 3),
        }
        targets = torch.randint(0, num_classes, (batch_size, image_size[0], image_size[1]))

        loss, loss_dict = criterion(outputs, targets)

        assert 'aux_loss' in loss_dict
        assert 'scale_loss' in loss_dict


class TestAuxiliaryScalePredictionHead:
    """Tests for auxiliary scale prediction head."""

    def test_head_initialization(self):
        """Test head initialization."""
        head = AuxiliaryScalePredictionHead(
            in_channels=256,
            num_scale_bins=3,
        )
        assert head is not None

    def test_head_forward(self, batch_size):
        """Test head forward pass."""
        head = AuxiliaryScalePredictionHead(
            in_channels=256,
            num_scale_bins=3,
        )

        features = torch.randn(batch_size, 256, 64, 128)
        predictions = head(features)

        assert predictions.shape == (batch_size, 3)


class TestFocalLoss:
    """Tests for focal loss."""

    def test_focal_loss_initialization(self):
        """Test focal loss initialization."""
        focal_loss = FocalLoss(alpha=0.25, gamma=2.0)
        assert focal_loss is not None

    def test_focal_loss_forward(self, num_classes, batch_size, image_size):
        """Test focal loss forward pass."""
        focal_loss = FocalLoss(alpha=0.25, gamma=2.0)

        inputs = torch.randn(batch_size, num_classes, image_size[0], image_size[1])
        targets = torch.randint(0, num_classes, (batch_size, image_size[0], image_size[1]))

        loss = focal_loss(inputs, targets)

        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0
        assert loss.item() >= 0
