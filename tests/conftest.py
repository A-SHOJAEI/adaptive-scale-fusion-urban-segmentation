"""Pytest configuration and fixtures."""

import pytest
import torch
import numpy as np


@pytest.fixture
def device():
    """Get device for testing."""
    return torch.device('cpu')


@pytest.fixture
def batch_size():
    """Standard batch size for testing."""
    return 2


@pytest.fixture
def num_classes():
    """Number of segmentation classes."""
    return 19


@pytest.fixture
def image_size():
    """Standard image size for testing."""
    return (256, 512)  # (height, width)


@pytest.fixture
def sample_batch(batch_size, image_size):
    """Create a sample batch of images and masks.

    Args:
        batch_size: Number of samples in batch.
        image_size: Image dimensions (height, width).

    Returns:
        Dictionary with 'image' and 'mask' tensors.
    """
    images = torch.randn(batch_size, 3, image_size[0], image_size[1])
    masks = torch.randint(0, 19, (batch_size, image_size[0], image_size[1]))

    return {'image': images, 'mask': masks}


@pytest.fixture
def config(image_size, num_classes):
    """Standard configuration for testing."""
    return {
        'seed': 42,
        'data_root': './data/cityscapes',
        'num_synthetic_samples': 10,
        'batch_size': 2,
        'num_workers': 0,
        'image_size': list(image_size),
        'model': {
            'encoder_name': 'resnet18',  # Smaller for faster testing
            'pretrained': False,
            'num_classes': num_classes,
            'num_scales': 4,
            'use_scale_attention': True,
            'use_aux_head': True,
        },
        'loss': {
            'aux_weight': 0.4,
            'scale_weight': 0.2,
            'ignore_index': 255,
        },
        'epochs': 2,
        'learning_rate': 0.001,
        'weight_decay': 0.0001,
        'optimizer': 'adamw',
        'grad_clip': 1.0,
        'use_amp': False,
        'scheduler': {
            'type': 'cosine',
            'warmup_epochs': 1,
            'min_lr': 0.00001,
        },
        'patience': 5,
        'min_delta': 0.001,
        'checkpoint_dir': './test_checkpoints',
        'log_interval': 1,
        'use_mlflow': False,
        'device': 'cpu',
    }
