"""Tests for data loading and preprocessing."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest
import torch
import numpy as np

from adaptive_scale_fusion_urban_segmentation.data.loader import (
    CityscapesDataset,
    get_dataloaders,
)
from adaptive_scale_fusion_urban_segmentation.data.preprocessing import (
    get_train_transforms,
    get_val_transforms,
    MixUp,
)


class TestCityscapesDataset:
    """Tests for Cityscapes dataset."""

    def test_dataset_initialization(self):
        """Test dataset can be initialized."""
        dataset = CityscapesDataset(
            root='./data/cityscapes',
            split='train',
            synthetic=True,
            num_samples=10,
        )
        assert len(dataset) == 10
        assert dataset.num_classes == 19

    def test_dataset_getitem(self):
        """Test dataset __getitem__ returns correct format."""
        dataset = CityscapesDataset(
            root='./data/cityscapes',
            split='train',
            synthetic=True,
            num_samples=5,
        )

        sample = dataset[0]
        assert 'image' in sample
        assert 'mask' in sample
        assert isinstance(sample['image'], torch.Tensor)
        assert isinstance(sample['mask'], torch.Tensor)

    def test_dataset_synthetic_generation(self):
        """Test synthetic data generation is deterministic."""
        dataset = CityscapesDataset(
            root='./data/cityscapes',
            split='train',
            synthetic=True,
            num_samples=5,
        )

        sample1 = dataset[0]
        sample2 = dataset[0]

        # Same index should give same result
        assert torch.equal(sample1['mask'], sample2['mask'])

    def test_dataset_class_definitions(self):
        """Test dataset has correct class definitions."""
        dataset = CityscapesDataset(
            root='./data/cityscapes',
            split='train',
            synthetic=True,
        )

        assert len(dataset.CLASSES) == 19
        assert 'road' in dataset.CLASSES
        assert 'person' in dataset.CLASSES
        assert len(dataset.SMALL_CLASSES) > 0
        assert len(dataset.LARGE_CLASSES) > 0


class TestDataTransforms:
    """Tests for data transforms."""

    def test_train_transforms(self, image_size):
        """Test training transforms."""
        transforms = get_train_transforms(image_size=image_size)

        image = np.random.randint(0, 256, (512, 1024, 3), dtype=np.uint8)
        mask = np.random.randint(0, 19, (512, 1024), dtype=np.int64)

        transformed = transforms(image=image, mask=mask)

        assert 'image' in transformed
        assert 'mask' in transformed
        assert transformed['image'].shape == (3, image_size[0], image_size[1])
        assert transformed['mask'].shape == (image_size[0], image_size[1])

    def test_val_transforms(self, image_size):
        """Test validation transforms."""
        transforms = get_val_transforms(image_size=image_size)

        image = np.random.randint(0, 256, (512, 1024, 3), dtype=np.uint8)
        mask = np.random.randint(0, 19, (512, 1024), dtype=np.int64)

        transformed = transforms(image=image, mask=mask)

        assert transformed['image'].shape == (3, image_size[0], image_size[1])
        assert transformed['mask'].shape == (image_size[0], image_size[1])

    def test_mixup(self):
        """Test MixUp augmentation."""
        mixup = MixUp(alpha=0.2)

        image1 = np.random.rand(256, 512, 3).astype(np.float32)
        mask1 = np.random.randint(0, 19, (256, 512))
        image2 = np.random.rand(256, 512, 3).astype(np.float32)
        mask2 = np.random.randint(0, 19, (256, 512))

        mixed_image, mixed_mask = mixup(image1, mask1, image2, mask2)

        assert mixed_image.shape == image1.shape
        assert mixed_mask.shape == mask1.shape


class TestDataLoaders:
    """Tests for data loaders."""

    def test_get_dataloaders(self, config):
        """Test dataloader creation."""
        train_loader, val_loader, test_loader = get_dataloaders(
            config,
            synthetic=True,
        )

        assert len(train_loader) > 0
        assert len(val_loader) > 0
        assert len(test_loader) > 0

    def test_dataloader_batch_format(self, config):
        """Test dataloader returns correct batch format."""
        train_loader, _, _ = get_dataloaders(config, synthetic=True)

        batch = next(iter(train_loader))

        assert 'image' in batch
        assert 'mask' in batch
        assert batch['image'].shape[0] == config['batch_size']
        assert batch['mask'].shape[0] == config['batch_size']

    def test_dataloader_data_types(self, config):
        """Test dataloader returns correct data types."""
        train_loader, _, _ = get_dataloaders(config, synthetic=True)

        batch = next(iter(train_loader))

        assert batch['image'].dtype == torch.float32
        assert batch['mask'].dtype == torch.int64
