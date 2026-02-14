"""Adaptive Scale Fusion for Urban Semantic Segmentation.

A multi-scale semantic segmentation framework with learned scale-attention
mechanisms for urban scene understanding.
"""

__version__ = "0.1.0"
__author__ = "Alireza Shojaei"

from adaptive_scale_fusion_urban_segmentation.models.model import AdaptiveScaleFusionSegmenter
from adaptive_scale_fusion_urban_segmentation.training.trainer import SegmentationTrainer

__all__ = ["AdaptiveScaleFusionSegmenter", "SegmentationTrainer"]
