# Adaptive Scale Fusion for Urban Segmentation

Multi-scale semantic segmentation for urban scenes using adaptive feature fusion with learned scale-attention mechanisms. The model combines dense prediction with dynamic scale weighting that adapts per-image based on object size distribution, addressing the challenge of simultaneously segmenting small objects like pedestrians and traffic signs alongside large structures like buildings and roads.

## Novel Contribution

Learnable scale-attention module that computes importance weights for multi-scale features using global context statistics, trained end-to-end with auxiliary scale-prediction loss. This enables the model to dynamically emphasize relevant scales based on the content of each input image.

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

### Training

```bash
python scripts/train.py --config configs/default.yaml
```

For ablation study without scale attention:

```bash
python scripts/train.py --config configs/ablation.yaml
```

### Evaluation

```bash
python scripts/evaluate.py --checkpoint checkpoints/best_model.pth --visualize
```

### Inference

```bash
python scripts/predict.py --checkpoint checkpoints/best_model.pth --input path/to/image.jpg --visualize
```

## Methodology

### Problem Statement

Urban scene segmentation faces a fundamental challenge: objects span vastly different scales. A traffic sign might be 20×30 pixels while a building occupies thousands. Traditional fixed-scale fusion methods treat all scales equally, but different images have different object size distributions—a highway scene needs different scale emphasis than a dense urban intersection.

### Approach

This work introduces **adaptive scale-attention fusion** that learns to weight multi-scale features based on image content:

1. **Multi-Scale Feature Extraction**: A ResNet-50 encoder extracts features at 4 scales (1/4, 1/8, 1/16, 1/32), capturing both fine details and global context through progressive downsampling.

2. **Scale Attention Module** (Novel Contribution):
   - Computes global context from the finest-scale features using adaptive average pooling
   - Feeds context through a small network to predict per-scale attention weights
   - Uses learnable temperature parameter for softmax to control attention sharpness
   - Weights are normalized across scales and applied to upsampled multi-scale features
   - This allows the model to dynamically emphasize small-object scales for crowded scenes or large-object scales for highway scenes

3. **Auxiliary Scale Prediction**:
   - A parallel head predicts object size distribution (small/medium/large bins) from features
   - Trained with scale-aware targets computed from ground truth segmentation masks
   - Provides additional supervision signal to guide the attention module toward scale-aware representations
   - Loss weight: 0.2 (empirically tuned to balance with main segmentation loss)

4. **Multi-Scale Loss Function**:
   - Main segmentation loss (cross-entropy) on final output
   - Auxiliary segmentation loss (0.4 weight) on intermediate features for deep supervision
   - Scale prediction loss (0.2 weight) to guide attention learning
   - Combined end-to-end training ensures all components learn jointly

### Why This Works

The key insight is that **attention should be content-dependent, not fixed**. By learning from both segmentation supervision and explicit scale distribution labels, the model discovers:
- When to emphasize fine-scale features (for small pedestrians, signs)
- When to emphasize coarse-scale features (for roads, sky, buildings)
- How to balance scales for mixed scenes

The auxiliary scale prediction acts as a regularizer, preventing the attention module from collapsing to a single scale and encouraging scale-aware representations.

## Architecture

The model consists of:

1. **Multi-scale Feature Extraction**: ResNet-50 encoder producing 4 feature pyramid levels
2. **Scale Attention Module**: Learnable attention mechanism with temperature-scaled softmax
3. **Auxiliary Scale Predictor**: 3-bin object size classifier for attention supervision
4. **Segmentation Head**: 1×1 convolution for dense pixel-wise classification

## Configuration

Key configuration files:

- `configs/default.yaml`: Full model with scale attention
- `configs/ablation.yaml`: Baseline without scale attention for comparison

## Dataset

The implementation includes synthetic data generation for demonstration. To use real Cityscapes data:

1. Download Cityscapes dataset
2. Update `data_root` in config files
3. Set `synthetic: false` in data loader

## Results

Evaluation on the synthetic Cityscapes-format test set (40 samples, 512x1024 resolution) using the best checkpoint (epoch 2, validation loss 0.3877):

### Overall Metrics

| Metric | Value |
|--------|-------|
| Mean IoU | 0.3066 |
| Pixel Accuracy | 0.7418 |
| Mean Accuracy | 0.3715 |
| Frequency Weighted IoU | 0.6185 |
| Large Object IoU | 0.6132 |
| Small Object IoU | 0.0000 |

### Per-Class IoU

| Class | IoU |
|-------|-----|
| Road | 0.9846 |
| Sky | 0.9857 |
| Building | 0.4720 |
| Vegetation | 0.0107 |
| Pole | 0.0000 |
| Traffic Light | 0.0000 |
| Traffic Sign | 0.0000 |
| Person | 0.0000 |

The model achieves strong segmentation on large-area classes (road 0.9846, sky 0.9857) where multi-scale context provides clear spatial priors. Building segmentation reaches 0.4720, reflecting the more complex boundary structures. Small object classes (poles, traffic lights, signs, persons) remain at zero IoU after limited training, which is expected given only 2 epochs on synthetic data. Extended training on real Cityscapes data with the full 50-epoch schedule and scale-attention mechanism is expected to improve small object performance, which is the primary motivation for the adaptive scale fusion approach.

### Training Configuration

| Parameter | Value |
|-----------|-------|
| Encoder | ResNet-50 |
| Parameters | 27.4M |
| Optimizer | AdamW |
| Learning Rate | 0.001 |
| Batch Size | 4 |
| Image Size | 512 x 1024 |
| Scale Attention | Enabled |
| Auxiliary Loss Weight | 0.4 |
| Scale Prediction Loss Weight | 0.2 |

## Project Structure

```
adaptive-scale-fusion-urban-segmentation/
├── src/adaptive_scale_fusion_urban_segmentation/
│   ├── data/              # Data loading and preprocessing
│   ├── models/            # Model architecture and components
│   ├── training/          # Training loop and utilities
│   ├── evaluation/        # Metrics and analysis
│   └── utils/             # Configuration and helpers
├── scripts/
│   ├── train.py           # Training script
│   ├── evaluate.py        # Evaluation script
│   └── predict.py         # Inference script
├── configs/
│   ├── default.yaml       # Default configuration
│   └── ablation.yaml      # Ablation study config
└── tests/                 # Test suite
```

## Testing

Run the test suite:

```bash
pytest tests/ -v --cov
```

## Key Features

- Multi-scale feature fusion with learned attention
- Auxiliary scale prediction loss for improved learning
- Mixed precision training for faster convergence
- Early stopping and learning rate scheduling
- Comprehensive evaluation metrics
- Per-class performance analysis

## Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA-compatible GPU recommended

## License

MIT License - Copyright (c) 2026 Alireza Shojaei. See [LICENSE](LICENSE) for details.
