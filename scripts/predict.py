#!/usr/bin/env python
"""Inference script for adaptive scale fusion urban segmentation."""

import sys
import argparse
import logging
from pathlib import Path

# Add project root and src/ to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from adaptive_scale_fusion_urban_segmentation.utils.config import (
    load_config,
    get_device,
)
from adaptive_scale_fusion_urban_segmentation.models.model import AdaptiveScaleFusionSegmenter
from adaptive_scale_fusion_urban_segmentation.data.preprocessing import get_val_transforms
from adaptive_scale_fusion_urban_segmentation.data.loader import CityscapesDataset


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Run inference with adaptive scale fusion segmentation model'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to model checkpoint',
    )
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Path to input image or directory',
    )
    parser.add_argument(
        '--config',
        type=str,
        default='configs/default.yaml',
        help='Path to configuration file',
    )
    parser.add_argument(
        '--output',
        type=str,
        default='./predictions',
        help='Directory to save predictions',
    )
    parser.add_argument(
        '--visualize',
        action='store_true',
        help='Save visualization with color-coded predictions',
    )
    parser.add_argument(
        '--confidence',
        action='store_true',
        help='Show confidence scores',
    )

    return parser.parse_args()


def load_model(checkpoint_path: str, config: dict, device: torch.device):
    """Load model from checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file.
        config: Configuration dictionary.
        device: Device to load model on.

    Returns:
        Loaded model in evaluation mode.
    """
    model_config = config.get('model', {})

    model = AdaptiveScaleFusionSegmenter(
        num_classes=model_config.get('num_classes', 19),
        encoder_name=model_config.get('encoder_name', 'resnet50'),
        pretrained=False,
        num_scales=model_config.get('num_scales', 4),
        use_scale_attention=model_config.get('use_scale_attention', True),
        use_aux_head=model_config.get('use_aux_head', True),
    )

    checkpoint = torch.load(checkpoint_path, map_location=device)

    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model = model.to(device)
    model.eval()

    logger.info(f"Loaded model from {checkpoint_path}")
    return model


def preprocess_image(image_path: str, config: dict):
    """Load and preprocess image for inference.

    Args:
        image_path: Path to image file.
        config: Configuration dictionary.

    Returns:
        Preprocessed image tensor and original image.
    """
    # Load image
    image = Image.open(image_path).convert('RGB')
    original_size = image.size  # (width, height)

    # Convert to numpy
    image_np = np.array(image)

    # Get transforms
    image_size = config.get('image_size', [512, 1024])
    transforms = get_val_transforms(image_size=tuple(image_size))

    # Apply transforms
    transformed = transforms(image=image_np)
    image_tensor = transformed['image'].unsqueeze(0)  # Add batch dimension

    return image_tensor, image_np, original_size


@torch.no_grad()
def predict(
    model: torch.nn.Module,
    image_tensor: torch.Tensor,
    device: torch.device,
    return_confidence: bool = False,
):
    """Run inference on an image.

    Args:
        model: Segmentation model.
        image_tensor: Preprocessed image tensor (1, 3, H, W).
        device: Device to run inference on.
        return_confidence: Whether to return confidence scores.

    Returns:
        Predicted segmentation mask and optional confidence map.
    """
    image_tensor = image_tensor.to(device)

    # Forward pass
    outputs = model(image_tensor, return_aux=False)
    logits = outputs['main']  # (1, num_classes, H, W)

    # Get predictions
    if return_confidence:
        probs = torch.softmax(logits, dim=1)
        confidence, prediction = probs.max(dim=1)
        return prediction.cpu().numpy()[0], confidence.cpu().numpy()[0]
    else:
        prediction = logits.argmax(dim=1)
        return prediction.cpu().numpy()[0], None


def visualize_prediction(
    image: np.ndarray,
    prediction: np.ndarray,
    confidence: np.ndarray,
    output_path: str,
    class_names: list,
    show_confidence: bool = False,
):
    """Create visualization of prediction.

    Args:
        image: Original image (H, W, 3).
        prediction: Predicted segmentation mask (H, W).
        confidence: Confidence map (H, W) or None.
        output_path: Path to save visualization.
        class_names: List of class names.
        show_confidence: Whether to show confidence scores.
    """
    # Resize prediction to match input image
    from PIL import Image as PILImage

    pred_img = PILImage.fromarray(prediction.astype(np.uint8))
    pred_resized = pred_img.resize((image.shape[1], image.shape[0]), PILImage.NEAREST)
    prediction = np.array(pred_resized)

    if show_confidence and confidence is not None:
        conf_img = PILImage.fromarray((confidence * 255).astype(np.uint8))
        conf_resized = conf_img.resize((image.shape[1], image.shape[0]), PILImage.BILINEAR)
        confidence = np.array(conf_resized).astype(np.float32) / 255.0

    # Create figure
    if show_confidence and confidence is not None:
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    else:
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Original image
    axes[0].imshow(image)
    axes[0].set_title('Input Image')
    axes[0].axis('off')

    # Prediction
    axes[1].imshow(prediction, cmap='tab20', vmin=0, vmax=19)
    axes[1].set_title('Segmentation Prediction')
    axes[1].axis('off')

    # Confidence map
    if show_confidence and confidence is not None:
        im = axes[2].imshow(confidence, cmap='viridis', vmin=0, vmax=1)
        axes[2].set_title('Confidence Map')
        axes[2].axis('off')
        plt.colorbar(im, ax=axes[2])

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    logger.info(f"Saved visualization to {output_path}")


def save_prediction(
    prediction: np.ndarray,
    output_path: str,
    original_size: tuple,
):
    """Save prediction mask as PNG.

    Args:
        prediction: Predicted segmentation mask (H, W).
        output_path: Path to save prediction.
        original_size: Original image size (width, height).
    """
    from PIL import Image as PILImage

    # Resize to original size
    pred_img = PILImage.fromarray(prediction.astype(np.uint8))
    pred_resized = pred_img.resize(original_size, PILImage.NEAREST)

    # Save
    pred_resized.save(output_path)
    logger.info(f"Saved prediction mask to {output_path}")


def main():
    """Main inference function."""
    args = parse_args()

    # Load configuration
    logger.info(f"Loading configuration from {args.config}")
    config = load_config(args.config)

    # Get device
    device = get_device(config)

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Load model
        logger.info("Loading model...")
        model = load_model(args.checkpoint, config, device)

        # Process input
        input_path = Path(args.input)

        if input_path.is_file():
            # Single image
            image_paths = [input_path]
        elif input_path.is_dir():
            # Directory of images
            image_paths = list(input_path.glob('*.png')) + \
                         list(input_path.glob('*.jpg')) + \
                         list(input_path.glob('*.jpeg'))
            logger.info(f"Found {len(image_paths)} images in directory")
        else:
            raise ValueError(f"Input path not found: {input_path}")

        # Process each image
        for img_path in image_paths:
            logger.info(f"Processing {img_path.name}...")

            # Preprocess
            image_tensor, image_np, original_size = preprocess_image(
                str(img_path),
                config,
            )

            # Predict
            prediction, confidence = predict(
                model=model,
                image_tensor=image_tensor,
                device=device,
                return_confidence=args.confidence,
            )

            # Save outputs
            output_name = img_path.stem

            # Save prediction mask
            pred_path = output_dir / f"{output_name}_pred.png"
            save_prediction(prediction, str(pred_path), original_size)

            # Save visualization
            if args.visualize:
                vis_path = output_dir / f"{output_name}_vis.png"
                visualize_prediction(
                    image=image_np,
                    prediction=prediction,
                    confidence=confidence,
                    output_path=str(vis_path),
                    class_names=CityscapesDataset.CLASSES,
                    show_confidence=args.confidence,
                )

            # Print statistics
            unique_classes = np.unique(prediction)
            logger.info(f"  Detected {len(unique_classes)} classes")

            if args.confidence and confidence is not None:
                avg_conf = confidence.mean()
                logger.info(f"  Average confidence: {avg_conf:.3f}")

        logger.info(f"Predictions saved to {output_dir}")

    except Exception as e:
        logger.error(f"Prediction failed with error: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
