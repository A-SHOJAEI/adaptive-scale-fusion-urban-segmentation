"""Configuration utilities for loading and managing experiment settings."""

import random
import logging
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import torch
import yaml


logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file.

    Args:
        config_path: Path to YAML configuration file.

    Returns:
        Dictionary containing configuration parameters.

    Raises:
        FileNotFoundError: If config file does not exist.
        yaml.YAMLError: If config file is malformed.
    """
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    try:
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"Loaded configuration from {config_path}")
        return config
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML config: {e}")
        raise


def set_random_seeds(seed: int = 42) -> None:
    """Set random seeds for reproducibility across all libraries.

    Args:
        seed: Random seed value to use.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Make cudnn deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    logger.info(f"Set random seeds to {seed}")


def save_config(config: Dict[str, Any], save_path: str) -> None:
    """Save configuration to YAML file.

    Args:
        config: Configuration dictionary to save.
        save_path: Path where to save the configuration.
    """
    save_file = Path(save_path)
    save_file.parent.mkdir(parents=True, exist_ok=True)

    with open(save_file, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    logger.info(f"Saved configuration to {save_path}")


def get_device(config: Optional[Dict[str, Any]] = None) -> torch.device:
    """Get the device to use for training/inference.

    Args:
        config: Optional configuration dictionary with 'device' key.

    Returns:
        PyTorch device object.
    """
    if config and 'device' in config:
        device_str = config['device']
    else:
        device_str = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Fall back to CPU if CUDA is requested but not available
    if device_str == 'cuda' and not torch.cuda.is_available():
        logger.warning("CUDA requested but not available, falling back to CPU")
        device_str = 'cpu'

    device = torch.device(device_str)
    logger.info(f"Using device: {device}")

    return device
