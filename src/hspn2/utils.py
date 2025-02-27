"""Utility functions.

Covers a few things like
- Data loading/processing
- Early stopping
- Config handling
- Visualization utils
"""

import logging
import os
from typing import Dict, List, Optional, cast

from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)


def load_config(config_path: str) -> DictConfig:
    """Load configuration from YAML.

    Args:
        config_path: Path to YAML config

    Returns:
        DictConfig: Loaded configuration

    Raises:
        FileNotFoundError: If the config file doesn't exist
        yaml.YAMLError: If the YAML is invalid
    """
    try:
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")

        config = OmegaConf.load(config_path)
        logger.info(f"Configuration loaded from {config_path}")
        assert isinstance(config, DictConfig)
        return config

    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        raise


def merge_configs(base_config: DictConfig, override_config: DictConfig) -> DictConfig:
    """Merge two configurations, with override taking precedence.

    Args:
        base_config: Base configuration
        override_config: Override configuration

    Returns:
        DictConfig: Merged configuration
    """
    return cast(DictConfig, OmegaConf.merge(base_config, override_config))


def create_directories(config: DictConfig) -> None:
    """Create necessary directories based on configuration.

    Args:
        config: Application configuration
    """
    dirs_to_create = [
        config.training.get("checkpoint_dir", "checkpoints"),
        config.get("log_dir", "logs"),
        config.get("output_dir", "output"),
    ]

    for directory in dirs_to_create:
        if directory:
            os.makedirs(directory, exist_ok=True)
            logger.debug(f"Created directory: {directory}")


def plot_training_history(history: Dict[str, List[float]], save_path: Optional[str] = None) -> None:
    """Plot training history.

    Args:
        history: Dictionary with training history
        save_path: Path to save the plot (optional)
    """
    try:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(10, 6))

        plt.plot(history["epochs"], history["train_loss"], label="Training Loss")
        plt.plot(history["epochs"], history["val_loss"], label="Validation Loss")

        # Add labels and title
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training History")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.7)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Training history plot saved to {save_path}")

        plt.show()

    except ImportError:
        logger.warning("Matplotlib not available. Cannot plot training history.")
    except Exception as e:
        logger.error(f"Error plotting training history: {e}")
