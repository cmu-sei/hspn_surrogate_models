"""Main entry point."""

import argparse
import logging
import sys

from omegaconf import DictConfig, OmegaConf
from train import Trainer
from utils import create_directories, load_config, plot_training_history
from src.hspn2 import setup_logging

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(description="DeepONet - Deep Operator Network Training")

    parser.add_argument(
        "-c", "--config", type=str, default="configs/default.yaml", help="Path to YAML configuration file"
    )

    # Process command-line argument overrides
    parser.add_argument("--train_path", type=str, help="Path to training data")

    parser.add_argument("--val_path", type=str, help="Path to validation data")

    parser.add_argument("--checkpoint_dir", type=str, help="Directory to save checkpoints")

    parser.add_argument(
        "--log_level", type=str, choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], help="Logging level"
    )

    parser.add_argument("--epochs", type=int, help="Number of training epochs")

    parser.add_argument("--batch_size", type=int, help="Training batch size")

    parser.add_argument("--learning_rate", type=float, help="Learning rate")

    parser.add_argument("--distributed", action="store_true", help="Enable distributed training with Horovod")

    parser.add_argument("--mixed_precision", action="store_true", help="Enable mixed precision training")

    parser.add_argument("--eval_only", action="store_true", help="Run evaluation only, no training")

    return parser.parse_args()


def prepare_config(args: argparse.Namespace) -> DictConfig:
    """Prepare config from YAML file and command-line overrides.

    Args:
        args: Parsed command-line arguments

    Returns:
        DictConfig: Complete configuration
    """
    config = load_config(args.config)  # base configuration w defaults
    override_dict = {}  # cli overrides

    if args.train_path:
        override_dict["data"] = override_dict.get("data", {})
        override_dict["data"]["train_path"] = args.train_path

    if args.val_path:
        override_dict["data"] = override_dict.get("data", {})
        override_dict["data"]["val_path"] = args.val_path

    if args.checkpoint_dir:
        override_dict["training"] = override_dict.get("training", {})
        override_dict["training"]["checkpoint_dir"] = args.checkpoint_dir

    if args.log_level:
        override_dict["log_level"] = args.log_level

    if args.epochs:
        override_dict["training"] = override_dict.get("training", {})
        override_dict["training"]["epochs"] = args.epochs

    if args.batch_size:
        override_dict["training"] = override_dict.get("training", {})
        override_dict["training"]["batch_size"] = args.batch_size

    if args.learning_rate:
        override_dict["training"] = override_dict.get("training", {})
        override_dict["training"]["learning_rate"] = args.learning_rate

    if args.distributed:
        override_dict["training"] = override_dict.get("training", {})
        override_dict["training"]["distributed"] = True

    if args.mixed_precision:
        override_dict["training"] = override_dict.get("training", {})
        override_dict["training"]["mixed_precision"] = True

    # Merge giving cli priority
    override_config = OmegaConf.create(override_dict)
    merged_config = OmegaConf.merge(config, override_config)
    assert isinstance(merged_config, DictConfig)

    return merged_config


def main() -> None:
    """Entrypoint."""
    try:
        args = parse_args()
        config = prepare_config(args)
        setup_logging(log_dir=config.get("log_dir", "logs"), log_level=config.get("log_level", "INFO"))
        create_directories(config)
        logger.info("Configuration:")
        logger.info(OmegaConf.to_yaml(config))
        trainer = Trainer(config)
        if args.eval_only:
            # TODO: evaluation mode
            logger.info("Evaluation-only mode not yet implemented")
        else:
            logger.info("Starting training")
            history = trainer.train()

            # plot
            if trainer.rank == 0:
                output_dir = config.get("output_dir", "output")
                plot_training_history(history, save_path=f"{output_dir}/training_history.png")

        logger.info("Done.s")

    except Exception as e:
        logger.error(f"Error during execution: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
