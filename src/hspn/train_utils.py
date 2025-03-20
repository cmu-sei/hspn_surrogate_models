import logging
import os
from pathlib import Path
from datetime import datetime
from typing import Any, Literal

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim

logger = logging.getLogger(__name__)


def setup_distributed(
    backend: Literal["nccl", "gloo"] = "nccl",
) -> tuple[int, int]:
    """Initialize distributed environment with support for SLURM.

    Handles both Slurm and manual initialization.

    Args:
        backend: PyTorch distributed backend

    Returns:
        Tuple of (local_rank, world_size)
    """
    # Check if running under Slurm
    if "SLURM_PROCID" in os.environ:
        logger.info(f"SLURM detected {os.environ['SLURM_PROCID']=}")
        rank = int(os.environ["SLURM_PROCID"])
        world_size = int(os.environ["SLURM_NTASKS"])
        local_rank = int(os.environ["SLURM_LOCALID"])

        # Set master address for communication
        if "SLURM_LAUNCH_NODE_IPADDR" in os.environ:
            os.environ["MASTER_ADDR"] = os.environ["SLURM_LAUNCH_NODE_IPADDR"]
        elif "SLURM_JOB_NODELIST" in os.environ:
            # Parse nodelist to get master node
            # might need to use the 'scontrol show hostname' command to do this properly
            # TODO:
            master_node = os.environ["SLURM_JOB_NODELIST"].split(",")[0]
            os.environ["MASTER_ADDR"] = master_node
        else:
            os.environ["MASTER_ADDR"] = "127.0.0.1"

        # Set master port (avoid conflicts between different jobs)
        job_id = os.environ.get("SLURM_JOB_ID", "0")
        os.environ["MASTER_PORT"] = str(12345 + int(job_id) % 10000)

    elif "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        # Manual distributed setup
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))

        if "MASTER_ADDR" not in os.environ:
            os.environ["MASTER_ADDR"] = "127.0.0.1"
        if "MASTER_PORT" not in os.environ:
            os.environ["MASTER_PORT"] = "12345"
    else:
        logger.info("Running without distributed.")
        rank = 0
        world_size = 1
        local_rank = 0

    if world_size > 1:
        logger.info("Initializing process group.")
        dist.init_process_group(backend=backend, rank=rank, world_size=world_size)

        logger.info(
            f"Initialized distributed process group: "
            f"rank={rank}/{world_size}, "
            f"local_rank={local_rank}, "
            f"master={os.environ['MASTER_ADDR']}:{os.environ['MASTER_PORT']}"
        )

        # Set device based on local rank
        if backend == "nccl":
            torch.cuda.set_device(local_rank)

    return rank, world_size


def save_checkpoint(
    filepath: Path,
    model: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: Any | None,
    epoch: int,
    metrics: dict[str, float],
    is_best: bool = False,
    extra: dict[str, Any] | None = None,
) -> None:
    """Save model checkpoint.

    Args:
        filepath: Path to save the checkpoint
        model: Model to save
        optimizer: Optimizer state to save
        scheduler: Learning rate scheduler to save (optional)
        epoch: Current epoch number
        metrics: Dictionary of metrics to save
        is_best: Whether this is the best model so far
        extra: Any extra metadata to save
    """
    filepath = filepath.resolve()
    filepath.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer else None,
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "metrics": metrics,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "extra": extra,
    }

    torch.save(checkpoint, filepath)

    if is_best:
        best_filepath = filepath.with_stem("best_model")
        torch.save(checkpoint, best_filepath)
        logger.info(f"Saved best model to {best_filepath!s}")

    logger.info(f"Saved checkpoint to {filepath!s}")


def load_checkpoint(
    filename: Path,
    model: nn.Module | None = None,
    optimizer: optim.Optimizer | None = None,
    scheduler: Any | None = None,
    map_location: str | torch.device | None = None,
) -> dict[str, Any]:
    """Load model checkpoint.

    Args:
        filename: Path to checkpoint file
        model: Model to load weights into (optional)
        optimizer: Optimizer to load state into (optional)
        scheduler: Learning rate scheduler to load state into (optional)
        map_location: Device to map tensors to

    Returns:
        Dictionary containing checkpoint data
    """
    logger.info(f"Loading checkpoint from {filename!s}")

    checkpoint = torch.load(filename, map_location=map_location)

    # Load model weights if provided
    if model is not None:
        model.load_state_dict(checkpoint["model_state_dict"])

    # Load optimizer state if provided
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    # Load scheduler state if provided
    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    return checkpoint
