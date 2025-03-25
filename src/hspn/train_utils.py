import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Literal, Optional, Tuple, Union

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.optim.optimizer import Optimizer

logger = logging.getLogger(__name__)


def setup_distributed(
    backend: Literal["nccl", "gloo"] = "nccl",
) -> Tuple[int, int]:
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
    filepath: Union[str, Path],
    model: nn.Module,
    optimizer: Optimizer,
    scheduler: Optional[Any],
    epoch: int,
    metrics: Dict[str, float],
    extra: Optional[Dict] = None,
) -> None:
    """Save model checkpoint.

    Args:
        filepath: Path to save the checkpoint
        model: Model to save
        optimizer: Optimizer state to save
        scheduler: Learning rate scheduler to save (optional)
        epoch: Current epoch number
        metrics: Dictionary of metrics to save
        extra: Any extra metadata to save
    """
    rank = dist.get_rank() if dist.is_initialized() else 0

    if rank != 0:
        return
    filepath = Path(filepath).resolve()
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
    logger.info(f"Saved checkpoint to {filepath!s}")


def load_checkpoint(
    filepath: Union[str, Path],
    model: Union[nn.Module, None] = None,
    optimizer: Union[Optimizer, None] = None,
    scheduler: Union[Any, None] = None,
    map_location: Union[str, torch.device, None] = None,
) -> Dict[str, Any]:
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
    logger.info(f"Loading checkpoint from {filepath!s}")

    checkpoint = torch.load(filepath, map_location=map_location)
    ts = checkpoint.get("timestamp")
    metrics = checkpoint.get("metrics")
    extra = checkpoint.get("extra")
    logger.info("Loaded checkpoint")
    if ts:
        logger.info(f"\ttimestamp: {ts}")
    if metrics:
        logger.info(f"\tmetrics: {json.dumps(metrics)}")
    if extra:
        logger.info(f"\textra: {json.dumps(extra)}")

    if model is not None:
        logger.info("Loading model state")
        model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        logger.info("Loading optimizer state")
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        logger.info("Loading scheduler state")
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    return checkpoint
