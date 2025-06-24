#
# HyperSPIN code - hspn_surrogate_models
#
# Copyright 2025 Carnegie Mellon University.
#
# NO WARRANTY. THIS CARNEGIE MELLON UNIVERSITY AND SOFTWARE ENGINEERING INSTITUTE MATERIAL IS FURNISHED ON AN "AS-IS" BASIS. CARNEGIE MELLON UNIVERSITY MAKES NO WARRANTIES OF ANY KIND, EITHER EXPRESSED OR IMPLIED, AS TO ANY MATTER INCLUDING, BUT NOT LIMITED TO, WARRANTY OF FITNESS FOR PURPOSE OR MERCHANTABILITY, EXCLUSIVITY, OR RESULTS OBTAINED FROM USE OF THE MATERIAL. CARNEGIE MELLON UNIVERSITY DOES NOT MAKE ANY WARRANTY OF ANY KIND WITH RESPECT TO FREEDOM FROM PATENT, TRADEMARK, OR COPYRIGHT INFRINGEMENT.
#
# Licensed under a MIT (SEI)-style license, please see license.txt or contact permission@sei.cmu.edu for full terms.
#
# [DISTRIBUTION STATEMENT A] This material has been approved for public release and unlimited distribution.  Please see Copyright notice for non-US Government use and distribution.
#
# This Software includes and/or makes use of Third-Party Software each subject to its own license.
#
# DM25-0396
#
from __future__ import annotations
import logging
import os
import socket
from datetime import datetime
from functools import wraps
from pathlib import Path
from pprint import pformat
from types import TracebackType
from typing import Any, Callable, Dict, Optional, OrderedDict, ParamSpec, Protocol, Type, TypeVar, Union

import torch
import torch.distributed as dist
from torch.multiprocessing.spawn import spawn
import torch.nn as nn
from rich.progress import TaskID
from torch import GradScaler
from torch.distributed.fsdp import FullyShardedDataParallel
from torch.nn.parallel import DistributedDataParallel
from torch.optim.optimizer import Optimizer

from hspn.context import Context

P = ParamSpec("P")
R = TypeVar("R")

logger = logging.getLogger(__name__)


def _get_master_port():
    preferred_port = 0
    # If we see GPU, try to disambiguate using and offset based on the first visible GPU index
    #  to account for multiple distributed trials on the same node.
    #  Otherwise, find a free port and use that
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cuda_visible:
        try:
            first_gpu = int(cuda_visible.split(",")[0])
            preferred_port = 29500 + first_gpu
            logger.info(
                f"Preferring master port {preferred_port} from GPU offset (regardless of whether we are using GPUs this is just for offset)"
            )
        except Exception:
            logger.info(
                f"CUDA available but failed to parse CUDA_VISIBLE_DEVICES{cuda_visible=} for master port offset"
            )
            pass
    logger.info("Creating socket")
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            logger.info(f"Asking for port {'from os' if preferred_port == 0 else preferred_port}")
            s.bind(("", preferred_port))
            port = s.getsockname()[1]
        except OSError:
            logger.info(f"Could not get port {preferred_port} asking for a port from os")
            s.bind(("", 0))
            port = s.getsockname()[1]
    logger.info(f"Setting master port {port}")
    return port


def worker_fn(rank: int, world_size: int, fn, args, kwargs) -> None:
    os.environ["RANK"] = str(rank)
    backend = "gloo"
    if torch.cuda.is_available():
        backend = "nccl"
        cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        if cvd and len(cvd.split(",")) == 1:
            logger.info(f'Found CUDA_VISIBLE_DEVICES="{cvd}" skipping `torch.cuda.set_device`')
        else:
            logger.info(f'Found CUDA_VISIBLE_DEVICES="{cvd}" setting `torch.cuda.set_device({rank})`')
            torch.cuda.set_device(rank)
    logger.info(f"Initializing process group {backend=} {rank=} {world_size=}")
    dist.init_process_group(backend=backend, init_method="env://", rank=rank, world_size=world_size)
    try:
        res = fn(*args, **kwargs)
        logger.info(f"returned from func w value {res} shutting down")
    finally:
        dist.destroy_process_group()


def wrap_as_distributed(fn: Callable[P, R]):
    @wraps(fn)
    def wrapper(*args: P.args, **kwargs: P.kwargs):
        if torch.cuda.is_available():
            world_size = torch.cuda.device_count()
        else:
            world_size = int(os.environ.get("WORLD_SIZE", "1"))

        if world_size <= 1:
            logger.info("Running without distributed.")
            fn(*args, **kwargs)
            return

        master_port = _get_master_port()
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["MASTER_PORT"] = str(master_port)
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        logger.info(f"Spawning {world_size=} processes. {os.environ.get('MASTER_ADDR')}:{master_port}")
        spawn(
            worker_fn,
            args=(world_size, fn, args, kwargs),
            nprocs=world_size,
            join=True,
        )

    return wrapper


@Context.on_rank(0)
def save_checkpoint(
    filepath: Union[str, Path],
    model: nn.Module,
    optimizer: Optimizer,
    scheduler: Optional[Any],
    gradscaler: Optional[GradScaler],
    epoch: int,
    metrics: Dict[str, float],
    extra: Optional[Dict] = None,
) -> None:
    """Save model checkpoint.

    Args:
        filepath: Path to save the checkpoint
        model: Model to save, will be heuristically unwrapped
        optimizer: Optimizer state to save
        scheduler: Learning rate scheduler to save (optional)
        gradscaler: `torch.GradScaler` to save (optional)
        epoch: Current epoch number
        metrics: Dictionary of metrics to save
        extra: Any extra metadata to save (optional)
    """
    logger.info("Saving checkpoint...")
    filepath = Path(filepath).resolve()
    filepath.parent.mkdir(parents=True, exist_ok=True)

    cpu_sd = {k: v.cpu() for k, v in unwrap(model).state_dict().items()}

    def optim_state_to(optim, device="cpu"):
        opt_sd = optim.state_dict()
        cpu_opt_state = {}
        for param_id, param_state in opt_sd["state"].items():
            cpu_opt_state[param_id] = {
                k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in param_state.items()
            }
        return opt_sd

    def sched_state_to(sched, device="cpu"):
        return {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in sched.state_dict().items()}

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": cpu_sd,
        "optimizer_state_dict": optim_state_to(optimizer) if optimizer else None,
        "scheduler_state_dict": sched_state_to(scheduler) if scheduler else None,
        "gradscaler_state_dict": sched_state_to(gradscaler) if gradscaler else None,
        "metrics": metrics,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S %Z"),
        "extra": extra,
    }

    torch.save(checkpoint, filepath)
    logger.info(f"Saved checkpoint to {filepath!s}")


def load_checkpoint(
    filepath: Union[str, Path],
    model: Optional[nn.Module] = None,
    optimizer: Optional[Optimizer] = None,
    scheduler: Optional[Any] = None,
    gradscaler: Optional[GradScaler] = None,
    map_location: Union[str, torch.device, None] = None,
) -> Dict[str, Any]:
    """Load model checkpoint.

    Args:
        filename: Path to checkpoint file
        model: Model to load weights into (optional)
        optimizer: Optimizer to load state into (optional)
        scheduler: Learning rate scheduler to load state into (optional)
        gradscaler: GradScaler to load state into (optional)
        map_location: Device to map tensors to

    Returns:
        Dictionary containing checkpoint data
    """
    logger.info(f"Loading checkpoint from {filepath!s}")

    checkpoint = torch.load(filepath, map_location=map_location, weights_only=True)
    logger.info("Loaded checkpoint")
    for k, v in checkpoint.items():
        if "state" in k or isinstance(v, (OrderedDict, torch.Tensor)):
            continue
        logger.info(f"\t{k}: {pformat(v)}")

    if model is not None:
        logger.info("Loading model state")
        model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        logger.info("Loading optimizer state")
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        logger.info("Loading scheduler state")
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    if gradscaler is not None and "gradscaler_state_dict" in checkpoint:
        logger.info("Loading gradscaler state")
        gradscaler.load_state_dict(checkpoint["gradscaler_state_dict"])

    return checkpoint


def unwrap(model: Union[nn.Module, DistributedDataParallel, FullyShardedDataParallel]) -> nn.Module:
    return getattr(model, "module", model)


class ProgressT(Protocol):
    """Duck progress bar"""

    def __enter__(self) -> "ProgressT": ...

    def __exit__(
        self,
        exc_type: Type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None: ...

    def add_task(self, *args, **kwargs) -> TaskID: ...

    def update(self, *args, **kwargs) -> None: ...

    def remove_task(self, *args, **kwargs) -> None: ...

    def stop(self) -> None: ...

    def start(self) -> None: ...

    def advance(self, *args, **kwargs) -> None: ...


class NullProgress:
    """No-op progress bar"""

    def __enter__(self) -> NullProgress:
        return self

    def __exit__(
        self,
        exc_type: Type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        _ = exc_type, exc_val, exc_tb

    def add_task(self, *args, **kwargs) -> TaskID:
        _ = args, kwargs
        return TaskID(0)

    def update(self, *args, **kwargs) -> None:
        del args, kwargs

    def remove_task(self, *args, **kwargs) -> None:
        del args, kwargs

    def stop(self) -> None:
        pass

    def start(self) -> None:
        pass

    def advance(self, *args, **kwargs) -> None:
        del args, kwargs
