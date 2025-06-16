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

import contextlib
import logging
import os
from functools import wraps
from typing import Callable, Optional, ParamSpec, TypeVar

import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel
from torch.nn.parallel import DistributedDataParallel

T = TypeVar("T")
P = ParamSpec("P")
R = TypeVar("R")

logger = logging.getLogger(__name__)


class Context:
    """Global distributed‑aware state."""

    _instance: Optional["Context"] = None

    def __new__(cls, *args, **kwargs) -> "Context":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        if getattr(self, "_initialized", False):
            return
        self._initialized = True

        if dist.is_available() and dist.is_initialized():
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
            self.backend = dist.get_backend()
            self.local_rank = int(os.environ.get("LOCAL_RANK", self.rank if self.backend == "nccl" else 0))
        else:
            # single‑process defaults
            self.rank = 0
            self.world_size = 1
            self.backend = None
            self.local_rank = 0

    @classmethod
    def get(cls) -> "Context":
        if cls._instance is None:
            raise RuntimeError(f"{cls.__qualname__} has not been initialized. Create an instance first.")
        return cls._instance

    @property
    def is_distributed(self) -> bool:
        return self.world_size > 1

    @property
    def is_main_process(self) -> bool:
        return self.rank == 0

    def sync(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        if torch.mps.is_available():
            torch.mps.synchronize()

    def barrier(self):
        """Block until all ranks synchronize (no-op if single‑process)."""
        if self.is_distributed:
            dist.barrier()

    def no_sync(self, ddp_module: torch.nn.Module, /, enabled: bool = True):
        """Temporarily disable grad sync.

        Only works for DDP modules in distributed mode.
        """
        if enabled and self.is_distributed and hasattr(ddp_module, "no_sync"):
            assert isinstance(ddp_module, (DistributedDataParallel, FullyShardedDataParallel))
            return ddp_module.no_sync()
        return contextlib.nullcontext()

    @classmethod
    def all_reduce_(cls, tensor: torch.Tensor, op=dist.ReduceOp.SUM):
        """All‑reduce, no-op unless distributed."""
        if cls.get().is_distributed:
            dist.all_reduce(tensor, op=op)

    @contextlib.contextmanager
    def model_eval(self, model: torch.nn.Module):
        """Temporarily switch a model to eval mode."""
        was_train = model.training
        model.eval()
        yield
        if was_train:
            model.train()

    @staticmethod
    def on_rank(r: int = 0):
        """Decorator to run function only on given rank with optional barrier fence.

        The function cannot return a value.
        """

        def deco(fn: Callable[..., None]) -> Callable[..., None]:
            import inspect

            if (ret := inspect.signature(fn).return_annotation) not in (None, inspect.Signature.empty):
                raise TypeError(f"Cannot use `on_rank` with a function that returns a value. Got {ret}")

            @wraps(fn)
            def wrapped(*args, **kwargs):
                ctx = Context.get()
                if ctx.rank == r:
                    fn(*args, **kwargs)

            return wrapped

        return deco

    @contextlib.contextmanager
    def timing(self, label: str, *, distributed: bool = False, reduce_op: str = "avg"):
        """Time a block across all ranks (if distributed), reduce (avg/min/max/sum) and log on main."""
        import time

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        if distributed and self.is_distributed:
            dist.barrier()
        start = time.perf_counter()
        yield
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        if distributed and self.is_distributed:
            dist.barrier()
        elapsed = time.perf_counter() - start
        reduce_op_str = ""
        if distributed and self.is_distributed:
            t = torch.tensor(elapsed, requires_grad=False, device=self.local_rank)
            if reduce_op == "min":
                dist.all_reduce(t, op=dist.ReduceOp.MIN)
            elif reduce_op == "max":
                dist.all_reduce(t, op=dist.ReduceOp.MAX)
            else:
                dist.all_reduce(t, op=dist.ReduceOp.SUM)
                if reduce_op != "sum":
                    t /= self.world_size
            reduce_op_str = f"({reduce_op})"
            elapsed = t.item()
        if self.is_main_process:
            logger.info(f"[{label}] {elapsed:.4f}s {reduce_op_str}")

    @staticmethod
    def timed(label: str, *, distributed: bool = False, reduce_op: str = "avg"):
        """Decorator to wrap function call in a distributed timing block."""

        def deco(fn):
            @wraps(fn)
            def wrapped(*args, **kwargs):
                ctx = Context.get()
                with ctx.timing(label, distributed=distributed, reduce_op=reduce_op):
                    return fn(*args, **kwargs)

            return wrapped

        return deco


if __name__ == "__main__":
    from typing import reveal_type

    test_ctx = Context.get()
    reveal_type(test_ctx)

    assert isinstance(test_ctx, Context)
    assert test_ctx is Context.get()  # Same instance
