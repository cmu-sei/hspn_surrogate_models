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

import logging
import os
import random
import time
import typing
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Optional, Tuple

import hydra
import numpy as np
import torch
from aim import Run
from aim.sdk.types import AimObjectDict
from omegaconf import DictConfig
from omegaconf.omegaconf import OmegaConf
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TaskProgressColumn,
    TimeRemainingColumn,
)
from torch import GradScaler, nn
from torch.optim.lr_scheduler import LRScheduler
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader

from hspn import install_global_log_context, set_log_context
from hspn.context import Context
from hspn.dataset import H5Dataset
from hspn.train_utils import (
    NullProgress,
    ProgressT,
    load_checkpoint,
    save_checkpoint,
    wrap_as_distributed,
)

logger = logging.getLogger(__name__)


@dataclass
class TrainConfig:
    """Training options for standard DON training"""

    seed: int
    n_epochs: int
    checkpoint_dir: Path
    amp_dtype: Optional[Literal["fp16", "bf16"]]
    grad_accum_steps: int
    grad_scaling: Optional[bool]
    grad_clip_norm: Optional[float]
    model: nn.Module
    dataloader: DataLoader
    val_dataloader: DataLoader
    optimizer: Optimizer
    scheduler: Optional[LRScheduler]
    comm_backend: Literal["nccl", "gloo"]
    log_interval: int
    tracker: Optional[Run]
    extra: Optional[Any] = None

    def __post_init__(self) -> None:
        self.checkpoint_dir = Path(self.checkpoint_dir)

    def validate(self) -> None:
        """Validate after config is instantiated."""

        # Typecheck
        def _runtime_checkable(tp: type) -> bool:
            try:
                isinstance(None, tp)
                return True
            except TypeError:
                return False

        for name, field_def in self.__dataclass_fields__.items():
            assert isinstance(field_def.type, typing._Final) or (type(field_def.type) is type), (
                f"name: {field_def} {field_def.type} {type(field_def.type)}"
            )
            field_type = field_def.type
            value = getattr(self, name)

            if _runtime_checkable(field_type):  # type: ignore
                if not isinstance(value, field_type):  # type: ignore
                    raise TypeError(f"Field '{name}' expected {field_type}, got {type(value)}: {value}")

        # Cant batch on H5Dataset
        if (
            isinstance(getattr(self.dataloader, "dataset"), H5Dataset)
            and self.dataloader.batch_size
            and self.dataloader.batch_size != 1
        ):
            raise ValueError(
                f"Found an invalid value for {self.dataloader.batch_size=} Batching is currently handled by {H5Dataset!s}"
                "Please apply batch settings to the dataset."
            )

        if self.grad_accum_steps < 1:
            raise ValueError(f"Invalid value for {self.grad_accum_steps=}. Must be >= 1.")

        # Validate AMP makes sense
        if self.amp_dtype == "fp16":
            if self.grad_scaling is False:
                logger.warning("FP16 AMP without grad scaling may cause gradient overflow")
        elif self.amp_dtype == "bf16":
            if self.grad_scaling is True:
                logger.warning("BF16 AMP doesn't need grad scaling")
        elif self.amp_dtype is None and self.grad_scaling is True:
            logger.warning("Grad scaling enabled without AMP has no effect")

    @classmethod
    def from_cfg(cls, cfg: DictConfig) -> "TrainConfig":
        model = hydra.utils.instantiate(cfg.model)
        optimizer: Optimizer = hydra.utils.instantiate(cfg.optimizer, params=model.parameters())

        # auto-detect grad_scaling if None, only enable for fp16
        grad_scaling = cfg.get("grad_scaling", cfg.get("amp_dtype") == "fp16")

        self: TrainConfig = TrainConfig(
            **hydra.utils.instantiate(
                cfg,
                model=model,
                optimizer=optimizer,
                scheduler=cfg.scheduler and hydra.utils.instantiate(cfg.scheduler, optimizer=optimizer),
                tracker=cfg.tracker if Context.get().is_main_process else None,
                grad_scaling=grad_scaling,
            )
        )
        self.validate()
        return self


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    progress=NullProgress(),
) -> float:
    numer_acc = torch.zeros(1, device=device)
    denom_acc = torch.zeros(1, device=device)
    diff_buf = None
    taskid = progress.add_task("Evaluating", total=len(dataloader))
    for batch in dataloader:
        batch = [b.to(device) for b in batch]
        *inputs, target = batch

        pred = model(*inputs)

        if diff_buf is None or diff_buf.shape != pred.shape:
            diff_buf = torch.empty_like(pred)

        torch.sub(pred, target, out=diff_buf)
        numer_acc.add_(torch.dot(diff_buf.flatten(), diff_buf.flatten()))
        denom_acc.add_(torch.dot(target.flatten(), target.flatten()))
        progress.update(taskid, advance=1)

    Context.all_reduce_(numer_acc, op=torch.distributed.ReduceOp.SUM)
    Context.all_reduce_(denom_acc, op=torch.distributed.ReduceOp.SUM)

    return (numer_acc / denom_acc.clamp_min(1e-12)).item() if denom_acc.item() > 0 else float("inf")


def train(
    model: nn.Module,
    dataloader: DataLoader,
    val_dataloader: Optional[DataLoader],
    optimizer: Optimizer,
    scheduler: Optional[LRScheduler],
    scaler: GradScaler,
    tracker: Optional[Run],
    checkpoint_dir: Path,
    n_epochs: int,
    device: torch.device,
    starting_epoch: int = 1,
    starting_global_step: int = 0,
    starting_best_val_loss: float = float("inf"),
    amp_dtype: Optional[Literal["fp16", "bf16"]] = None,
    grad_accum_steps: int = 1,
    grad_clip_norm: Optional[float] = None,
    log_interval: int = 100,
    progress_bar: ProgressT = NullProgress(),
    extra_tracker_context: Optional[AimObjectDict] = None,
    extra_best_checkpoint_context: Optional[AimObjectDict] = None,
) -> Tuple[float, int, int]:
    """Train a model.

    Context must be initialized before calling this function.

    Returns:
        Tuple of `(best_val_loss, best_epoch, final_global_step)`
    """
    ctx = Context.get()

    logger.info(f"Using {device}")

    model.train().to(device)
    global_step = starting_global_step
    best_val_loss = starting_best_val_loss
    best_epoch = starting_epoch
    warned = False

    amp_enabled = amp_dtype is not None
    autocast_dtype = torch.bfloat16 if amp_dtype == "bf16" else torch.float16 if amp_dtype == "fp16" else torch.float32

    with progress_bar:
        for epoch in range(starting_epoch, n_epochs + 1):
            ctx.barrier()
            epoch_total_loss = 0.0
            epoch_batches = 0
            if os.environ.get("HSPN_PRECISE_TIMING", False):
                ctx.sync()
            if ctx.is_distributed and (set_epoch := getattr(dataloader.sampler, "set_epoch", lambda _: None)):
                set_epoch(epoch)
            epoch_start_time = time.time()

            task = progress_bar.add_task(f"Train Epoch {epoch}", total=len(dataloader))
            optimizer.zero_grad()

            for i, (*batch,) in enumerate(dataloader, start=1):
                *inputs, target = batch

                pass_ctx = ExitStack()
                pass_ctx.enter_context(torch.autocast(device.type, dtype=autocast_dtype, enabled=amp_enabled))
                # Sync DDP only on last microbatch
                if hasattr(model, "no_sync") and (i % grad_accum_steps != 0) and (i != len(dataloader)):
                    pass_ctx.enter_context(model.no_sync())

                with pass_ctx:
                    preds = model(*[x.to(device, non_blocking=False) for x in inputs])
                    loss = torch.nn.functional.mse_loss(preds, target.to(device), reduction="mean")
                    scaled_loss = loss / grad_accum_steps

                for e in inputs:
                    del e
                del target
                scaler.scale(scaled_loss.float()).backward()

                if device.type == "mps" and hasattr(scaler, "_scale") and not warned:
                    logger.warning(
                        "Patching scaler to avoid fp64 call on MPS. You might also need to set "
                        '`PYTORCH_ENABLE_MPS_FALLBACK="1"`'
                    )
                    warned = True
                    scaler._scale.double = scaler._scale.float  # type: ignore

                epoch_batches += 1
                epoch_total_loss += loss.item()

                if i % grad_accum_steps == 0:
                    if grad_clip_norm:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
                    scaler.step(optimizer)
                    scaler.update()
                    if scheduler:
                        scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1
                    progress_bar.update(task, advance=1)

                    if i > 0 and i % (log_interval / grad_accum_steps) == 0:
                        logger.info(
                            f"Epoch {epoch} [batch {i}/{len(dataloader)}] "
                            f"Loss: {loss.item():.6f}, "
                            f"Epoch Time Elapsed: {time.time() - epoch_start_time:.3f}s"
                        )

            # Handle tail flush
            if len(dataloader) % grad_accum_steps != 0:
                if grad_clip_norm:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
                scaler.step(optimizer)
                scaler.update()
                if scheduler:
                    scheduler.step()
                optimizer.zero_grad()
                global_step += 1

            avg_loss = epoch_total_loss / epoch_batches
            if tracker and ctx.is_main_process:
                log_context = {"phase": "train"} | (extra_tracker_context or {})
                tracker.track(epoch, "epoch", global_step, context=log_context)
                tracker.track(epoch_total_loss, "loss", global_step, context=log_context)
                tracker.track(avg_loss, "avg_loss", global_step, context=log_context)

                current_lr = scheduler.get_last_lr()[0] if scheduler else optimizer.param_groups[0]["lr"]
                tracker.track(current_lr, "learning_rate", global_step, context=log_context)

            progress_bar.remove_task(task)
            ctx.barrier()

            if ctx.is_main_process:
                if os.environ.get("HSPN_PRECISE_TIMING", False):
                    ctx.sync()
                logger.info(
                    f"Train Epoch: {epoch} completed in {time.time() - epoch_start_time:.3f}s "
                    f"Batches: {epoch_batches}, Avg Batch Total Loss: {avg_loss:.6f}"
                )

            # Val set
            val_loss = None
            if val_dataloader:
                with ctx.model_eval(model):
                    val_loss = evaluate(model, val_dataloader, device)
                    logger.info(f"Validation Relative L2 Loss: {val_loss:.6f}")

                if ctx.is_main_process:
                    if tracker:
                        log_context = extra_tracker_context or {}
                        log_context["phase"] = "val"
                        tracker.track(val_loss, "relative_l2_loss", global_step, context=log_context)

                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_epoch = epoch
                        logger.info(f"New Best Epoch: {epoch}")
                        save_checkpoint(
                            checkpoint_dir / "best_model.pt",
                            model=model,
                            optimizer=optimizer,
                            scheduler=scheduler,
                            gradscaler=scaler,
                            epoch=epoch,
                            metrics={
                                "val_loss": best_val_loss,
                                "best_val_loss": best_val_loss,
                                "epoch": epoch,
                                "global_step": global_step,
                            },
                            extra=extra_best_checkpoint_context,
                        )

    if _get_ddp_log := getattr(model, "_get_ddp_logging_data", False):
        logger.info("can_set_static_graph=%s", _get_ddp_log().get("can_set_static_graph"))  # type: ignore

    return best_val_loss, best_epoch, global_step


def _main(cfg: DictConfig) -> float:
    ctx = Context()

    rank, world_size = ctx.rank, ctx.world_size
    if ctx.is_distributed:
        set_log_context(rank=ctx.rank, world_size=ctx.world_size)
        install_global_log_context()

    logger.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")

    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)

    for k, v in os.environ.items():
        if any(x in k.lower() for x in ("ray", "hspn", "pbs", "slurm", "nvidia", "cuda", "tainer")):
            logger.info(f"[ENV] {k}={v}")

    best_val_loss = float("inf")
    epoch = best_epoch = 1
    start_time = time.time()
    tracker = None

    try:
        config = TrainConfig.from_cfg(cfg)

        device = torch.device(rank if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            torch.cuda.set_device(device)
        logger.info(f"Using {device}/{torch.get_device_module(device).device_count()}")

        model = config.model.train()
        if world_size > 1:
            model = nn.parallel.DistributedDataParallel(model)
        else:
            model.to(device)

        scaler = GradScaler(device=device.type, enabled=config.grad_scaling or False)

        if scaler.is_enabled():
            logger.info("AMP grad scaling enabled")
        if config.grad_clip_norm:
            logger.info(f"Grad clipping enabled with norm {config.grad_clip_norm}")

        global_step = 0

        # Load checkpoint if available
        if config.checkpoint_dir.exists() and (ckpts := list(config.checkpoint_dir.glob("checkpoint_*.pt"))):
            latest = max(ckpts)
            ckpt = load_checkpoint(
                latest,
                model=model,
                optimizer=config.optimizer,
                scheduler=config.scheduler,
                gradscaler=scaler,
                map_location=device,
            )
            best_epoch = epoch = ckpt.get("epoch", 1)
            global_step = ckpt.get("global_step", 0)
            best_val_loss = ckpt.get("best_val_loss", best_val_loss)
            if epoch >= config.n_epochs:
                logger.warning(f"Loaded checkpoint has epoch={ckpt['epoch']} >= {config.n_epochs=}")
            logger.info(f"After loading checkpoint epoch=best_epoch={best_epoch} {global_step=} {best_val_loss=}")

        cfg_dict = OmegaConf.to_container(cfg)
        assert isinstance(cfg_dict, dict)

        progress_bar: ProgressT
        if ctx.is_main_process:
            tracker = config.tracker
            if tracker:
                tracker["hparams"] = cfg_dict

            progress_bar = Progress(
                "[progress.description]{task.description}",
                MofNCompleteColumn(),
                BarColumn(bar_width=None),
                TaskProgressColumn(show_speed=True),
                TimeRemainingColumn(),
            )
        else:
            progress_bar = NullProgress()

        best_val_loss, best_epoch, _ = train(
            model=config.model,
            dataloader=config.dataloader,
            val_dataloader=config.val_dataloader,
            optimizer=config.optimizer,
            scheduler=config.scheduler,
            scaler=scaler,
            tracker=tracker,
            checkpoint_dir=config.checkpoint_dir,
            n_epochs=config.n_epochs,
            starting_epoch=epoch,
            starting_global_step=global_step,
            starting_best_val_loss=best_val_loss,
            amp_dtype=config.amp_dtype,
            grad_accum_steps=config.grad_accum_steps,
            grad_clip_norm=config.grad_clip_norm,
            log_interval=config.log_interval,
            device=device,
            progress_bar=progress_bar,
        )

    except:
        logger.exception("Failed")
        raise
    finally:
        if tracker and ctx.is_main_process:
            logger.info(f"{epoch}/{cfg.n_epochs} train epochs completed in {time.time() - start_time:.3f}s")
            logger.info(f"    Best Epoch: {best_epoch}")
            logger.info(f"    Best Val Relative L2 Loss: {best_val_loss:.6f}")
            tracker.close()

    return best_val_loss


_w_main = wrap_as_distributed(_main)
main = hydra.main(config_path="pkg://hspn.conf", config_name="train", version_base=None)(_w_main)

if __name__ == "__main__":
    import logging
    import sys
    from logging import StreamHandler

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[StreamHandler(sys.stdout)],
    )
    main()
