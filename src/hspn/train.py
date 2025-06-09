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
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Literal, Optional

import hydra
import numpy
import torch
from omegaconf import DictConfig
from omegaconf.omegaconf import OmegaConf
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TaskProgressColumn,
    TimeRemainingColumn,
)
from torch import nn
from torch.optim.lr_scheduler import LRScheduler
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader

from hspn.context import Context
from hspn.dataset import H5Dataset
from hspn.tracker import Tracker
from hspn.train_utils import (
    NullProgress,
    install_global_log_context,
    load_checkpoint,
    save_checkpoint,
    set_log_context,
    wrap_as_distributed,
)

logger = logging.getLogger(__name__)


@dataclass
class TrainConfig:
    seed: int
    n_epochs: int
    checkpoint_dir: Path
    enable_amp: bool
    grad_accum_steps: int
    enable_grad_scaling: bool
    grad_clip_norm: Optional[float]
    model: nn.Module
    dataloader: DataLoader
    val_dataloader: DataLoader
    optimizer_factory: Callable[..., Optimizer]
    scheduler_factory: Callable[..., LRScheduler]
    comm_backend: Literal["nccl", "gloo"]
    log_interval: int
    tracker_config: Optional[Dict[str, Any]]
    extra: Optional[Any] = None

    def __post_init__(self):
        self.checkpoint_dir = Path(self.checkpoint_dir)

    def validate(self):
        """Validate after config is instantiated."""
        if (
            isinstance(getattr(self.dataloader, "dataset"), H5Dataset)
            and self.dataloader.batch_size
            and self.dataloader.batch_size != 1
        ):
            raise ValueError(
                f"Found an invalid value for {self.dataloader.batch_size=} Batching is currently handled by {H5Dataset!s}"
                "Please apply batch settings to the dataset."
            )
        assert self.grad_accum_steps >= 1


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
    for branch_in, trunk_in, output in dataloader:
        output = output.to(device, non_blocking=False)
        branch_in = branch_in.to(device, non_blocking=False)
        trunk_in = trunk_in.to(device, non_blocking=False)

        pred = model(branch_in, trunk_in)

        if diff_buf is None or diff_buf.shape != pred.shape:
            diff_buf = torch.empty_like(pred)

        torch.sub(pred, output, out=diff_buf)
        numer_acc.add_(torch.dot(diff_buf.flatten(), diff_buf.flatten()))
        denom_acc.add_(torch.dot(output.flatten(), output.flatten()))
        progress.update(taskid, advance=1)

    Context.all_reduce_(numer_acc, op=torch.distributed.ReduceOp.SUM)
    Context.all_reduce_(denom_acc, op=torch.distributed.ReduceOp.SUM)

    return (numer_acc / denom_acc.clamp_min(1e-12)).item() if denom_acc.item() > 0 else float("inf")


def train(
    model: nn.Module,
    dataloader: DataLoader,
    val_dataloader: DataLoader,
    optimizer: Optimizer,
    scheduler: Optional[LRScheduler],
    scaler: torch.amp.GradScaler,
    tracker: Optional[Tracker],
    checkpoint_dir: Path,
    n_epochs: int,
    device: torch.device,
    starting_epoch: int = 1,
    starting_global_step: int = 0,
    starting_best_val_loss: float = float("inf"),
    enable_amp: bool = False,
    grad_accum_steps: int = 1,
    grad_clip_norm: Optional[float] = None,
    log_interval: int = 100,
    progress_bar: Progress | NullProgress = NullProgress(),
) -> tuple[float, int, int]:
    """Train a model.

    Context must be initialized before calling this function.

    Returns:
        tuple[best_val_loss, best_epoch, final_global_step]
    """
    ctx = Context.get()

    world_size = ctx.world_size
    logger.info(f"Using {device}")

    model.train().to(device)
    global_step = starting_global_step
    best_val_loss = starting_best_val_loss
    best_epoch = starting_epoch
    warned = False

    with progress_bar:
        for epoch in range(starting_epoch, n_epochs + 1):
            ctx.barrier()
            epoch_total_loss = 0.0
            epoch_batches = 0
            if os.environ.get("HSPN_PRECISE_TIMING", False):
                ctx.sync()
            epoch_start_time = time.time()

            task = progress_bar.add_task(f"Train Epoch {epoch}", total=len(dataloader))
            accum = grad_accum_steps
            optimizer.zero_grad()

            for i, (branch_in, trunk_in, output) in enumerate(dataloader, start=1):
                output = output.to(device, non_blocking=False)

                with torch.autocast(device.type, dtype=torch.bfloat16, enabled=enable_amp):
                    b_in = branch_in.to(device, non_blocking=False)
                    t_in = trunk_in.to(device, non_blocking=False)
                    preds = model(b_in, t_in)
                    del b_in, t_in
                    loss = torch.nn.functional.mse_loss(preds, output, reduction="mean")
                    loss.mul_(world_size / accum)

                del branch_in, trunk_in
                scaler.scale(loss.float()).backward()

                if device.type == "mps" and hasattr(scaler, "_scale") and not warned:
                    logger.warning(
                        'Patching scaler to avoid fp64 call on MPS. You might also need to set `PYTORCH_ENABLE_MPS_FALLBACK="1"`'
                    )
                    warned = True
                    scaler._scale.double = scaler._scale.float

                epoch_batches += 1
                epoch_total_loss += loss.item()

                if i % accum == 0:
                    if grad_clip_norm:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    if scheduler:
                        scheduler.step()
                    global_step += 1
                    progress_bar.update(task, advance=1)

                    if i > 0 and i % (log_interval / grad_accum_steps) == 0:
                        logger.info(
                            f"Epoch {epoch} [batch {i}/{len(dataloader)}] "
                            f"Loss: {loss.item():.6f}, "
                            f"Epoch Time Elapsed: {time.time() - epoch_start_time:.3f}s"
                        )

            avg_loss = epoch_total_loss / epoch_batches
            if tracker and ctx.is_main_process:
                tracker.log_scalar("train/epoch", epoch, global_step)
                tracker.log_scalar("train/loss", epoch_total_loss, global_step)
                tracker.log_scalar("train/avg_loss", avg_loss, global_step)

                current_lr = scheduler.get_last_lr()[0] if scheduler else optimizer.param_groups[0]["lr"]
                tracker.log_scalar("train/learning_rate", current_lr, global_step)

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
            ctx.barrier()
            with ctx.model_eval(model):
                val_loss = evaluate(model, val_dataloader, device)
                logger.info(f"Validation Relative L2 Loss: {val_loss:.6f}")
            ctx.barrier()

            if ctx.is_main_process:
                if tracker:
                    tracker.log_scalar("val/relative_l2_loss", val_loss, global_step)

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
                        extra=None,  # Will be passed from main
                    )

    if _get_ddp_log := getattr(model, "_get_ddp_logging_data", False):
        logger.info("can_set_static_graph=%s", _get_ddp_log().get("can_set_static_graph"))

    return best_val_loss, best_epoch, global_step


@hydra.main(config_path="pkg://hspn.conf", config_name="train", version_base=None)
def _main(cfg: DictConfig) -> float:
    ctx = Context()
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    rank, world_size = ctx.rank, ctx.world_size
    if ctx.is_distributed:
        set_log_context(rank=ctx.rank, world_size=ctx.world_size)
        install_global_log_context()

    OmegaConf.resolve(cfg)
    logger.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")

    # Set seeds
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)
    numpy.random.seed(cfg.seed)
    random.seed(cfg.seed)

    # Log env vars
    for k, v in os.environ.items():
        if any(x in k.lower() for x in ("ray", "hspn", "pbs", "slurm", "nvidia", "cuda", "tainer")):
            logger.info(f"[ENV] {k}={v}")

    best_val_loss = float("inf")
    epoch = best_epoch = 1
    start_time = time.time()
    tracker = None

    try:
        config: TrainConfig = TrainConfig(**hydra.utils.instantiate(cfg))
        config.validate()

        device = torch.device(rank)
        if torch.cuda.is_available():
            torch.cuda.set_device(device)
        logger.info(f"Using {device}")

        model = config.model.train()
        model.to(device)
        if world_size > 1:
            model = nn.parallel.DistributedDataParallel(model)

        dataloader = config.dataloader
        optimizer = config.optimizer_factory(model.parameters())
        scheduler = config.scheduler_factory(optimizer) if config.scheduler_factory else None
        scaler = torch.amp.GradScaler(device=device.type, enabled=config.enable_grad_scaling)

        if scaler.is_enabled():
            logger.info("AMP grad scaling enabled")
        if config.grad_clip_norm:
            logger.info(f"Grad clipping enabled with norm {config.grad_clip_norm}")

        global_step = 0

        # Load checkpoint if available
        if config.checkpoint_dir.exists() and (ckpts := list(config.checkpoint_dir.glob("checkpoint_*.pt"))):
            latest = sorted(ckpts)[-1]
            ckpt = load_checkpoint(
                latest,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
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

        if ctx.is_main_process:
            if config.tracker_config:
                tracker = Tracker(**config.tracker_config)
                tracker.log_hparams(cfg_dict)

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
            model=model,
            dataloader=dataloader,
            val_dataloader=config.val_dataloader,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            tracker=tracker,
            checkpoint_dir=config.checkpoint_dir,
            n_epochs=config.n_epochs,
            starting_epoch=epoch,
            starting_global_step=global_step,
            starting_best_val_loss=best_val_loss,
            enable_amp=config.enable_amp,
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


main = wrap_as_distributed(_main)

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
