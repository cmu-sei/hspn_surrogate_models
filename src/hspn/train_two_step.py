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
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Optional, Tuple, Union

import h5py
import hydra
import numpy
import torch
from aim import Run
from omegaconf import DictConfig, OmegaConf
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TaskProgressColumn, TimeRemainingColumn
from torch import GradScaler, Tensor, nn
from torch.optim.lr_scheduler import LRScheduler
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader, DistributedSampler, TensorDataset

from hspn import install_global_log_context, set_log_context
from hspn.context import Context
from hspn.model import DeepOperatorNet
from hspn.train import evaluate, train
from hspn.train_utils import (
    NullProgress,
    ProgressT,
    save_checkpoint,
    wrap_as_distributed,
)

logger = logging.getLogger(__name__)


@dataclass
class TwoStepTrainingState:
    """Two step training state."""

    x_grid: Tensor
    F_train: Tensor
    U_train: Tensor
    A: Optional[Tensor] = None
    trunk_out_with_bias: Optional[Tensor] = None
    T_inv: Optional[Tensor] = None
    A_target: Optional[Tensor] = None


@dataclass(frozen=True)
class DONData:
    """Simple dataset loader for two step training."""

    trunk: Tensor
    branch: Tensor
    output: Tensor

    @classmethod
    def from_h5(
        cls,
        file_path: Path,
        branch_start: Union[int, float],
        branch_end: Union[int, float],
        trunk_start: Union[int, float] = 0.0,
        trunk_end: Union[int, float] = 1.0,
        dtype: torch.dtype = torch.float32,
    ) -> DONData:
        file_path = Path(file_path).resolve()

        logger.info(f"Loading HDF5 dataset from {file_path}")
        file = h5py.File(file_path, "r", swmr=True)

        branch = file["branch"]
        assert isinstance(branch, h5py.Dataset)

        trunk = file["trunk"]
        assert isinstance(trunk, h5py.Dataset)

        output = file["output"]
        assert isinstance(output, h5py.Dataset)

        trunk_subset, branch_subset, output_subset = cls._subset(
            branch, trunk, output, branch_start, branch_end, trunk_start, trunk_end, dtype=dtype
        )
        return cls(
            trunk=trunk_subset,
            branch=branch_subset,
            output=output_subset,
        )

    @classmethod
    def from_npy(
        cls,
        trunk_path: Path,
        branch_path: Path,
        output_path: Path,
        branch_start: Union[int, float],
        branch_end: Union[int, float],
        trunk_start: Union[int, float] = 0.0,
        trunk_end: Union[int, float] = 1.0,
        dtype: torch.dtype = torch.float32,
    ) -> "DONData":
        trunk = numpy.load(trunk_path)
        branch = numpy.load(branch_path)
        output = numpy.load(output_path)
        trunk_subset, branch_subset, output_subset = cls._subset(
            branch, trunk, output, branch_start, branch_end, trunk_start, trunk_end, dtype=dtype
        )
        return cls(
            trunk=trunk_subset,
            branch=branch_subset,
            output=output_subset,
        )

    @staticmethod
    def _subset(
        branch,
        trunk,
        output,
        branch_start: Union[int, float],
        branch_end: Union[int, float],
        trunk_start: Union[int, float],
        trunk_end: Union[int, float],
        dtype: torch.dtype,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        logger.info(f"Branch={branch.shape} Trunk={trunk.shape} Output={output.shape}")
        if branch_start <= 1:
            global_branch_start = int(branch_start * branch.shape[0])
        else:
            assert isinstance(branch_start, int)
            global_branch_start = branch_start
        if trunk_end <= 1:
            global_branch_end = int(branch_end * branch.shape[0])
        else:
            assert isinstance(branch_end, int)
            global_branch_end = branch_end
        branch_n = global_branch_end - global_branch_start
        assert branch_n > 0, f"Branch start and end indices must be valid ({global_branch_start}, {global_branch_end})"

        if trunk_start <= 1:
            global_trunk_start = int(trunk_start * trunk.shape[0])
        else:
            assert isinstance(trunk_start, int)
            global_trunk_start = trunk_start
        if trunk_end <= 1:
            global_trunk_end = int(trunk_end * trunk.shape[0])
        else:
            assert isinstance(trunk_end, int)
            global_trunk_end = trunk_end

        trunk_n = global_trunk_end - global_trunk_start
        logger.info(
            f"Calculated dataset subset: "
            f"{global_branch_start=} {global_branch_end=} {branch_n=} "
            f"{trunk_start=} {trunk_end=} {global_trunk_start=} {global_trunk_end=} {trunk_n=}"
        )
        assert trunk_n > 0, f"Trunk start and end indices must be valid ({global_trunk_start}, {global_trunk_end})"
        return (
            torch.tensor(trunk[global_trunk_start:global_trunk_end], dtype=dtype),
            torch.tensor(branch[global_branch_start:global_branch_end], dtype=dtype),
            torch.tensor(
                output[global_branch_start:global_branch_end, global_trunk_start:global_trunk_end], dtype=dtype
            ),
        )


@dataclass
class SampleConfig:
    """Optional sampling behavior"""

    seed: int = 0
    drop_last: bool = False
    shuffle: bool = False


@dataclass
class StepConfig:
    """Training config for a step of two-step training."""

    n_epochs: int
    enable_amp: bool
    grad_accum_steps: int
    enable_grad_scaling: bool
    grad_clip_norm: Optional[float]
    optimizer: Optimizer
    scheduler: Optional[LRScheduler]
    batch_size: int = 32
    sample_config: SampleConfig = field(default_factory=SampleConfig)


@dataclass
class TwoStepTrainConfig:
    """Main config for two-step training."""

    seed: int
    checkpoint_dir: Path
    comm_backend: Literal["nccl", "gloo"]
    log_interval: int
    model: DeepOperatorNet
    train_dataset: DONData
    val_dataloader: Optional[DataLoader]
    tracker: Optional[Run]
    trunk_config: StepConfig
    branch_config: StepConfig
    extra: Optional[Any] = None

    def __post_init__(self) -> None:
        self.checkpoint_dir = Path(self.checkpoint_dir)

    def validate(self) -> None:
        for step_name, step_config in [("trunk", self.trunk_config), ("branch", self.branch_config)]:
            if step_config.grad_accum_steps < 1:
                raise ValueError(
                    f"Invalid {step_name}_config.grad_accum_steps={step_config.grad_accum_steps}. Must be >= 1."
                )
            if step_config.n_epochs < 1:
                raise ValueError(f"Invalid {step_name}_config.n_epochs={step_config.n_epochs}. Must be >= 1.")
            if step_config.batch_size < 1:
                raise ValueError(f"Invalid {step_name}_config.batch_size={step_config.batch_size}. Must be >= 1.")

    @classmethod
    def from_cfg(cls, cfg: DictConfig) -> "TwoStepTrainConfig":
        """Instantiate from a `DictConfig` with dependency injection."""

        model: DeepOperatorNet = hydra.utils.instantiate(cfg.model)
        trunk_optimizer: Optimizer = hydra.utils.instantiate(
            cfg.trunk_config.optimizer, params=model.trunk_net.parameters()
        )
        trunk_scheduler = (
            hydra.utils.instantiate(cfg.trunk_config.scheduler, optimizer=trunk_optimizer)
            if cfg.trunk_config.get("scheduler")
            else None
        )
        trunk_config = StepConfig(
            n_epochs=cfg.trunk_config.n_epochs,
            enable_amp=cfg.trunk_config.enable_amp,
            grad_accum_steps=cfg.trunk_config.grad_accum_steps,
            enable_grad_scaling=cfg.trunk_config.enable_grad_scaling,
            grad_clip_norm=cfg.trunk_config.get("grad_clip_norm"),
            optimizer=trunk_optimizer,
            scheduler=trunk_scheduler,
            batch_size=cfg.trunk_config.get("batch_size", 32),
            sample_config=SampleConfig(**cfg.trunk_config.sample_config),
        )
        branch_optimizer: Optimizer = hydra.utils.instantiate(
            cfg.branch_config.optimizer, params=model.branch_net.parameters()
        )
        branch_scheduler = (
            hydra.utils.instantiate(cfg.branch_config.scheduler, optimizer=branch_optimizer)
            if cfg.branch_config.get("scheduler")
            else None
        )
        branch_config = StepConfig(
            n_epochs=cfg.branch_config.n_epochs,
            enable_amp=cfg.branch_config.enable_amp,
            grad_accum_steps=cfg.branch_config.grad_accum_steps,
            enable_grad_scaling=cfg.branch_config.enable_grad_scaling,
            grad_clip_norm=cfg.branch_config.get("grad_clip_norm"),
            optimizer=branch_optimizer,
            scheduler=branch_scheduler,
            batch_size=cfg.branch_config.get("batch_size", 32),
            sample_config=SampleConfig(**cfg.branch_config.sample_config),
        )
        train_dataset = hydra.utils.instantiate(cfg.train_dataset)
        val_dataloader = hydra.utils.instantiate(cfg.val_dataloader) if cfg.get("val_dataloader") else None
        tracker = hydra.utils.instantiate(cfg.tracker) if cfg.get("tracker") and Context.get().is_main_process else None
        self = cls(
            seed=cfg.seed,
            checkpoint_dir=cfg.checkpoint_dir,
            comm_backend=cfg.comm_backend,
            log_interval=cfg.log_interval,
            model=model,
            train_dataset=train_dataset,
            val_dataloader=val_dataloader,
            tracker=tracker,
            trunk_config=trunk_config,
            branch_config=branch_config,
            extra=cfg.get("extra"),
        )
        self.validate()
        return self


class TrunkAdapter(nn.Module):
    """Module adapter for trunk training stage in two step."""

    def __init__(self, trunk_net: nn.Module, A: Tensor):
        super().__init__()
        self.trunk = trunk_net
        self.A = A

    def forward(self, x: Tensor) -> Tensor:
        """Trunk forward"""
        return self.trunk(x) @ self.A


class BranchAdapter(nn.Module):
    """Module adapter for branch training stage in two step."""

    def __init__(self, branch_net: nn.Module):
        super().__init__()
        self.branch = branch_net

    def forward(self, f: Tensor) -> Tensor:
        """Branch forward"""
        z: Tensor = self.branch(f)
        ones = torch.ones(z.shape[0], 1, device=z.device)
        return torch.cat([z, ones], dim=1)


def train_two_step(
    config: TwoStepTrainConfig,
    state: TwoStepTrainingState,
    device: torch.device,
    progress_bar: ProgressT = NullProgress(),
) -> tuple[float, float]:
    ctx = Context.get()
    model = config.model

    A = torch.nn.Parameter(torch.randn((model.latent_dim, state.U_train.shape[0]), device=device))
    state.A = A

    trunk_model = TrunkAdapter(model.trunk_net, A)
    trunk_optimizer = config.trunk_config.optimizer.__class__(
        list(model.trunk_net.parameters()) + [A], **config.trunk_config.optimizer.defaults
    )

    trunk_ds = TensorDataset(state.x_grid, state.U_train.T)
    sampler = None
    shuffle = config.trunk_config.sample_config.shuffle
    if ctx.is_distributed:
        sampler = DistributedSampler(
            trunk_ds,
            shuffle=shuffle,
            seed=config.trunk_config.sample_config.seed,
            drop_last=config.trunk_config.sample_config.drop_last,
        )

    trunk_loader = DataLoader(
        trunk_ds,
        batch_size=config.trunk_config.batch_size,
        sampler=sampler,
        shuffle=shuffle and (sampler is None),
        pin_memory=torch.cuda.is_available(),
    )
    best_trunk_val_loss, best_trunk_epoch, _ = train(
        model=trunk_model,
        dataloader=trunk_loader,
        val_dataloader=None,
        optimizer=trunk_optimizer,
        scheduler=config.trunk_config.scheduler,
        scaler=GradScaler(device=device.type, enabled=config.trunk_config.enable_grad_scaling),
        tracker=config.tracker,
        checkpoint_dir=config.checkpoint_dir / "trunk",
        n_epochs=config.trunk_config.n_epochs,
        device=device,
        enable_amp=config.trunk_config.enable_amp,
        grad_accum_steps=config.trunk_config.grad_accum_steps,
        grad_clip_norm=config.trunk_config.grad_clip_norm,
        log_interval=config.log_interval,
        progress_bar=progress_bar,
        extra_tracker_context={"two_step_phase": "train_trunk"},
    )

    with torch.no_grad():
        trunk_out = model.trunk_net(state.x_grid.to(device))
        bias_col = torch.ones(trunk_out.shape[0], 1, device=device)
        trunk_out_with_bias = torch.cat([trunk_out, bias_col], dim=1)
        _, r_mat = torch.linalg.qr(trunk_out_with_bias, mode="reduced")
        T_inv: Tensor = torch.linalg.inv(r_mat)
        A_st: Tensor = torch.linalg.lstsq(trunk_out_with_bias, state.U_train.T).solution
        A_target: Tensor = r_mat @ A_st
        state.trunk_out_with_bias = trunk_out_with_bias
        state.T_inv = T_inv
        state.A_target = A_target.T

    branch_model = BranchAdapter(model.branch_net)

    sampler = None
    shuffle = config.branch_config.sample_config.shuffle
    if ctx.is_distributed:
        sampler = DistributedSampler(
            trunk_ds,
            shuffle=shuffle,
            seed=config.branch_config.sample_config.seed,
            drop_last=config.branch_config.sample_config.drop_last,
        )

    branch_ds = TensorDataset(state.F_train, state.A_target)
    branch_loader = DataLoader(
        branch_ds,
        batch_size=config.branch_config.batch_size,
        sampler=sampler,
        shuffle=shuffle and (sampler is None),
        pin_memory=torch.cuda.is_available(),
    )
    best_branch_val_loss, best_branch_epoch, _ = train(
        model=branch_model,
        dataloader=branch_loader,
        val_dataloader=None,
        optimizer=config.branch_config.optimizer,
        scheduler=config.branch_config.scheduler,
        scaler=GradScaler(device=device.type, enabled=config.branch_config.enable_grad_scaling),
        tracker=config.tracker,
        checkpoint_dir=config.checkpoint_dir / "branch",
        n_epochs=config.branch_config.n_epochs,
        device=device,
        enable_amp=config.branch_config.enable_amp,
        grad_accum_steps=config.branch_config.grad_accum_steps,
        grad_clip_norm=config.branch_config.grad_clip_norm,
        log_interval=config.log_interval,
        progress_bar=progress_bar,
        extra_tracker_context={"two_step_phase": "train_branch"},
    )

    if ctx.is_main_process:
        save_checkpoint(
            config.checkpoint_dir / "two_step_final.pt",
            model=model,
            optimizer=config.branch_config.optimizer,
            scheduler=config.branch_config.scheduler,
            gradscaler=GradScaler(device=device.type, enabled=config.branch_config.enable_grad_scaling),
            epoch=config.branch_config.n_epochs,
            metrics={
                "trunk_val_loss": best_trunk_val_loss,
                "branch_val_loss": best_branch_val_loss,
                "trunk_epoch": best_trunk_epoch,
                "branch_epoch": best_branch_epoch,
            },
            extra={"training_method": "two_step", "T_inv": state.T_inv.cpu() if state.T_inv is not None else None},
        )

    return best_trunk_val_loss, best_branch_val_loss


@hydra.main(config_path="pkg://hspn.conf", config_name="train_two_step", version_base=None)
def _main(cfg: DictConfig) -> float:
    ctx = Context()

    if os.environ.get("HSPN_DETERMINISTIC", False):
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

    # Log environment variables
    for k, v in os.environ.items():
        if any(x in k.lower() for x in ("ray", "hspn", "pbs", "slurm", "nvidia", "cuda", "tainer")):
            logger.info(f"[ENV] {k}={v}")

    start_time = time.time()
    best_trunk_loss = float("inf")
    best_branch_loss = float("inf")
    config = None

    try:
        config = TwoStepTrainConfig.from_cfg(cfg)

        device = torch.device(rank)
        if torch.cuda.is_available():
            torch.cuda.set_device(device)
        logger.info(f"Using {device}/{torch.get_device_module(device).device_count()}")

        model: nn.Module = config.model.train().to(device)
        if world_size > 1:
            model = nn.parallel.DistributedDataParallel(model)

        # Load training data
        state = TwoStepTrainingState(
            x_grid=config.train_dataset.trunk,
            F_train=config.train_dataset.branch,
            U_train=config.train_dataset.output,
        )

        # Setup progress bar
        progress_bar: ProgressT
        if ctx.is_main_process:
            progress_bar = Progress(
                "[progress.description]{task.description}",
                MofNCompleteColumn(),
                BarColumn(bar_width=None),
                TaskProgressColumn(show_speed=True),
                TimeRemainingColumn(),
            )
        else:
            progress_bar = NullProgress()

        # Run two-step training
        best_trunk_loss, best_branch_loss = train_two_step(
            config=config,
            state=state,
            device=device,
            progress_bar=progress_bar,
        )

        # Run validation if provided
        if config.val_dataloader and ctx.is_main_process:
            logger.info("Running final validation...")
            with ctx.model_eval(model):
                val_loss = evaluate(model, config.val_dataloader, device)
                logger.info(f"Final validation relative L2 loss: {val_loss:.6f}")

                if config.tracker:
                    config.tracker.track(val_loss, "final_val_loss", 0, context={"phase": "final_val"})

    except Exception:
        logger.exception("Training failed")
        raise

    finally:
        if ctx.is_main_process:
            duration = time.time() - start_time
            logger.info(f"Two-step training completed in {duration:.3f}s")
            logger.info(f"Best trunk loss: {best_trunk_loss:.6f}")
            logger.info(f"Best branch loss: {best_branch_loss:.6f}")

            if config and config.tracker:
                config.tracker.close()

    return min(best_trunk_loss, best_branch_loss)


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
