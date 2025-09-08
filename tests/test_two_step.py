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

import operator
import uuid
from pathlib import Path
from typing import Generator, NamedTuple, Tuple

import numpy
import pytest
import torch
from hydra import compose, initialize
from omegaconf import DictConfig

from hspn.context import Context
from hspn.train_two_step import DONData, TwoStepTrainConfig, TwoStepTrainingState, train_two_step
from hspn.train_utils import NullProgress


class NpyPaths(NamedTuple):
    f_path: Path
    u_path: Path
    x_grid_path: Path
    f_train_path: Path
    f_test_path: Path
    u_train_path: Path
    u_test_path: Path


@pytest.fixture
def npy_data(
    *,
    tmp_path: Path,
    n_trunk: int = 1024,
    trunk_dim: int = 3,
    n_branch_train: int = 15,
    n_branch_test: int = 6,
    branch_dim: int = 1,
    output_dir: str = "mock-data",
) -> Tuple[Path, Path, Path, Path, Path, Path, Path]:
    """Generate mock data for two-step training."""
    x_grid = torch.rand(n_trunk, trunk_dim)
    f_train = torch.rand(n_branch_train, branch_dim)
    f_test = torch.rand(n_branch_test, branch_dim)
    u_train = torch.rand(n_branch_train, n_trunk)
    u_test = torch.rand(n_branch_test, n_trunk)

    f = torch.concat((f_train, f_test))
    u = torch.concat((u_train, u_test))

    def _save(data: torch.Tensor, fpath: Path) -> None:
        numpy.save(fpath, data.numpy())

    f_path = tmp_path / "f.npy"
    u_path = tmp_path / "u.npy"
    x_grid_path = tmp_path / "x_grid.npy"
    f_train_path = tmp_path / "f_train.npy"
    f_test_path = tmp_path / "f_test.npy"
    u_train_path = tmp_path / "u_train.npy"
    u_test_path = tmp_path / "u_test.npy"

    _save(x_grid, x_grid_path)
    _save(f_train, f_train_path)
    _save(f_test, f_test_path)
    _save(u_train, u_train_path)
    _save(u_test, u_test_path)
    _save(f, f_path)
    _save(u, u_path)

    return NpyPaths(f_path, u_path, x_grid_path, f_train_path, f_test_path, u_train_path, u_test_path)


@pytest.fixture
def config_file(tmp_path: Path, npy_data: NpyPaths) -> Generator[Path, None, None]:
    base = Path(__file__).parent / "test-configs"
    base.mkdir(parents=True, exist_ok=True)
    path = base / f"test_two_step_config_{uuid.uuid4().hex}.yaml"
    path.write_text(f"""
# Basic training options -------------------------------------------------------
seed: 42
comm_backend: nccl
log_interval: 100
checkpoint_dir: {tmp_path}

# Data file paths ---------------------------------------------------------------
train_dataset:
  _target_: hspn.train_two_step.DONData.from_npy
  trunk_path: {npy_data.x_grid_path!s}
  branch_path: {npy_data.f_train_path!s}
  output_path: {npy_data.u_train_path!s}
  branch_start: 0.0 # integer offset or float 0.0 <= x <= 1.0 for a percentage
  branch_end: 1.0

# Model config ------------------------------------------------------------------
model:
  _target_: hspn.model.DeepOperatorNet
  branch_dim: 1
  trunk_dim: 3
  branch_config:
    width: 100
    depth: 4
    activation:
      _target_: torch.nn.ELU
  trunk_config:
    width: 100
    depth: 5
    activation:
      _target_: torch.nn.ReLU
  latent_dim: 25
  einsum_pattern: ij,kj->ik

# Trunk training configuration -------------------------------------------------
trunk_config:
  n_epochs: 50
  amp_dtype: bf16 # or fp16 for older hardware (consider grad scaling if fp16)
  grad_scaling: null # force enable/disable or auto-detect. Autodetect enables for fp16, otherwise disabled.
  grad_accum_steps: 1
  grad_clip_norm: null
  batch_size: 100_000

  optimizer:
    _target_: torch.optim.Adam
    lr: 0.001
    weight_decay: 1e-5

  scheduler:
    _target_: torch.optim.lr_scheduler.CosineAnnealingLR
    T_max: ${{trunk_config.n_epochs}}
    eta_min: 1e-6

  sample_config:
    seed: 0
    drop_last: false
    shuffle: false

# Branch training configuration ------------------------------------------------
branch_config:
    n_epochs: 100
    amp_dtype: bf16 # or fp16 for older hardware (consider grad scaling if fp16)
    grad_scaling: null # force enable/disable or auto-detect. Autodetect enables for fp16, otherwise disabled.
    grad_accum_steps: 1
    grad_clip_norm: null
    batch_size: 100_000

    optimizer:
      _target_: torch.optim.Adam
      lr: 0.0005
      weight_decay: 1e-4

    scheduler:
      _target_: torch.optim.lr_scheduler.StepLR
      step_size: 30
      gamma: 0.5

    sample_config:
      seed: 0
      drop_last: false
      shuffle: false

# Optional validation dataloader -----------------------------------------------
# val_dataloader: null
# Ingested by the above model at the end of training, so any dataloader provided
#   must follow the standard DON batching pattern.
val_dataloader:
  _target_: torch.utils.data.DataLoader
  dataset:
    # since we arent running DDP tests this is fine
    _target_: hspn.train_two_step.DONData.from_npy
    trunk_path: {npy_data.x_grid_path!s}
    branch_path: {npy_data.f_train_path!s}
    output_path: {npy_data.u_train_path!s}
    branch_start: 0.0 # integer offset or float 0.0 <= x <= 1.0 for a percentage
    branch_end: 1.0


# Experiment tracking config ---------------------------------------------------
tracker: null

# extras -----------------------------------------------------------------------
extra: null
""")
    yield path
    path.unlink(missing_ok=False)


@pytest.fixture
def minimal_cfg(config_file: Path) -> DictConfig:
    with initialize(version_base=None, config_path="test-configs"):
        cfg = compose(config_name=config_file.stem)
    return cfg


def test_train_config_valid(minimal_cfg):
    _ = Context()
    config = TwoStepTrainConfig.from_cfg(minimal_cfg)
    assert isinstance(config.model, torch.nn.Module)
    assert isinstance(config.train_dataset, DONData)
    assert config.checkpoint_dir.exists() or config.checkpoint_dir.parent.exists()


def test_train_single_epoch(minimal_cfg):
    _ = Context()

    config = TwoStepTrainConfig.from_cfg(minimal_cfg)
    state = TwoStepTrainingState(
        x_grid=config.train_dataset.trunk,
        F_train=config.train_dataset.branch,
        U_train=config.train_dataset.output,
    )

    best_trunk_loss, best_branch_loss = train_two_step(
        config=config,
        state=state,
        device=torch.device("cpu"),
        progress_bar=NullProgress(),
    )

    assert isinstance(best_trunk_loss, float)
    assert isinstance(best_branch_loss, float)
    assert best_trunk_loss > float("-inf")
    assert best_branch_loss > float("-inf")

    assert "two_step_final.pt" in tuple(map(operator.attrgetter("name"), config.checkpoint_dir.iterdir()))
