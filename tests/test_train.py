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
from typing import Generator

import h5py
import numpy as np
import pytest
import torch
from hydra import compose, initialize
from omegaconf import DictConfig
from torch.amp.grad_scaler import GradScaler

from hspn.context import Context
from hspn.train import TrainConfig, train
from hspn.train_utils import NullProgress


@pytest.fixture
def dummy_h5_file(tmp_path: Path) -> Path:
    path = tmp_path / "dummy.h5"
    with h5py.File(path, "w") as f:
        f.create_dataset("branch", data=np.random.rand(10, 1).astype("float32"))
        f.create_dataset("trunk", data=np.random.rand(20, 3).astype("float32"))
        f.create_dataset("output", data=np.random.rand(10, 20).astype("float32"))
    return path


@pytest.fixture
def config_file() -> Generator[Path, None, None]:
    base = Path(__file__).parent / "test-configs"
    base.mkdir(parents=True, exist_ok=True)
    path = base / f"test_train_config_{uuid.uuid4().hex}.yaml"
    path.write_text("""
seed: 0
n_epochs: 1
comm_backend: gloo
log_interval: 1
checkpoint_dir: placeholder
enable_amp: false
grad_accum_steps: 1
enable_grad_scaling: false
grad_clip_norm: null

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

dataloader:
  _target_: torch.utils.data.DataLoader
  dataset:
    _target_: hspn.dataset.H5Dataset
    file_path: ./dummy.h5
    branch_batch_size: 2
    trunk_batch_size: 3
  batch_size: null
  shuffle: false
  num_workers: 0
  pin_memory: false

val_dataloader: ${dataloader}
optimizer:
  _target_: torch.optim.SGD
  lr: 0.0001
scheduler: null
tracker: null
""")
    yield path
    path.unlink(missing_ok=False)


@pytest.fixture
def minimal_cfg(tmp_path: Path, dummy_h5_file: Path, config_file: Path) -> DictConfig:
    with initialize(version_base=None, config_path="test-configs"):
        cfg = compose(config_name=config_file.stem)
    cfg.dataloader.dataset.file_path = str(dummy_h5_file)
    cfg.val_dataloader.dataset.file_path = str(dummy_h5_file)
    cfg.checkpoint_dir = str(tmp_path / "ckpt")
    return cfg


def test_train_config_valid(minimal_cfg):
    _ = Context()
    config = TrainConfig.from_cfg(minimal_cfg)
    assert isinstance(config.model, torch.nn.Module)
    assert isinstance(config.dataloader, torch.utils.data.DataLoader)
    assert isinstance(config.optimizer, torch.optim.Optimizer)
    assert config.checkpoint_dir.exists() or config.checkpoint_dir.parent.exists()


def test_no_context_raises(minimal_cfg):
    try:
        Context._instance = None
    finally:
        with pytest.raises(RuntimeError):
            _ = TrainConfig.from_cfg(minimal_cfg)


def test_train_single_epoch(minimal_cfg):
    _ = Context()

    config = TrainConfig.from_cfg(minimal_cfg)
    model = config.model.to("cpu").train()
    optimizer = config.optimizer
    dataloader = config.dataloader
    val_dataloader = config.val_dataloader
    scaler = GradScaler(enabled=config.enable_grad_scaling)
    scheduler = config.scheduler

    best_val_loss, best_epoch, global_step = train(
        model=model,
        dataloader=dataloader,
        val_dataloader=val_dataloader,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        tracker=None,
        checkpoint_dir=config.checkpoint_dir,
        n_epochs=config.n_epochs,
        starting_epoch=1,
        starting_global_step=0,
        starting_best_val_loss=float("inf"),
        enable_amp=config.enable_amp,
        grad_accum_steps=config.grad_accum_steps,
        grad_clip_norm=config.grad_clip_norm,
        log_interval=config.log_interval,
        device=torch.device("cpu"),
        progress_bar=NullProgress(),
    )

    assert isinstance(best_val_loss, float)
    assert isinstance(best_epoch, int)
    assert best_val_loss > float("-inf")
    assert best_epoch == 1
    assert global_step > 0

    assert "best_model.pt" in tuple(map(operator.attrgetter("name"), config.checkpoint_dir.iterdir()))
