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

from pathlib import Path
from typing import Tuple

import h5py
import numpy as np
import pytest
import torch
from hspn.dataset import H5Dataset, build_dataloader


NUM_BRANCH_SAMPLES = 20
NUM_TRUNK_SAMPLES = 40
BRANCH_FEATURES = 3
TRUNK_FEATURES = 2
BATCH_SIZE_BRANCH = 5
BATCH_SIZE_TRUNK = 10
OUTPUT_SHAPE = (NUM_BRANCH_SAMPLES, NUM_TRUNK_SAMPLES)


@pytest.fixture
def dummy_h5_file(tmp_path: Path) -> Path:
    """Creates a temporary HDF5 file with dummy 'branch', 'trunk', and 'output' datasets."""
    path = tmp_path / "dummy.h5"
    with h5py.File(path, "w") as f:
        f.create_dataset("branch", data=np.random.rand(NUM_BRANCH_SAMPLES, BRANCH_FEATURES).astype("float32"))
        f.create_dataset("trunk", data=np.random.rand(NUM_TRUNK_SAMPLES, TRUNK_FEATURES).astype("float32"))
        f.create_dataset("output", data=np.random.rand(*OUTPUT_SHAPE).astype("float32"))

    return path


def test_dataset_shapes_and_types(dummy_h5_file: Path) -> None:
    with H5Dataset(dummy_h5_file, branch_batch_size=BATCH_SIZE_BRANCH, trunk_batch_size=BATCH_SIZE_TRUNK) as dataset:
        for branch, trunk, output in dataset:
            assert isinstance(branch, torch.Tensor)
            assert isinstance(trunk, torch.Tensor)
            assert isinstance(output, torch.Tensor)
            assert branch.shape[1] == BRANCH_FEATURES
            assert trunk.shape[1] == TRUNK_FEATURES
            assert output.shape == (branch.shape[0], trunk.shape[0])


@pytest.mark.parametrize(
    "branch_bs, trunk_bs",
    [(5, 5), (5, 10)],
    ids=["5x5", "5x10"],
)
def test_len_matches_expected_batches(
    dummy_h5_file: Path,
    branch_bs: int,
    trunk_bs: int,
) -> None:
    with H5Dataset(dummy_h5_file, branch_batch_size=branch_bs, trunk_batch_size=trunk_bs) as dataset:
        n_batches = 0
        branch_total = 0
        trunk_total = 0
        output_total = 0
        for branch, trunk, output in dataset:
            branch_total += branch.shape[0]
            trunk_total += trunk.shape[0]
            output_total += output.shape[0]
            n_batches += 1
        assert n_batches == len(dataset)


@pytest.mark.parametrize(
    "branch_bs, trunk_bs, expected_branch_batch_shape, expected_trunk_batch_shape, expected_output_batch_shape",
    [
        (5, 10, (5, BRANCH_FEATURES), (10, TRUNK_FEATURES), (5, 10)),
        (10, 10, (10, BRANCH_FEATURES), (10, TRUNK_FEATURES), (10, 10)),
    ],
)
def test_dataloader_batch_shapes(
    dummy_h5_file: Path,
    branch_bs: int,
    trunk_bs: int,
    expected_branch_batch_shape: Tuple[int, int],
    expected_trunk_batch_shape: Tuple[int, int],
    expected_output_batch_shape: Tuple[int, int],
) -> None:
    dataloader = build_dataloader(dummy_h5_file, batch_branch_size=branch_bs, batch_trunk_size=trunk_bs)
    for branch_batch, trunk_batch, output_batch in dataloader:
        assert tuple(trunk_batch.shape) == expected_trunk_batch_shape
        assert tuple(branch_batch.shape) == expected_branch_batch_shape
        assert tuple(output_batch.shape) == expected_output_batch_shape


@pytest.mark.parametrize(
    "world_size, rank, branch_bs, trunk_bs, expected_branch_batch_shape, expected_trunk_batch_shape, expected_output_batch_shape",
    [
        (2, 0, 5, 10, (5, BRANCH_FEATURES), (10, TRUNK_FEATURES), (5, 10)),
        (2, 1, 5, 10, (5, BRANCH_FEATURES), (10, TRUNK_FEATURES), (5, 10)),
    ],
)
def test_partitioning_logic(
    monkeypatch,
    dummy_h5_file: Path,
    world_size: int,
    rank: int,
    branch_bs: int,
    trunk_bs: int,
    expected_branch_batch_shape: Tuple[int, int],
    expected_trunk_batch_shape: Tuple[int, int],
    expected_output_batch_shape: Tuple[int, int],
) -> None:
    monkeypatch.setattr("torch.distributed.is_initialized", lambda: True)
    monkeypatch.setattr("torch.distributed.get_rank", lambda: rank)
    monkeypatch.setattr("torch.distributed.get_world_size", lambda: world_size)

    with H5Dataset(dummy_h5_file, branch_batch_size=branch_bs, trunk_batch_size=trunk_bs) as dataset:
        branch_batch, trunk_batch, output_batch = next(iter(dataset))
        assert tuple(trunk_batch.shape) == expected_trunk_batch_shape
        assert tuple(branch_batch.shape) == expected_branch_batch_shape
        assert tuple(output_batch.shape) == expected_output_batch_shape


def test_invalid_batch_size_raises(dummy_h5_file: Path):
    with pytest.raises(ValueError):
        _ = H5Dataset(dummy_h5_file, branch_batch_size=0, trunk_batch_size=10)

    with pytest.raises(ValueError):
        _ = H5Dataset(dummy_h5_file, branch_batch_size=10, trunk_batch_size="10")  # type: ignore

    with pytest.raises(ValueError):
        _ = H5Dataset(dummy_h5_file, branch_batch_size="10", trunk_batch_size=10)  # type: ignore

    with pytest.raises(ValueError):
        _ = H5Dataset(dummy_h5_file, branch_batch_size=-1, trunk_batch_size=10)  # type: ignore
