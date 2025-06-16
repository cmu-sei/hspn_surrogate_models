from pathlib import Path
from typing import Tuple

import numpy
import torch


def mock_data(
    *,
    n_trunk: int = 1024,
    trunk_dim: int = 3,
    n_branch_train: int = 21,
    n_branch_test: int = 21,
    branch_dim: int = 1,
    output_dir: str = "mock-data",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate mock data for two-step training."""
    x_grid = torch.rand(n_trunk, trunk_dim)
    f_train = torch.rand(n_branch_train, branch_dim)
    f_test = torch.rand(n_branch_test, branch_dim)
    u_train = torch.rand(n_branch_train, n_trunk)
    u_test = torch.rand(n_branch_test, n_trunk)

    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    def _save(data: torch.Tensor, fpath: Path) -> None:
        numpy.save(fpath, data.numpy())

    _save(x_grid, output_dir_path / "x_grid.npy")
    _save(f_train, output_dir_path / "f_train.npy")
    _save(f_test, output_dir_path / "f_test.npy")
    _save(u_train, output_dir_path / "u_train.npy")
    _save(u_test, output_dir_path / "u_test.npy")

    return x_grid, f_train, f_test, u_train, u_test


if __name__ == "__main__":
    mock_data()
