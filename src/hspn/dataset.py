import logging
from pathlib import Path
from typing import Optional, Union

import h5py
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, IterableDataset

logger = logging.getLogger(__name__)


class H5Dataset(IterableDataset):
    """Yields batches of an HDF5 dataset deterministically traversing a Cartesian grid."""

    def __init__(
        self,
        file_path: Path,
        branch_batch_size: Optional[int | float] = 100,
        trunk_batch_size: Optional[int | float] = 100_000,
        branch_start: Union[int, float] = 0,
        branch_end: Union[int, float] = 1,
        trunk_start: Union[int, float] = 0,
        trunk_end: Union[int, float] = 1,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.file_path = Path(file_path).resolve()
        self.dtype = dtype

        self.is_distributed = dist.is_initialized()
        self.rank = dist.get_rank() if self.is_distributed else 0
        self.world_size = dist.get_world_size() if self.is_distributed else 1

        logger.info(f"Loading HDF5 dataset from {self.file_path}")
        self.file = h5py.File(self.file_path, "r", swmr=True)

        branch = self.file["branch"]
        assert isinstance(branch, h5py.Dataset)

        trunk = self.file["trunk"]
        assert isinstance(trunk, h5py.Dataset)

        output = self.file["output"]
        assert isinstance(output, h5py.Dataset)

        if branch_batch_size:
            assert branch_batch_size > 0
            if 0 < branch_batch_size < 1:
                assert isinstance(branch_batch_size, int)
                self.branch_batch_size = int(branch_batch_size * branch.shape[0])
            elif branch_batch_size >= 1:
                self.branch_batch_size = int(branch_batch_size)
        else:
            self.branch_batch_size = branch.shape[0]

        if trunk_batch_size:
            assert trunk_batch_size > 0
            if 0 < trunk_batch_size < 1:
                self.trunk_batch_size = int(trunk_batch_size * trunk.shape[0])
            elif trunk_batch_size >= 1:
                assert isinstance(trunk_batch_size, int)
                self.trunk_batch_size = trunk_batch_size
        else:
            self.trunk_batch_size = trunk.shape[0]

        logger.info(f"Branch={branch.shape} Trunk={trunk.shape} Output={output.shape}")

        if branch_start <= 1:
            global_branch_start = int(branch_start * branch.shape[0])
        else:
            global_branch_start = branch_start
            assert isinstance(branch_start, int)
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
            global_trunk_start = trunk_start
            assert isinstance(trunk_start, int)
        if trunk_end <= 1:
            global_trunk_end = int(trunk_end * trunk.shape[0])
        else:
            assert isinstance(trunk_end, int)
            global_trunk_end = trunk_end

        trunk_n = global_trunk_end - global_trunk_start
        self.branch_batch_size = min(self.branch_batch_size, branch_n)
        self.trunk_batch_size = min(self.trunk_batch_size, trunk_n)
        logger.info(
            f"Calculated dataset subset: "
            f"{global_branch_start=} {global_branch_end=} {branch_n=} "
            f"{trunk_start=} {trunk_end=} {global_trunk_start=} {global_trunk_end=} {trunk_n=}"
        )
        assert trunk_n > 0, f"Trunk start and end indices must be valid ({global_trunk_start}, {global_trunk_end})"

        # # Each worker gets a slice of the branch data
        # branch_per_rank = branch_n // self.world_size
        # branch_start = self.rank * branch_per_rank
        # branch_end = (self.rank + 1) * branch_per_rank if self.rank != self.world_size - 1 else branch_n

        # Each worker gets a slice of the trunk data
        chunk_size = trunk_n // self.world_size
        trunk_start = global_trunk_start + self.rank * chunk_size
        trunk_end = trunk_start + ((self.rank + 1) * chunk_size if self.rank < self.world_size - 1 else trunk_n)

        if (branch_end - branch_start) / branch.shape[0]:
            logger.info(
                f"Subsetting along branch {branch_start:_} to {branch_end:_} ({branch_end - branch_start:_}/{branch.shape[0]:_})"
            )
        if (trunk_end - trunk_start) / trunk.shape[0]:
            logger.info(
                f"Subsetting along trunk {trunk_start:_} to {trunk_end:_} ({trunk_end - trunk_start:_}/{trunk.shape[0]:_})"
            )

        def GiB(n, itemsize=dtype.itemsize) -> float:
            return n * itemsize / (1024**3)

        def GiBfmt(x) -> str:
            return f"{x:.2f}GiB"

        # Branch data is not chunked among workers as it is typically quite small
        logger.info(f"Preloading branch data: {branch.size:_} elements in {dtype} {GiBfmt(GiB(branch.size))}")
        self.branch = branch[global_branch_start:global_branch_end, ...]

        trunk_chunk_size = (trunk_end - trunk_start) * trunk.shape[1]  # trunk chunk size * n trunk features
        logger.info(
            f"Preloading trunk data for worker {self.rank + 1}/{self.world_size}: {trunk_chunk_size:_} elements in {trunk.dtype} "
            f"{GiBfmt(GiB(trunk_chunk_size, trunk.dtype.itemsize))}"
        )
        self.trunk = trunk[trunk_start:trunk_end]

        output_chunk_size = output.shape[0] * (trunk_end - trunk_start)  # branch size * trunk chunk size
        logger.info(
            f"Preloading output data for worker {self.rank + 1}/{self.world_size}: {output_chunk_size:_} elements in {output.dtype} "
            f"{GiBfmt(GiB(output_chunk_size, output.dtype.itemsize))}"
        )
        self.output = output[:, trunk_start:trunk_end]  # (n_branch, n_trunk)

        bb_gib = GiB(self.branch_batch_size * self.branch.shape[1], dtype.itemsize)
        tb_gib = GiB(self.trunk_batch_size * self.trunk.shape[1], dtype.itemsize)
        ob_gib = GiB(self.branch_batch_size * self.trunk_batch_size, dtype.itemsize)

        logger.info(f"Using Branch={self.branch.shape} Trunk={self.trunk.shape} Output={self.output.shape}")
        logger.info(
            f"Branch Batch Shape: ({self.branch_batch_size}, {self.branch.shape[1]}) "
            + f"{self.branch_batch_size * self.branch.shape[1]} elements "
            + GiBfmt(bb_gib)
            + f" dtype={dtype} (param: {branch_batch_size=})"
        )
        logger.info(
            f"Trunk Batch Shape: ({self.trunk_batch_size}, {self.trunk.shape[1]}) "
            + f"{self.trunk_batch_size * self.trunk.shape[1]} elements "
            + GiBfmt(tb_gib)
            + f" dtype={dtype} {self.trunk_batch_size / trunk_chunk_size:.1%} of chunk (param: {trunk_batch_size=})"
        )
        logger.info(
            f"Output Batch: ({self.branch_batch_size}, {self.trunk_batch_size}) "
            + f"{self.branch_batch_size * self.trunk_batch_size} elements "
            + GiBfmt(ob_gib)
            + f" dtype={dtype}"
        )
        logger.info(f"Total Batch: {GiBfmt(bb_gib + tb_gib + ob_gib)}")

    def __iter__(self):
        """Yields complete batches."""
        n_trunk = self.trunk.shape[0]
        n_branch = self.branch.shape[0]
        for trunk_start in range(0, n_trunk, self.trunk_batch_size):
            trunk_end = min(trunk_start + self.trunk_batch_size, n_trunk)
            for branch_start in range(0, n_branch, self.branch_batch_size):
                branch_end = min(branch_start + self.branch_batch_size, n_branch)
                if len(self.trunk.shape) == 2:
                    trunk_slice = self.trunk[trunk_start:trunk_end]
                elif len(self.trunk.shape) == 3:
                    trunk_slice = self.trunk[:, trunk_start:trunk_end]
                else:
                    raise ValueError(f"Trunk has invalid ndim ({len(self.trunk.shape)})")
                yield (
                    torch.tensor(self.branch[branch_start:branch_end], dtype=self.dtype),
                    torch.tensor(trunk_slice, dtype=self.dtype),
                    torch.tensor(
                        self.output[branch_start:branch_end, trunk_start:trunk_end],
                        dtype=self.dtype,
                    ),
                )

    def __len__(self):
        """Return the total number of batches that will be yielded."""
        n_trunk = self.trunk.shape[0]
        n_branch = self.branch.shape[0]

        # Number of trunk batches, including partial batches
        trunk_batches = (n_trunk + self.trunk_batch_size - 1) // self.trunk_batch_size

        # Number of branch batches, including partial batches
        branch_batches = (n_branch + self.branch_batch_size - 1) // self.branch_batch_size

        return int(trunk_batches * branch_batches)

    def close(self) -> None:
        """Close the data file."""
        if hasattr(self, "file") and self.file is not None:
            self.file.close()

    def __del__(self) -> None:
        """Ensure file is closed when object is deleted."""
        self.close()


def build_dataloader(
    file_path: Path,
    batch_branch_size: int = 5,
    batch_trunk_size: int = 10,
    num_workers: int = 0,
    prefetch_factor: Optional[int] = None,
    pin_memory: bool = True,
    persistent_workers: bool = False,
) -> DataLoader:
    """Create a DataLoader with specialized batching pattern.

    Args:
        file_path: Path to the HDF5 data file
        batch_branch_size: Number of branch samples per batch
        batch_trunk_size: Number of trunk samples per batch
        num_workers: Number of DataLoader workers
        prefetch_factor: Prefetch factor for DataLoader
        pin_memory: Whether to pin memory for DataLoader
        persistent_workers: Whether to use persistent workers for DataLoader

    Returns:
        Configured DataLoader.
    """
    dataset = H5Dataset(
        file_path=file_path,
        branch_batch_size=batch_branch_size,
        trunk_batch_size=batch_trunk_size,
    )

    return DataLoader(
        dataset=dataset,
        batch_size=None,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )


if __name__ == "__main__":
    # Runs a simple smoke test
    import os
    from pathlib import Path

    data_path = Path(os.environ["HSPN_H5_PATH"]).expanduser()

    dataloader = build_dataloader(
        file_path=data_path,
        batch_branch_size=5,
        batch_trunk_size=10,
    )

    logger.info("\nTesting DataLoader:")
    for i, batch in enumerate(dataloader):
        if i >= 20:
            break

        branch_batch, trunk_batch, output_batch = batch

        logger.info(f"Batch {i + 1}:")
        logger.info(f"Batch {i + 1} Branch shape: {branch_batch.shape}")
        logger.info(f"Batch {i + 1} Trunk shape: {trunk_batch.shape}")
        logger.info(f"Batch {i + 1} Y shape: {output_batch.shape}")

        if branch_batch.shape[0] != 5 or trunk_batch.shape[0] != 10:
            logger.warning("WARNING: Shapes don't match expected (5, 10) pattern")
