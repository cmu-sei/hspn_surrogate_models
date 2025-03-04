import logging
from pathlib import Path
from typing import Optional

import h5py
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, IterableDataset

logger = logging.getLogger(__name__)


class H5Dataset(IterableDataset):
    """Yields batches of an HDF5 dataset deterministically traversing a Cartesian grid."""

    def __init__(
        self,
        file_path: Path,
        branch_batch_size: int = 5,
        trunk_batch_size: int = 10,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.file_path = file_path
        self.branch_batch_size = branch_batch_size
        self.trunk_batch_size = trunk_batch_size
        self.dtype = dtype

        self.is_distributed = dist.is_initialized()
        self.rank = dist.get_rank() if self.is_distributed else 0
        self.world_size = dist.get_world_size() if self.is_distributed else 1

        logger.info(f"Loading HDF5 dataset from {self.file_path}")
        self.file = h5py.File(self.file_path, "r", swmr=True)
        self.branch = self.file["branch"][:]
        trunk = self.file["trunk"]
        output = self.file["output"]

        logger.info(
            f"Loaded Branch={self.branch.shape} Trunk={trunk.shape} Output={output.shape}"
        )

        # Each worker gets a slice of the trunk data
        chunk_size = trunk.shape[0] // self.world_size
        trunk_start = self.rank * chunk_size
        trunk_end = (
            (self.rank + 1) * chunk_size
            if self.rank < self.world_size - 1
            else trunk.shape[0]
        )

        logger.info(
            f"Subsetting rows {trunk_start:,} to {trunk_end:,} ({trunk_end-trunk_start:,}/{trunk.shape[0]:,})"
        )

        trunk_chunk_size = (trunk_end - trunk_start) * trunk.shape[
            1
        ]  # trunk chunk size * n trunk features
        logger.info(
            f"Preloading trunk data for worker {self.rank}/{self.world_size}: {trunk_chunk_size:,} elements in {dtype} "
            f"{trunk_chunk_size * dtype.itemsize / 1e9:.2f}gb"
        )
        self.trunk = self.file["trunk"][trunk_start:trunk_end]

        output_chunk_size = output.shape[0] * (
            trunk_end - trunk_start
        )  # branch size * trunk chunk size
        logger.info(
            f"Preloading output data for worker {self.rank}/{self.world_size}: {output_chunk_size:,} elements in {dtype} "
            f"{output_chunk_size * dtype.itemsize / 1e9:.2f}gb"
        )
        self.output = self.file["output"][
            :, trunk_start:trunk_end
        ]  # (n_branch, n_trunk)

        logger.info(
            f"Branch={self.branch.shape} Trunk={self.trunk.shape} Output={self.output.shape}"
        )

    def __iter__(self):
        """Yields complete batches."""
        n_trunk = self.trunk.shape[0]
        n_branch = self.branch.shape[0]
        end = n_trunk + (
            n_trunk % self.trunk_batch_size
        )  # Account for an incomplete last batch
        for trunk_start in range(0, end, self.trunk_batch_size):
            trunk_end = min(trunk_start + self.trunk_batch_size, n_trunk)
            end = n_branch + (
                n_branch % self.branch_batch_size
            )  # Account for an incomplete last batch
            for branch_start in range(0, end, self.branch_batch_size):
                branch_end = min(branch_start + self.branch_batch_size, n_branch)
                logging.info(
                    f"    yielding branch slice: {branch_start}:{branch_end} trunk slice: {trunk_start}:{trunk_end}"
                )
                yield (
                    torch.tensor(
                        self.branch[branch_start:branch_end], dtype=self.dtype
                    ),
                    torch.tensor(self.trunk[trunk_start:trunk_end], dtype=self.dtype),
                    torch.tensor(
                        self.output[branch_start:branch_end, trunk_start:trunk_end],
                        dtype=self.dtype,
                    ),
                )

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
    from pathlib import Path
    import os

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
