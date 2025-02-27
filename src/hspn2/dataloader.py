import logging
import os
from typing import Tuple

import h5py
import numpy as np
import tensorflow as tf
from omegaconf import DictConfig

logger = logging.getLogger(__name__)


NDArrayx3 = Tuple[np.ndarray, np.ndarray, np.ndarray]


class DataLoader:
    """Data loader for DON training and evaluation.

    Handles:
    - Loading data from files
    - Preprocessing and normalization
    - Batch generation for training
    - Splitting data into training and validation sets

    Attributes:
        config: Data configuration
        train_data: Training data as (branch_input, trunk_input, output)
        val_data: Validation data as (branch_input, trunk_input, output)
        test_data: Test data as (branch_input, trunk_input, output)
    """

    def __init__(self, config: DictConfig) -> None:
        """Initialize data loader.

        Args:
            config: Data configuration
        """
        self.config = config

        self.train_data: NDArrayx3 | None = None
        self.val_data: NDArrayx3 | None = None
        self.test_data: NDArrayx3 | None = None

        self._load_data()

        # FIXME: I forgot about precision, also kind of confused how tf does this
        # self.train_data.astype(np.float32)
        # self.val_data.astype(np.float32)
        # self.test_data.astype(np.float32)

    def _load_data(self) -> None:
        """Load data from files specified in configuration.

        Handles loading data from NPZ files, HDF5, or other formats.
        """
        try:
            train_path = self.config.get("train_path")
            val_path = self.config.get("val_path")
            test_path = self.config.get("test_path")

            if not train_path:
                raise ValueError("Training data path must be provided")

            logger.info(f"Loading training data from {train_path}")
            self.train_data = self._load_data_file(train_path)

            if val_path:
                logger.info(f"Loading validation data from {val_path}")
                self.val_data = self._load_data_file(val_path)
            else:
                logger.info("Creating validation split from training data")
                self.train_data, self.val_data = self._create_validation_split(
                    self.train_data, val_split=self.config.get("val_split", 0.2)
                )

            if test_path:
                logger.info(f"Loading test data from {test_path}")
                self.test_data = self._load_data_file(test_path)

            self._log_data_shapes()

        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise

    @staticmethod
    def _load_data_file(file_path: str) -> NDArrayx3:
        """Load data from a NPZ/HDF5 file.

        Args:
            file_path: Path to the data file

        Returns:
            Tuple of (branch_input, trunk_input, output) arrays

        Raises:
            ValueError: If file format is unsupported or file doesn't contain required data
        """
        # TODO: this is rough I need test data, some hdf5 files would be nice
        file_ext = os.path.splitext(file_path)[1].lower()

        if file_ext == ".npz":
            data = np.load(file_path)

            try:
                branch_input = data["branch_input"]
                trunk_input = data["trunk_input"]
                output = data["output"]
            except KeyError as e:
                raise ValueError(f"NPZ file missing required array: {e}")

        elif file_ext == ".h5" or file_ext == ".hdf5":
            with h5py.File(file_path, "r") as f:
                required_keys = ["branch_input", "trunk_input", "output"]
                missing_keys = [key for key in required_keys if key not in f]

                if missing_keys:
                    raise ValueError(f"HDF5 file missing required datasets: {missing_keys}")

                branch_input = f["branch_input"][:]
                trunk_input = f["trunk_input"][:]
                output = f["output"][:]

        else:
            raise ValueError(f"Unsupported file format: {file_ext}")

        return branch_input, trunk_input, output

    def _create_validation_split(
        self, data: Tuple[np.ndarray, np.ndarray, np.ndarray], val_split: float = 0.2, shuffle: bool = True
    ) -> Tuple[Tuple[np.ndarray, np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Split data into train/val.

        Args:
            data: Tuple of (branch_input, trunk_input, output) arrays
            val_split: Fraction of data to use for validation
            shuffle: Whether to shuffle data before splitting

        Returns:
            Tuple of (train_data, val_data), each a tuple of (branch_input, trunk_input, output)
        """
        branch_input, trunk_input, output = data
        n_samples = len(branch_input)

        indices = np.arange(n_samples)
        if shuffle:
            np.random.shuffle(indices)

        split_idx = int(n_samples * (1 - val_split))

        train_indices = indices[:split_idx]
        val_indices = indices[split_idx:]

        train_data = (branch_input[train_indices], trunk_input[train_indices], output[train_indices])
        val_data = (branch_input[val_indices], trunk_input[val_indices], output[val_indices])

        return train_data, val_data

    def _log_data_shapes(self) -> None:
        """Log shapes for debugging."""
        if self.train_data:
            branch, trunk, output = self.train_data
            logger.info(f"Training data shapes: branch={branch.shape}, trunk={trunk.shape}, output={output.shape}")

        if self.val_data:
            branch, trunk, output = self.val_data
            logger.info(f"Validation data shapes: branch={branch.shape}, trunk={trunk.shape}, output={output.shape}")

        if self.test_data:
            branch, trunk, output = self.test_data
            logger.info(f"Test data shapes: branch={branch.shape}, trunk={trunk.shape}, output={output.shape}")

    def get_shapes(self) -> Tuple[Tuple[int, ...], Tuple[int, ...], Tuple[int, ...]]:
        """Get shapes of the data.

        Returns:
            Tuple of (branch_shape, trunk_shape, output_shape)

        Raises:
            ValueError: If no data is loaded
        """
        if self.train_data is None:
            raise ValueError("No data loaded")

        branch, trunk, output = self.train_data
        return branch.shape, trunk.shape, output.shape

    def get_train_data(self, batch_size=32, shuffle=True):
        """Get training data as a generator with proper batching.

        Only batches branch inputs, keeps trunk inputs fixed.

        Args:
            batch_size: Batch size for branch inputs
            shuffle: Whether to shuffle the data

        Returns:
            Generator yielding (branch_batch, trunk_input, output_batch) tuples
        """
        if self.train_data is None:
            raise ValueError("No training data loaded")

        branch_input, trunk_input, output = self.train_data
        n_samples = len(branch_input)

        indices = np.arange(n_samples)
        if shuffle:
            np.random.shuffle(indices)

        # Batch branch inputs
        for start_idx in range(0, n_samples, batch_size):
            end_idx = min(start_idx + batch_size, n_samples)
            batch_indices = indices[start_idx:end_idx]

            branch_batch = branch_input[batch_indices]

            # Get corresponding output batch
            # Output shape is [n_samples, trunk_points]?
            output_batch = output[batch_indices]

            yield branch_batch, trunk_input, output_batch

    def get_val_data(self) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """Get validation data as tensors.

        Returns:
            Tuple of (branch_input, trunk_input, output) tensors

        Raises:
            ValueError: If no validation data is loaded
        """
        if self.val_data is None:
            raise ValueError("No validation data loaded")

        branch, trunk, output = self.val_data

        branch_tensor = tf.convert_to_tensor(branch, dtype=tf.float32)
        trunk_tensor = tf.convert_to_tensor(trunk, dtype=tf.float32)
        output_tensor = tf.convert_to_tensor(output, dtype=tf.float32)

        return branch_tensor, trunk_tensor, output_tensor

    def get_test_data(self) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """Get test data as tensors.

        Returns:
            Tuple of (branch_input, trunk_input, output) tensors

        Raises:
            ValueError: If no test data is loaded
        """
        if self.test_data is None:
            raise ValueError("No test data loaded")

        branch, trunk, output = self.test_data

        branch_tensor = tf.convert_to_tensor(branch, dtype=tf.float32)
        trunk_tensor = tf.convert_to_tensor(trunk, dtype=tf.float32)
        output_tensor = tf.convert_to_tensor(output, dtype=tf.float32)

        return branch_tensor, trunk_tensor, output_tensor
