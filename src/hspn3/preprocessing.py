# TODO(@mchristiani): fix axis handling, trunk mean uses axis=0 and branch/output use elementwise. ask jasmine if this
# needs to be configurable.
import os
import logging
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime

import numpy as np
import h5py

logger = logging.getLogger(__name__)


def validate_shapes(branch_data: np.ndarray, trunk_data: np.ndarray, output_data: np.ndarray) -> bool:
    """Validate shapes are consistent with what we expect for DON training.

    Args:
        branch_data: Branch input data of shape [n_branch, branch_dim]
        trunk_data: Trunk input data of shape [n_trunk, trunk_dim]
        output_data: Output data of shape [n_branch, n_trunk]

    Returns:
        True if shapes are valid, False otherwise
    """
    logger.debug(f"Branch shape:\t{branch_data.shape}")
    logger.debug(f"Trunk shape:\t{trunk_data.shape}")
    logger.debug(f"Output shape:\t{output_data.shape}")

    valid = True
    if output_data.shape[0] != branch_data.shape[0]:
        logger.error("First dimension of output should be the same as first dimension of branch")
        valid = False

    if output_data.shape[1] != trunk_data.shape[0]:
        logger.error("Second dimension of output should be the same as first dimension of trunk")
        valid = False

    return valid


def normalize_data(
    data: np.ndarray, method: str = "none", axis: Optional[int] = None
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Normalize data using the specified method.

    Args:
        data: Input data to normalize
        method: Normalization method ('minmax', 'standard', 'none')
        axis: Axis along which to normalize (None for global)

    Returns:
        Tuple of (normalized_data, normalization_params)
    """
    norm_params = {"method": method}

    if method.lower() == "none":
        return data, norm_params

    if method.lower() == "minmax":
        data_min = np.min(data, axis=axis, keepdims=True)
        data_max = np.max(data, axis=axis, keepdims=True)
        norm_params["min"] = data_min
        norm_params["max"] = data_max

        denom = data_max - data_min
        denom[denom == 0] = 1.0  # avoid div by 0

        normalized = (data - data_min) / denom

    elif method.lower() == "standard":
        data_mean = np.mean(data, axis=axis, keepdims=True)
        data_std = np.std(data, axis=axis, keepdims=True)
        norm_params["mean"] = data_mean
        norm_params["std"] = data_std

        data_std[data_std == 0] = 1.0  # avoid div by 0

        normalized = (data - data_mean) / data_std
    else:
        raise ValueError(f"Unsupported normalization method: {method}")

    return normalized, norm_params


def load_npy_files(
    data_dir: str, branch_files: List[str], trunk_files: List[str], output_files: List[str]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load data from numpy files and combine them if multiple files are provided.

    Args:
        data_dir: Directory containing the data files
        branch_files: List of branch data files
        trunk_files: List of trunk data files
        output_files: List of output data files

    Returns:
        Tuple of (branch_data, trunk_data, output_data)
    """
    if not (len(branch_files) == len(trunk_files) == len(output_files)):
        raise ValueError("Number of branch, trunk, and output files must be the same")

    # Got a single file
    if len(branch_files) == 1:
        branch_path = os.path.join(data_dir, branch_files[0])
        trunk_path = os.path.join(data_dir, trunk_files[0])
        output_path = os.path.join(data_dir, output_files[0])

        logger.info(f"Loading single file set:")
        logger.info(f"  Branch: {branch_path}")
        logger.info(f"  Trunk: {trunk_path}")
        logger.info(f"  Output: {output_path}")

        branch_data = np.load(branch_path, allow_pickle=True)
        trunk_data = np.load(trunk_path, allow_pickle=True)
        output_data = np.load(output_path, allow_pickle=True)

        if not validate_shapes(branch_data, trunk_data, output_data):
            raise ValueError("Data shape validation failed")

        return branch_data, trunk_data, output_data

    # Got multiple files
    logger.info(f"Loading {len(branch_files)} file sets")

    all_branch = []
    all_trunk = []
    all_output = []

    for i, (branch_file, trunk_file, output_file) in enumerate(zip(branch_files, trunk_files, output_files)):
        branch_path = os.path.join(data_dir, branch_file)
        trunk_path = os.path.join(data_dir, trunk_file)
        output_path = os.path.join(data_dir, output_file)

        logger.info(f"Loading set {i + 1}:")
        logger.info(f"  Branch: {branch_path}")
        logger.info(f"  Trunk: {trunk_path}")
        logger.info(f"  Output: {output_path}")

        branch = np.load(branch_path, allow_pickle=True)
        trunk = np.load(trunk_path, allow_pickle=True)
        output = np.load(output_path, allow_pickle=True)

        if not validate_shapes(branch, trunk, output):
            logger.warning(f"Skipping invalid data set {i + 1}")
            continue

        all_branch.append(branch)
        all_trunk.append(trunk)
        all_output.append(output)

    if not all_branch:
        raise ValueError("No valid data sets found")

    branch_data = np.concatenate(all_branch, axis=0)
    trunk_data = np.concatenate(all_trunk, axis=0)

    # Combined outupt needs to be [total_branch_size, total_trunk_size]
    total_branch_size = branch_data.shape[0]
    total_trunk_size = trunk_data.shape[0]

    output_data = np.zeros((total_branch_size, total_trunk_size))

    # Fill vals
    b_offset = 0
    t_offset = 0

    for i in range(len(all_branch)):
        b_size = all_branch[i].shape[0]
        t_size = all_trunk[i].shape[0]

        output_data[b_offset : b_offset + b_size, t_offset : t_offset + t_size] = all_output[i]

        b_offset += b_size
        t_offset += t_size

    logger.info(f"Combined data shapes:")
    logger.info(f"  Branch: {branch_data.shape}")
    logger.info(f"  Trunk: {trunk_data.shape}")
    logger.info(f"  Output: {output_data.shape}")

    return branch_data, trunk_data, output_data


def create_hdf5_file(
    output_path: str,
    branch_data: np.ndarray,
    trunk_data: np.ndarray,
    output_data: np.ndarray,
    norm_method: str = "none",
    branch_norm_axis: Optional[int] = None,
    trunk_norm_axis: Optional[int] = None,
    output_norm_axis: Optional[int] = None,
    metadata: Optional[Dict[str, Any]] = None,
    compression: Optional[str] = None,
) -> None:
    """Create HDF5 file from numpy data with optional normalization.

    Args:
        output_path: Path to save the HDF5 file
        branch_data: Branch data array
        trunk_data: Trunk data array
        output_data: Output data array
        norm_method: Normalization method, applies to all data
        branch_norm_axis: Axis for branch normalization
        trunk_norm_axis: Axis for trunk normalization
        output_norm_axis: Axis for output normalization
        metadata: Additional metadata to store, containers will be converted to string
        compression: Compression method for HDF5 file, e.g. "gzip", "lzf", or None

    Raises:
        ValueError: if shapes of the input data are not what we expect for DON training
    """
    if not validate_shapes(branch_data, trunk_data, output_data):
        raise ValueError("Data shape validation failed")

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    # Preprocess
    if norm_method != "none":
        logger.info(f"Normalizing branch data with method {norm_method}")
        branch_data, branch_params = normalize_data(branch_data, norm_method, branch_norm_axis)
        logger.info(f"Normalizing trunk data with method {norm_method}")
        trunk_data, trunk_params = normalize_data(trunk_data, norm_method, trunk_norm_axis)
        logger.info(f"Normalizing output data with method {norm_method}")
        output_data, output_params = normalize_data(output_data, norm_method, output_norm_axis)
        norm_params = {"branch": branch_params, "trunk": trunk_params, "output": output_params}
    else:
        norm_params = {}

    # Save output
    logger.info(f"Creating HDF5 file at {output_path}")

    with h5py.File(output_path, "w") as f:
        logger.info("Adding data for branch")
        f.create_dataset("branch", data=branch_data, compression=compression)

        logger.info("Adding data for trunk")
        f.create_dataset("trunk", shape=trunk_data.shape, data=trunk_data, compression=compression, chunks=True)

        logger.info("Adding data for y")
        f.create_dataset("y", shape=output_data.shape, data=output_data, compression=compression, chunks=True)
        logger.info("Adding metadata")

        # Add metadata for normalization params (e.g. min/max/mu/std, if applicable)
        norm_group = f.create_group("normalization")

        for key, params in norm_params.items():
            param_group = norm_group.create_group(key)

            for param_key, param_value in params.items():
                if isinstance(param_value, np.ndarray):
                    param_group.create_dataset(param_key, data=param_value, compression=compression)
                else:
                    param_group.attrs[param_key] = param_value

        # Add user metadata
        if metadata:
            meta_group = f.create_group("metadata")

            for key, value in metadata.items():
                if isinstance(value, (dict, list)):
                    meta_group.attrs[key] = str(value)
                else:
                    meta_group.attrs[key] = value

        # Add attrs with shape info and creation time
        f.attrs["branch_size"] = branch_data.shape[0]
        f.attrs["branch_dim"] = branch_data.shape[1] if len(branch_data.shape) > 1 else 1
        f.attrs["trunk_size"] = trunk_data.shape[0]
        f.attrs["trunk_dim"] = trunk_data.shape[1] if len(trunk_data.shape) > 1 else 1
        f.attrs["creation_time"] = np.string_(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    logger.info("HDF5 file created successfully")
