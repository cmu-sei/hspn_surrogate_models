"""Standalone script for converting DON training data from numpy files to HDF5 or NetCDF format.

Chose to remove control of compression (because our data is not that big and it really slows everything) and control
over chunking. Chunking might be added back, but can be modified other ways and made this script more complex.

NB: NetCDF produces some big files.

Usage:
    python -m hspn.prepare --help
    python -m hspn.prepare format=HDF5 data_dir=./data
    python -m hspn.prepare format=HDF5 data_dir=./data  branch_files=[branch.npy]
    python -m hspn.prepare format=NETCDF data_dir=./data branch_files=[branch.npy]
    python -m hspn.prepare format=HDF5 data_dir=./data branch_files=[f_total.npy] trunk_files=[xyz.npy] output_files=[y_total.npy]

Note: Default config is prepare.yaml which can also be edited directly if desired.
"""

import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import List, Optional, TypedDict, Union, Tuple

import h5py
import hydra
import netCDF4 as nc  # noqa: N813
import numpy as np
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)


class FormatType(str, Enum):
    HDF5 = "hdf5"
    NETCDF = "netcdf"


class NormMethod(str, Enum):
    NONE = "none"
    MINMAX = "minmax"
    STANDARD = "standard"


@dataclass
class NormalizationConfig:
    _target_: str = "hspn.prepare.NormalizationConfig"
    method: NormMethod = NormMethod.NONE
    axis: Optional[int] = None


@dataclass
class ConversionConfig:
    _target_: str = "hspn.prepare.ConversionConfig"
    format: FormatType = FormatType.HDF5
    data_dir: str = "./data"
    output_path: str = ""
    force: bool = False
    branch_files: List[str] = field(default_factory=lambda: ["branch.npy"])
    trunk_files: List[str] = field(default_factory=lambda: ["trunk.npy"])
    output_files: List[str] = field(default_factory=lambda: ["output.npy"])

    # Normalization settings
    branch_normalization: NormalizationConfig = field(default_factory=NormalizationConfig)
    trunk_normalization: NormalizationConfig = field(default_factory=NormalizationConfig)
    output_normalization: NormalizationConfig = field(default_factory=NormalizationConfig)


def validate_shapes(branch_data: np.ndarray, trunk_data: np.ndarray, output_data: np.ndarray) -> bool:
    """Validate shapes are consistent with what we expect for DON training."""
    logger.info(f"Branch shape:\t{branch_data.shape}")
    logger.info(f"Trunk shape:\t{trunk_data.shape}")
    logger.info(f"Output shape:\t{output_data.shape}")

    if output_data.shape[0] != branch_data.shape[0]:
        logger.error("First dimension of output should be the same as first dimension of branch")
        return False

    if output_data.shape[1] != trunk_data.shape[0]:
        logger.error("Second dimension of output should be the same as first dimension of trunk")
        return False

    return True


class NoneNormResult(TypedDict):
    data: np.ndarray


class MinMaxNormResult(TypedDict):
    data: np.ndarray
    min: np.ndarray
    max: np.ndarray


class StandardNormResult(TypedDict):
    data: np.ndarray
    mean: np.ndarray
    std: np.ndarray


def normalize_data(
    data: np.ndarray, config: NormalizationConfig, name: str = ""
) -> Union[NoneNormResult, MinMaxNormResult, StandardNormResult]:
    """Normalize data using the specified method."""
    if config.method == NormMethod.NONE:
        logger.info(f"Skipping normalization for {name}")
        return NoneNormResult(data=data)  # NOT a copy

    logger.info(f"Normalizing {name} data with method {config.method} along axis {config.axis}")

    if config.method == NormMethod.MINMAX:
        data_min: np.ndarray = np.min(data, axis=config.axis, keepdims=True)
        data_max: np.ndarray = np.max(data, axis=config.axis, keepdims=True)
        denom: np.ndarray = data_max - data_min
        denom[denom == 0] = 1.0  # avoid div by 0
        normalized = (data - data_min) / denom
        return MinMaxNormResult(data=normalized, min=data_min, max=data_max)

    elif config.method == NormMethod.STANDARD:
        data_mean = np.mean(data, axis=config.axis, keepdims=True)
        data_std = np.std(data, axis=config.axis, keepdims=True)
        data_std[data_std == 0] = 1.0  # avoid div by 0
        normalized = (data - data_mean) / data_std
        return StandardNormResult(data=normalized, mean=data_mean, std=data_std)

    raise ValueError(f"Unsupported normalization method: {config.method}")


def load_npy_files(
    data_dir: str, branch_files: List[str], trunk_files: List[str], output_files: List[str]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load data from numpy files and combine them if multiple files are provided."""
    if not (len(branch_files) == len(trunk_files) == len(output_files)):
        raise ValueError("Number of branch, trunk, and output files must be the same")

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

    # Combined output needs to be [total_branch_size, total_trunk_size]
    total_branch_size = branch_data.shape[0]
    total_trunk_size = trunk_data.shape[0]

    output_data = np.zeros((total_branch_size, total_trunk_size))

    # Fill values
    b_offset = 0
    t_offset = 0

    for i in range(len(all_branch)):
        b_size = all_branch[i].shape[0]
        t_size = all_trunk[i].shape[0]

        output_data[b_offset : b_offset + b_size, t_offset : t_offset + t_size] = all_output[i]

        b_offset += b_size
        t_offset += t_size

    logger.info("Combined data shapes:")
    logger.info(f"  Branch: {branch_data.shape}")
    logger.info(f"  Trunk: {trunk_data.shape}")
    logger.info(f"  Output: {output_data.shape}")

    return branch_data, trunk_data, output_data


def create_hdf5_file(
    output_path: str,
    branch_data: np.ndarray,
    trunk_data: np.ndarray,
    output_data: np.ndarray,
    config: ConversionConfig,
) -> None:
    """Create HDF5 file from numpy data with optional normalization."""
    if not validate_shapes(branch_data, trunk_data, output_data):
        raise ValueError("Data shape validation failed")

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    # Process data
    branch_proc = normalize_data(branch_data, config.branch_normalization, "branch")
    trunk_proc = normalize_data(trunk_data, config.trunk_normalization, "trunk")
    output_proc = normalize_data(output_data, config.output_normalization, "output")

    # Get dimensions
    n_branch, branch_dim = (
        branch_proc["data"].shape if len(branch_proc["data"].shape) > 1 else (branch_proc["data"].shape[0], 1)
    )
    n_trunk, trunk_dim = (
        trunk_proc["data"].shape if len(trunk_proc["data"].shape) > 1 else (trunk_proc["data"].shape[0], 1)
    )

    # Create HDF5 file
    logger.info(f"Creating HDF5 file at {output_path}")
    with h5py.File(output_path, "w") as f:
        logger.info("Adding branch data")
        f.create_dataset("branch", data=branch_proc["data"])

        logger.info("Adding trunk data")
        f.create_dataset("trunk", data=trunk_proc["data"])

        logger.info("Adding output data")
        f.create_dataset("output", data=output_proc["data"])

        # Add normalization data (method, axis, min/max/etc)
        norm_group = f.create_group("normalization")
        for name, result in [("branch", branch_proc), ("trunk", trunk_proc), ("output", output_proc)]:
            # Skip if no normalization was applied
            norm_config = (
                config.branch_normalization
                if name == "branch"
                else config.trunk_normalization
                if name == "trunk"
                else config.output_normalization
            )

            # Create a group for this component
            component_group = norm_group.create_group(name)

            # Add method and axis info
            component_group.attrs["method"] = norm_config.method.value
            component_group.attrs["axis"] = -1 if norm_config.axis is None else norm_config.axis

            # Save all parameters except the normalized data itself (e.g. max, min)
            for key, value in result.items():
                if key != "data" and isinstance(value, (np.ndarray, float, int)):
                    if isinstance(value, np.ndarray):
                        component_group.create_dataset(key, data=value)
                    else:
                        component_group.attrs[key] = value

        # Add metadata
        meta_group = f.create_group("metadata")
        meta_group.attrs["creation_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        meta_group.attrs["data_dir"] = config.data_dir
        meta_group.attrs["branch_files"] = str(config.branch_files)
        meta_group.attrs["trunk_files"] = str(config.trunk_files)
        meta_group.attrs["output_files"] = str(config.output_files)

        f.attrs["branch_size"] = n_branch
        f.attrs["branch_dim"] = branch_dim
        f.attrs["trunk_size"] = n_trunk
        f.attrs["trunk_dim"] = trunk_dim
        f.attrs["creation_time"] = np.string_(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    logger.info("HDF5 file created.")


def create_netcdf_file(
    output_path: str,
    branch_data: np.ndarray,
    trunk_data: np.ndarray,
    output_data: np.ndarray,
    config: ConversionConfig,
) -> None:
    """Create NetCDF file from numpy data with optional normalization."""
    if not validate_shapes(branch_data, trunk_data, output_data):
        raise ValueError("Data shape validation failed")

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    # Process data
    branch_proc = normalize_data(branch_data, config.branch_normalization, "branch")
    trunk_proc = normalize_data(trunk_data, config.trunk_normalization, "trunk")
    output_proc = normalize_data(output_data, config.output_normalization, "output")

    # Get dimensions
    n_branch, branch_dim = (
        branch_proc["data"].shape if len(branch_proc["data"].shape) > 1 else (branch_proc["data"].shape[0], 1)
    )
    n_trunk, trunk_dim = (
        trunk_proc["data"].shape if len(trunk_proc["data"].shape) > 1 else (trunk_proc["data"].shape[0], 1)
    )

    logger.info(f"Creating NetCDF file at {output_path}")

    with nc.Dataset(output_path, "w", format="NETCDF4") as ncfile:
        ncfile.createDimension("n_branch", n_branch)
        ncfile.createDimension("branch_dim", branch_dim)
        ncfile.createDimension("n_trunk", n_trunk)
        ncfile.createDimension("trunk_dim", trunk_dim)
        metadata_group = ncfile.createGroup("metadata")

        # Create variables
        branch_var = ncfile.createVariable(
            "branch",
            "f4",
            ("n_branch", "branch_dim"),
        )
        branch_var[:] = branch_proc["data"]
        branch_var.long_name = "Branch parameters/functions"

        trunk_var = ncfile.createVariable(
            "trunk",
            "f4",
            ("n_trunk", "trunk_dim"),
        )
        trunk_var[:] = trunk_proc["data"]
        trunk_var.long_name = "Trunk spatial points"

        output_var = ncfile.createVariable(
            "output",
            "f4",
            ("n_branch", "n_trunk"),
        )
        output_var[:] = output_proc["data"]
        output_var.long_name = "Output values"
        # NOTE: I am not actually sure how the metadata should be structured, specifically because we have nd-arrays
        #  its not just all scalars we can attach as attrs. This appears to be correct but seems inconvenient to read.
        # Add normalization info as simple attributes on the data variables
        for var_name, result, nrm_config in [
            ("branch", branch_proc, config.branch_normalization),
            ("trunk", trunk_proc, config.trunk_normalization),
            ("output", output_proc, config.output_normalization),
        ]:
            var = ncfile.variables[var_name]
            var.normalization_method = nrm_config.method.value
            var.normalization_axis = -1 if nrm_config.axis is None else nrm_config.axis

            # Add any normalization parameters directly to the variable
            for key, value in result.items():
                if key != "data":
                    if isinstance(value, np.ndarray):
                        norm_var = metadata_group.createVariable(f"{var_name}_{key}", "f4", var.dimensions)
                        norm_var[:] = value
                    else:
                        logger.warning(f"Got an unexpected dtype in normalization parameters. {key}={type(value)}")
        # Add metadata
        metadata_group.creation_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        metadata_group.branch_size = n_branch
        metadata_group.branch_dim = branch_dim
        metadata_group.trunk_size = n_trunk
        metadata_group.trunk_dim = trunk_dim
        metadata_group.data_dir = config.data_dir
        metadata_group.branch_files = str(config.branch_files)
        metadata_group.trunk_files = str(config.trunk_files)
        metadata_group.output_files = str(config.output_files)

    logger.info("NetCDF file created.")


ConfigStore.instance().store("prepare_spec", ConversionConfig)


@hydra.main(config_path="pkg://hspn.conf", config_name="prepare", version_base=None)
def main(cfg: DictConfig) -> None:
    """Main entry point for data preparation."""
    OmegaConf.resolve(cfg)
    logger.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")
    config: ConversionConfig = hydra.utils.instantiate(cfg)

    if not config.output_path:
        ext = ".h5" if config.format == FormatType.HDF5 else ".nc"
        config.output_path = os.path.join(config.data_dir, f"don_dataset{ext}")

    if os.path.exists(config.output_path) and not config.force:
        logger.error(f"Output file already exists at {config.output_path}. Use force=true to overwrite.")
        return

    branch_data, trunk_data, output_data = load_npy_files(
        config.data_dir, config.branch_files, config.trunk_files, config.output_files
    )

    if config.format == FormatType.HDF5:
        create_hdf5_file(
            config.output_path,
            branch_data,
            trunk_data,
            output_data,
            config,
        )
    else:  # NetCDF
        create_netcdf_file(
            config.output_path,
            branch_data,
            trunk_data,
            output_data,
            config,
        )

    logger.info(f"Data preparation complete. File saved to {config.output_path}")


if __name__ == "__main__":
    main()
