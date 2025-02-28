"""Simple script to convert data from numpy files to a single HDF5 file."""

import os
import argparse
import logging
from typing import Union, List, Optional, Dict, Any

from hspn3.preprocessing import load_npy_files, create_hdf5_file

logger = logging.getLogger(__name__)


def convert_npy_to_hdf5(
        data_dir: str,
        output_path: str,
        branch_files: Union[str, List[str]],
        trunk_files: Union[str, List[str]],
        output_files: Union[str, List[str]],
        norm_method: str = "none",
        branch_norm_axis: Optional[int] = None,
        trunk_norm_axis: int = 0,
        output_norm_axis: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
        compression: str = "gzip",
) -> None:
    """Convert numpy files to a single HDF5 file with optional normalization.

    This function handles the entire conversion process.

    Args:
        data_dir: Directory containing numpy files
        output_path: Path to save the HDF5 file
        branch_files: Single filename or list of filenames for branch data
        trunk_files: Single filename or list of filenames for trunk data. (Default: 0)
        output_files: Single filename or list of filenames for output data
        norm_method: Normalization method, applies to all data
        branch_norm_axis: Axis for branch normalization
        trunk_norm_axis: Axis for trunk normalization
        output_norm_axis: Axis for output normalization
        metadata: Additional metadata to store
        compression: Compression method for HDF5 file
    """
    # Make sure we have lists
    branch_data, trunk_data, output_data = load_npy_files(data_dir, list(branch_files), list(trunk_files), list(output_files))
    create_hdf5_file(
        output_path,
        branch_data,
        trunk_data,
        output_data,
        norm_method,
        branch_norm_axis,
        trunk_norm_axis,
        output_norm_axis,
        metadata,
        compression,
    )


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Convert DON data from numpy files to HDF5")

    # Inputs
    parser.add_argument("--data-dir", type=str, required=True, help="Directory containing numpy files")
    parser.add_argument(
        "--branch-files", type=str, nargs="+", required=True, help="[in] Numpy files containing branch data"
    )
    parser.add_argument(
        "--trunk-files", type=str, nargs="+", required=True, help="[in] Numpy files containing trunk data"
    )
    parser.add_argument(
        "--output-files", type=str, nargs="+", required=True, help="[in] Numpy files containing output data"
    )

    # Outputs
    parser.add_argument("--h5-path", type=str, required=True, help="[out] Path to save the HDF5 file")

    # Preprocessing
    parser.add_argument(
        "--norm-method",
        type=str,
        default="none",
        choices=["none", "minmax", "standard"],
        help="Normalization method, applies to all data.",
    )

    # Output opts
    parser.add_argument(
        "--compression",
        required=False,
        type=lambda x: None if x.lower() == "none" else x,
        default="gzip",
        choices=[None, "gzip", "lzf"],
        help="Compression method for HDF5 file. Has a significant impact on processing time, gzip can be quite slow "
             "while lzf is faster but less efficient. "
    )
    parser.add_argument("--force", action="store_true", help="Force overwrite if HDF5 file exists")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if os.path.exists(args.h5_path) and not args.force:
        logger.error(f"HDF5 file already exists at {args.h5_path}. Use --force to overwrite.")
        exit(1)

    metadata = {
        "data_source": {
            "data_dir": args.data_dir,
            "branch_files": args.branch_files,
            "trunk_files": args.trunk_files,
            "output_files": args.output_files,
        },
        "preprocessing": {
            "norm_method": args.norm_method,
        },
    }

    logger.info("Starting conversion from numpy files to HDF5")

    convert_npy_to_hdf5(
        data_dir=args.data_dir,
        output_path=args.h5_path,
        branch_files=args.branch_files,
        trunk_files=args.trunk_files,
        output_files=args.output_files,
        norm_method=args.norm_method,
        metadata=metadata,
        compression=args.compression,
    )

    logger.info(f"Conversion done. HDF5 file saved at {args.h5_path}")
