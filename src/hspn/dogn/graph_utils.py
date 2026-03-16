import os

from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch

from loguru import logger
from numpy.lib.recfunctions import \
    structured_to_unstructured as to_unstructured

local_rank = int(os.environ.get('LOCAL_RANK', 0))

def load_node_data(
    node_file: Path, dimensions: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    """
    Load node positions, types, and IDs from file.

    Args:
        node_file: Path to node data file
        dimensions: 2 or 3 for 2D or 3D data

    Returns:
        - node_positions: (n_nodes, dimensions)
        - node_types: (n_nodes, 1)
        - node_ids: (n_nodes, 1)
    """
    data = np.load(node_file)

    # Extract position coordinates
    if dimensions == 2:
        pos_fields = ["X", "Y"]
    elif dimensions == 3:
        pos_fields = ["X", "Y", "Z"]
    else:
        raise ValueError(f"Invalid dimensions: {dimensions}")

    node_positions = to_unstructured(data[pos_fields])
    node_types = data["node_type"]
    node_ids = data["point_id"]
    node_bcs = data["bc"].astype(str) if "bc" in data.dtype.names else None
    node_types_str = ", ".join([str(i) for i in np.unique(node_types).tolist()])
    if local_rank ==0:
        logger.info(f"Loaded {len(node_positions)} nodes ({dimensions}D) with unique node types: {node_types_str}")
    return node_positions, node_types, node_ids, node_bcs


def get_edge_features(pos, edge_index):
    diff = pos[edge_index[1]] - pos[edge_index[0]]  # Position difference
    dist = torch.norm(diff, dim=1, keepdim=True)  # Distance
    return torch.cat([diff, dist], dim=-1)  # Concatenate


def load_and_process_edges(edge_file: Path, node_ids: np.ndarray) -> np.ndarray:
    """
    Load edges and convert from node IDs to array indices.

    Args:
        edge_file: Path to edge file
        node_ids: Array of node IDs from load_node_data

    Returns:
        edges: (2, n_edges) array with bidirectional edges
    """
    # Load edges (assumes they contain point_ids, not indices)
    edges_raw = np.load(edge_file).T.astype(np.int64)  # Shape: (2, n_edges)

    # Create mapping from point_id to array index
    point_id_to_idx = {int(node_ids[i]): i for i in range(len(node_ids))}

    # Map edges from point_ids to indices
    edges = np.zeros_like(edges_raw)
    for i in range(edges_raw.shape[1]):
        edges[0, i] = point_id_to_idx[edges_raw[0, i]]
        edges[1, i] = point_id_to_idx[edges_raw[1, i]]

    # Make edges bidirectional by adding reverse edges
    # Remove duplicates
    edges_bidirectional = np.concatenate([edges, edges[[1, 0], :]], axis=1)
    edges_unique = np.unique(edges_bidirectional, axis=1)

    print(
        f"Loaded {edges_raw.shape[1]} edges -> {edges_unique.shape[1]} bidirectional unique edges"
    )

    # Validation
    assert edges_unique.min() >= 0, f"Negative edge indices: {edges_unique.min()}"
    assert edges_unique.max() < len(
        node_ids
    ), f"Edge index {edges_unique.max()} >= {len(node_ids)} nodes"

    return edges_unique

def create_bc_masks(node_bcs, fields):
    """Create boolean masks for each field's boundary conditions"""
    bc_masks = {}
    node_bcs = node_bcs.flatten()
    for field_idx, field_name in enumerate(fields):
        # Check if this field appears in the bc column
        mask = np.char.find(node_bcs, field_name) >= 0
        bc_masks[field_idx] = torch.from_numpy(mask)
    
    return bc_masks


def handle_periodic_boundaries(
    node_positions: np.ndarray,  # Shape: (n_nodes, 3)
    node_types: np.ndarray,  # Shape: (n_nodes, 1)
    edges: np.ndarray,  # Shape: (2, n_edges)
    tolerance: float = 1e-10,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Handle z-periodic boundary conditions by mapping max-z nodes to min-z nodes.

    Args:
        node_positions: Node coordinates (must be 3D)
        node_types: Node type array
        edges: Edge connectivity
        tolerance: Tolerance for position matching

    Returns:
        - node_positions_kept: Positions with max-z nodes removed
        - node_types_kept: Node types with max-z nodes removed
        - edges_remapped: Edges with max-z->min-z mapping applied
        - kept_node_indices: Indices of nodes kept from input
    """
    assert node_positions.shape[1] == 3, "Periodic BC only for 3D"

    min_z = node_positions[:, 2].min()
    max_z = node_positions[:, 2].max()

    is_min_z = np.isclose(node_positions[:, 2], min_z, atol=tolerance)
    is_max_z = np.isclose(node_positions[:, 2], max_z, atol=tolerance)

    min_z_indices = np.where(is_min_z)[0]
    max_z_indices = np.where(is_max_z)[0]

    # Map max-z nodes to corresponding min-z nodes by (x,y) position
    scale = 1.0 / tolerance
    min_z_pos_map = {}
    for idx in min_z_indices:
        key = tuple(np.round(node_positions[idx, :2] * scale).astype(int))
        min_z_pos_map[key] = idx

    max_to_min_mapping = {}
    for idx in max_z_indices:
        key = tuple(np.round(node_positions[idx, :2] * scale).astype(int))
        if key in min_z_pos_map:
            max_to_min_mapping[idx] = min_z_pos_map[key]

    # Remap edges: max-z nodes -> min-z nodes
    edges_remapped = edges.copy()
    for i in range(edges.shape[1]):
        if edges[0, i] in max_to_min_mapping:
            edges_remapped[0, i] = max_to_min_mapping[edges[0, i]]
        if edges[1, i] in max_to_min_mapping:
            edges_remapped[1, i] = max_to_min_mapping[edges[1, i]]

    # Remove edges between max-z nodes (they're redundant)
    keep_edge = np.ones(edges.shape[1], dtype=bool)
    for i in range(edges.shape[1]):
        if edges[0, i] in max_z_indices and edges[1, i] in max_z_indices:
            keep_edge[i] = False
    edges_remapped = edges_remapped[:, keep_edge]

    # Determine which nodes to keep (all except max-z)
    keep_node = np.ones(len(node_positions), dtype=bool)
    keep_node[max_z_indices] = False
    kept_node_indices = np.where(keep_node)[0]

    # Create old->new index mapping
    old_to_new = np.full(len(node_positions), -1, dtype=np.int32)
    old_to_new[kept_node_indices] = np.arange(len(kept_node_indices))

    # Renumber edges to new node indices
    edges_remapped = old_to_new[edges_remapped]

    # Filter node data
    node_positions_kept = node_positions[kept_node_indices]
    node_types_kept = node_types[kept_node_indices]

    print(f"Periodic BC: {len(node_positions)} -> {len(kept_node_indices)} nodes")
    print(f"             Mapped {len(max_to_min_mapping)} max-z to min-z nodes")

    # Validation
    assert (
        edges_remapped.min() >= 0 if edges_remapped.size > 0 else True
    ), f"Invalid edge mapping: negative indices"
    assert (
        edges_remapped.max() < len(kept_node_indices)
        if edges_remapped.size > 0
        else True
    ), f"Invalid edge mapping: index {edges_remapped.max()} >= {len(kept_node_indices)}"
    assert len(kept_node_indices) == len(node_positions) - len(
        max_z_indices
    ), f"Node count mismatch: {len(kept_node_indices)} != {len(node_positions)} - {len(max_z_indices)}"

    return (node_positions_kept, node_types_kept, edges_remapped, kept_node_indices)


def create_spatial_mask(points, bounds):
    """Create a boolean mask for points within spatial bounds."""
    dims = min(points.shape[1], 3)
    mask = np.all(
        [
            (points[:, i] >= bounds.get(k + "_min", -np.inf))
            & (points[:, i] <= bounds.get(k + "_max", np.inf))
            for i, k in enumerate(["x", "y", "z"][:dims])
        ],
        axis=0,
    )
    return mask


def apply_spatial_crop(node_positions, edges, bounds):
    """
    Apply spatial cropping to nodes and edges.

    Args:
        node_positions: (n_nodes, dims) array
        edges: (2, n_edges) array
        bounds: Dict with x_min, x_max, y_min, y_max, (z_min, z_max)

    Returns:
        - node_positions_cropped: Cropped positions
        - edges_cropped: Edges with updated indices
        - kept_node_indices: Indices of kept nodes from input array
    """
    # Use the shared helper function
    in_bounds = create_spatial_mask(node_positions, bounds)
    kept_node_indices = np.where(in_bounds)[0]

    # Create mapping from old indices to new indices
    old_to_new = np.full(len(node_positions), -1, dtype=np.int32)
    old_to_new[kept_node_indices] = np.arange(len(kept_node_indices))

    # Filter edges - keep only if both nodes are kept
    edge_valid = (old_to_new[edges[0]] >= 0) & (old_to_new[edges[1]] >= 0)

    # Remap edge indices to new numbering
    edges_cropped = np.stack(
        [old_to_new[edges[0, edge_valid]], old_to_new[edges[1, edge_valid]]]
    )

    # Crop node data
    node_positions_cropped = node_positions[kept_node_indices]

    print(f"Spatial crop: {len(node_positions)} -> {len(kept_node_indices)} nodes")
    print(f"              {edges.shape[1]} -> {edges_cropped.shape[1]} edges")

    return node_positions_cropped, edges_cropped, kept_node_indices


def create_one_hot_encoding(node_type):
    unique_types = np.unique(node_type)
    n_node_types = unique_types.shape[0]
    val_to_index = {v: i for i, v in enumerate(unique_types)}
    indices = np.array([val_to_index[v] for v in node_type.squeeze()])
    tnt = torch.from_numpy(indices)
    return torch.nn.functional.one_hot(tnt, n_node_types), val_to_index


def create_update_mask(
    node_types: np.ndarray, update_node_tags: List[int]
) -> torch.Tensor:
    mask = torch.zeros(len(node_types), dtype=torch.bool)

    for tag in update_node_tags:
        mask |= torch.from_numpy(node_types.squeeze()) == tag

    return mask
