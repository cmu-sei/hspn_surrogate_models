import json
from pathlib import Path
from typing import Dict, List
import hashlib
import numpy as np
import torch

from torch.utils.data import ConcatDataset, Dataset
from numpy.lib.recfunctions import \
    structured_to_unstructured as to_unstructured
from torch.utils.data.distributed import DistributedSampler

from hspn.dogn.graph_utils import (apply_spatial_crop, create_one_hot_encoding,
                                    create_update_mask, create_bc_masks, get_edge_features,
                                    handle_periodic_boundaries, load_and_process_edges,
                                    load_node_data)


def load_trajectory_data(
    trajectory_path: Path,
    config: dict,
    node_mapping: np.ndarray,
    edges: np.ndarray = None,
) -> dict:
    """Load field data for a single trajectory, handling moving nodes if specified."""

    # Load parameters
    with open(trajectory_path / "params.json", "r") as f:
        params_dict = json.load(f)
    param_keys = config.get("param_keys", [])
    if param_keys:
        missing = [k for k in param_keys if k not in params_dict]
        if missing:
            raise ValueError(
                f"{trajectory_path}/params.json missing keys: {missing}. "
                f"Available keys: {list(params_dict.keys())}"
            )
        extra = [k for k in params_dict if k not in param_keys]
        if extra:
            logger.warning(f"{trajectory_path}/params.json has extra keys not in param_keys: {extra}")
        params = torch.tensor([params_dict[k] for k in param_keys], dtype=torch.float32)
    else:
        params = torch.tensor(list(params_dict.values()), dtype=torch.float32)
    print(f"{trajectory_path.name}: params={params.tolist()}")
    # Determine fields to load
    fields = config["fields"].copy()
    node_pos_fields = config.get("node_positions", [])
    if node_pos_fields:
        fields.extend(node_pos_fields)

    # Load time series
    if config["max_samples_per"]<=0:
        config["max_samples_per"] = 1e12
        
    files = sorted(trajectory_path.glob("f*_*.np*"))  # Gets both .npy and .npz
    start = config["time_minimum"]
    end = (
        min(min(config["time_maximum"]+1, config["max_samples_per"]+start), len(files))
        if config["time_maximum"] > 0
        else len(files)
    )

    field_data = []
    edge_features_update = []

    for file in files[start:end]:
        # Load and extract fields
        data = np.load(file)
        field_array = data["raw"][fields] if file.suffix == ".npz" else data[fields]
        field_tensor = torch.from_numpy(to_unstructured(field_array).copy()).float()[
            node_mapping
        ]

        # Handle moving nodes
        if node_pos_fields:
            n_pos = len(node_pos_fields)
            positions = field_tensor[:, -n_pos:]
            field_tensor = field_tensor[:, :-n_pos]
            if edges is not None:
                edge_features_update.append(
                    get_edge_features(positions, torch.from_numpy(edges))
                )

        field_data.append(field_tensor)
    return {
        "field_data": field_data,
        "params": params,
        "edge_features_update": edge_features_update if node_pos_fields else None,
    }


def apply_temporal_bundling(
    trajectory_data: dict, bundle_size: int, overlap_bundles: bool, push_forward: int
):
    """Bundle time series data into training samples."""
    field_data = trajectory_data["field_data"]
    edge_features = trajectory_data.get("edge_features_update")

    def bundle(data_list):
        window_size = bundle_size * (2 + push_forward)
        if overlap_bundles:
            return [
                torch.stack(data_list[i : i + window_size], dim=-1)
                for i in range(len(data_list) - window_size + 1)
            ]
        else:
            return [
                torch.stack(data_list[i : i + window_size], dim=-1)
                for i in range(
                    0, len(data_list) - window_size + 1, bundle_size  # Changed: step by bundle_size, check window_size
                )
            ]

    bundled = bundle(field_data)
    edge_bundled = bundle(edge_features) if edge_features else None

    # Changed: extract the three components from each bundled window
    field_data_out = [b[..., :bundle_size] for b in bundled]
    single_step_y = [b[..., bundle_size:2*bundle_size] for b in bundled]
    push_forward_y = (
        [b[..., bundle_size*(1+push_forward):bundle_size*(2+push_forward)] for b in bundled]
        if push_forward > 0
        else None
    )

    return {
        "field_data": field_data_out,
        "single_step_y": single_step_y,
        "push_forward_y": push_forward_y,
        "params": trajectory_data["params"],
        "edge_features_update": [e[..., :bundle_size] for e in edge_bundled] if edge_bundled else None,
    }


def compute_statistics(
    trajectory_data: List[dict], norm_by_traj=False, L2_var_fields=False
) -> dict:
    """
    Compute statistics for normalization using only input data.
    """

    if norm_by_traj:
        raise NotImplementedError("IDK if this is necessary")
    if L2_var_fields:
        raise NotImplementedError(
            "In case you want to take the L2 norm of vector fields. not implemented"
        )

    all_x = []
    for traj in trajectory_data:
        all_x.extend(traj["field_data"])

    stacked = torch.stack(
        all_x
    )  # Now shape: (n_samples, n_nodes, n_fields, bundle_size)

    print(f"Mean: { stacked.mean(dim=(0, 1, 3)).cpu().numpy()}")
    print(f"Var: {stacked.var(dim=(0, 1, 3)).cpu().numpy()}")
    print(f"Shape: mean {stacked.mean(dim=(0, 1, 3)).shape}, var {stacked.var(dim=(0, 1, 3)).shape}")

    return {
        "mean": stacked.mean(dim=(0, 1, 3)),  # (n_fields,)
        "var": stacked.var(dim=(0, 1, 3)),  # (n_fields,)
        "min": stacked.min(dim=0)[0].min(dim=0)[0].min(dim=1)[0],  # (n_fields,)
        "max": stacked.max(dim=0)[0].max(dim=0)[0].max(dim=1)[0],  # (n_fields,)
        "count": stacked.shape[0] * stacked.shape[1],
    }


def merge_statistics(stats_list):
    """Merge statistics from multiple datasets"""
    if not stats_list:
        return {"mean": 0, "var": 0, "min": 0, "max": 0, "count": 0}

    r_stats = stats_list[0].copy()

    for stats in stats_list[1:]:
        N0, N1 = r_stats["count"], stats["count"]
        if N0 == 0:
            r_stats = stats.copy()
        else:
            N_total = N0 + N1
            delta = stats["mean"] - r_stats["mean"]
            r_stats["count"] = N_total
            r_stats["mean"] = r_stats["mean"] + delta * (N1 / N_total)
            r_stats["var"] = (
                N0 * r_stats["var"] + N1 * stats["var"] + (N0 * N1) * delta**2 / N_total
            ) / N_total
            r_stats["min"] = torch.minimum(r_stats["min"], stats["min"])
            r_stats["max"] = torch.maximum(r_stats["max"], stats["max"])

    return r_stats


def _load_single_dataset(dataset_cfg, common_cfg, rank=0):
    """
    Load and process a single CFD mesh dataset entry.

    This is the internal workhorse. External code should call get_dataset() instead.

    Pipeline:
    1. Load graph structure (nodes, edges)
    2. Handle periodic boundaries (if applicable)
    3. Apply spatial cropping
    4. Load field data for the selected nodes
    5. Apply temporal processing (bundling)
    6. Compute statistics
    7. Return processed dataset
    """

    # ----- STEP 1: SETUP AND LOAD GRAPH STRUCTURE
    config = {**common_cfg, **dataset_cfg}
    root = Path(config.get("root", "/"))

    # Load nodes and edges
    node_positions, node_types, node_ids, node_bcs = load_node_data(
        root / config.get("node_file"), config.get("dimensions")
    )
    edges = load_and_process_edges(root / config.get("edge_file"), node_ids)

    # ----- STEP 2: HANDLE PERIODIC BOUNDARIES (if applicable)
    if config.get("z_repeats", False) and config.get("dimensions") == 3:
        node_positions, node_types, edges, periodic_node_mapping = (
            handle_periodic_boundaries(node_positions, node_types, edges)
        )
    else:
        periodic_node_mapping = np.arange(len(node_positions))

    # ----- STEP 3: SPATIAL CROPPING
    bounds = config.get("bounds", {})
    node_positions, edges, crop_node_mapping = apply_spatial_crop(
        node_positions, edges, bounds
    )
    node_types = node_types[crop_node_mapping]

    #          Compose mappings: original -> periodic -> cropped
    final_node_mapping = periodic_node_mapping[crop_node_mapping]

    # ----- STEP 4: COMPUTE GRAPH EDGES
    pos_tensor = torch.from_numpy(node_positions).float()
    edges_tensor = torch.from_numpy(edges).long()
    edge_features = get_edge_features(pos_tensor, edges_tensor)

    # Create one-hot encoding for node types
    one_hot_node_type, val_to_index = create_one_hot_encoding(node_types)

    # Create update mask based on node types
    update_mask = create_update_mask(node_types, config.get("update_node_tags", []))
    
    #Create dirichlet mask dictionary
    dirichlet_masks = None
    if config.get("dirichlet_bc") and node_bcs is not None:
        dirichlet_masks = create_bc_masks(node_bcs, config.get('fields',[])) 

    # ----- STEP 5: LOAD AND PROCESS FIELD DATA
    field_dirs = config.get("field_dir", [])
    all_trajectory_data = []

    for trajectory_dir in field_dirs:
        trajectory_data = load_trajectory_data(
            root / trajectory_dir, config, final_node_mapping
        )
        all_trajectory_data.append(trajectory_data)
        
    # ----- STEP 6: TEMPORAL PROCESSING
    bundled_data = []
    for trajectory in all_trajectory_data:
        bundled = apply_temporal_bundling(
            trajectory,
            config.get("bundle", 1),
            config.get("overlap_bundles", False),
            config.get("push_forward", 0),
        )
        bundled_data.append(bundled)
        
    # ---- STEP 7: COMPUTE STATISTICS
    nf = len(config.get('fields',[]))
    statistics =  {"mean": torch.zeros(nf), "var": torch.ones(nf), "min": torch.zeros(nf),  "max": torch.zeros(nf), "count": 0}
    
    if config.get("norm"):
        statistics = compute_statistics(
            bundled_data,
            config.get("norm_by_traj", False),
            config.get("L2_var_fields", False),
        )
    

    # Hash node_types
    pos_bytes = pos_tensor.cpu().numpy().tobytes()
    onehot_bytes = one_hot_node_type.cpu().numpy().tobytes()
    mask_bytes = update_mask.cpu().numpy().tobytes()
    node_types_bytes = node_types.tobytes()  # numpy array
    sort_idx = torch.argsort(edges_tensor[0] * 10000 + edges_tensor[1])
    edges_sorted = edges_tensor[:, sort_idx]
    features_sorted = edge_features[sort_idx]

    print(f"Debug - Pos hash: {hashlib.md5(pos_bytes).hexdigest()}") #use hashes to compare to other code
    print(f"Debug - One-hot hash: {hashlib.md5(onehot_bytes).hexdigest()}")
    print(f"Debug - Mask hash: {hashlib.md5(mask_bytes).hexdigest()}")
    print(f"Debug - Node types hash: {hashlib.md5(node_types_bytes).hexdigest()}")
    print(f"Debug - Sorted edges hash: {hashlib.md5(edges_sorted.cpu().numpy().tobytes()).hexdigest()}")
    print(f"Debug - Sorted features hash: {hashlib.md5(features_sorted.cpu().numpy().tobytes()).hexdigest()}")
    
    return {
        "root": root,
        "data": bundled_data,  # List of dicts with x, single_step_y, push_forward_y, etc.
        "statistics": statistics,
        "edge_index": edges_tensor.long(),  # tensor (long)
        "edge_features": edge_features,  # tensor
        "one_hot_node_type": one_hot_node_type,  # tensor
        "pos": torch.from_numpy(node_positions).float(),  # tensor
        "mask": update_mask,  # tensor
        "dirichlet_masks": dirichlet_masks,
        "push_forward_count":config.get("push_forward",0)
    }

def get_dataset(mode, config, train_stats=None):
    """
    Load dataset from Hydra config. 
    Args:
        mode: "train" or "valid"
        config: The resolved dataset config section (dict or DictConfig).
                Expected to have train_datasets/valid_datasets entries.
        train_stats: If provided, override computed statistics (for val set).
                     Dict with "mean" and "var" as tensors.

    Returns:
        MeshDataset or ConcatDatasetWithStats, with .mean, .std, .fields,
        .bundle, .n_node_types attributes that train_gnn.py reads.
    """
    # Handle DictConfig from Hydra
    try:
        from omegaconf import DictConfig, OmegaConf
        if isinstance(config, DictConfig):
            config = OmegaConf.to_container(config, resolve=True)
    except ImportError:
        pass

    datasets_key = "train_datasets" if mode == "train" else "valid_datasets"
    dataset_entries = config.get(datasets_key, [])

    if not dataset_entries:
        raise ValueError(f"No {datasets_key} found in config")

    # Build the "common config" that _load_single_dataset expects.
    common_cfg = {
        "fields": config.get("fields", []),
        "time_minimum": config.get("time_minimum", 0),
        "time_maximum": config.get("time_maximum", -1),
        "bundle": config.get("bundle", 1),
        "overlap_bundles": config.get("bundle_style", "non_overlapping") == "overlapping",
        "push_forward": config.get("push_forward", 0),
        "dimensions": config.get("dimensions", 2),
        "param_keys": config.get("param_keys", []),
        "max_samples_per": config.get("max_samples_per_field_dir", -1),
        "update_node_tags": config.get("update_node_tags", [0]),
        "norm": True,
        "bounds": config.get("bounds", {}),
        "dirichlet_bc": config.get("dirichlet_bc", False),
        "node_positions": config.get("node_positions", []),
        "z_repeats": config.get("z_repeats", False),
    }

    noise_level = config.get("noise", 0.0) if mode == "train" else 0.0

    
    # Load each dataset entry through the pipeline
    all_results = []
    for entry in dataset_entries:
        dataset_cfg = {
            "root": entry.get("root", config.get("root", "")),
            "node_file": entry.get("node_file", config.get("node_file", "nodes.npy")),
            "edge_file": entry.get("edge_file", config.get("edge_file", "edges_cell.npy")),
            "field_dir": entry.get("field_dir", []),
        }
        result = _load_single_dataset(dataset_cfg, common_cfg)
        all_results.append(result)

    # Compute or apply statistics
    if train_stats is not None:
        statistics = train_stats
    else:
        stats_list = [r["statistics"] for r in all_results]
        statistics = merge_statistics(stats_list) if len(stats_list) > 1 else stats_list[0]

    # Build MeshDataset objects
    fields_list = config.get("fields", [])
    bundle_size = config.get("bundle", 1)

    datasets = []
    for result in all_results:
        ds = MeshDataset(
            data=result["data"],
            statistics=statistics,
            push_forward_count=result["push_forward_count"],
            edge_index=result["edge_index"],
            edge_features=result["edge_features"],
            one_hot_node_type=result["one_hot_node_type"],
            pos=result["pos"],
            mask=result["mask"],
            dirichlet_masks=result.get("dirichlet_masks"),
            noise=noise_level,
            mode=mode,
            fields=fields_list,
            bundle=bundle_size,
        )
        datasets.append(ds)

    if len(datasets) == 1:
        return datasets[0]
    else:
        return ConcatDatasetWithStats(datasets)
class MeshDataset(Dataset):
    def __init__(
        self,
        data,
        statistics,
        push_forward_count, 
        edge_index,
        edge_features,
        one_hot_node_type,
        pos,
        mask,
        dirichlet_masks=None, 
        device="cpu",
        pin_memory=True,
        noise=0.0,
        mode="train",
        fields=None,
        bundle=1,
    ):
        
        self.mean = statistics["mean"].float()
        self.std = torch.sqrt(statistics["var"]).float()
        self.noise = noise
        self.mode = mode
        self.n_node_types = one_hot_node_type.shape[1]
        self.fields = fields or []
        self.bundle = bundle

        # Store static graph structure
        self.edge_index = edge_index
        self.edge_features = edge_features
        self.one_hot_node_type = one_hot_node_type
        self.pos = pos
        self.mask = mask
        self.dirichlet_masks = dirichlet_masks
        self.push_forward_count = push_forward_count

        # print("Stacking all data...")

        # Stack all data at once
        self.field_data = torch.stack([i for a in data for i in a["field_data"]])
        self.single_step_y = torch.stack([i for a in data for i in a["single_step_y"]])
        self.n_fields = data[0]["field_data"][0].size(1)
        self.n_graphs = self.field_data.size(0)


        # Handle params
        params_list = []
        for sample in data:
            n_bundles = len(sample["field_data"])
            params_list.extend([sample["params"]] * n_bundles)
        self.params = torch.stack(params_list)

        # Optional data
        self.push_forward_y = None
        self.edge_update = None
        if data[0]["push_forward_y"] is not None:
            self.push_forward_y = torch.stack(
                [i for a in data for i in a["push_forward_y"]]
            )

        if (
             "edge_features_update" in data[0]
            and data[0]["edge_features_update"] is not None
        ):
            self.edge_update = torch.stack(
                [i for a in data for i in a["edge_features_update"]]
            )

        # Pin memory for faster GPU transfer
        if pin_memory and device != "cpu":
            self.field_data = self.field_data.pin_memory()
            self.single_step_y = self.single_step_y.pin_memory()
            self.params = self.params.pin_memory()
            if self.push_forward_y is not None:
                self.push_forward_y = self.push_forward_y.pin_memory()
            if self.edge_update is not None:
                self.edge_update = self.edge_update.pin_memory()

    def __len__(self):
        return len(self.field_data)

    def get_pos(self):
        return self.pos
    def reset_statistics(self, mean, var):
        """Reset statistics and handle various input types."""
        if isinstance(mean, np.ndarray):
            mean = torch.from_numpy(mean).float()
        elif not isinstance(mean, torch.Tensor):
            mean = torch.tensor(mean).float()
        
        if isinstance(var, np.ndarray):
            var = torch.from_numpy(var).float()
        elif not isinstance(var, torch.Tensor):
            var = torch.tensor(var).float()
        
        if mean.numel() == 1:
            mean = mean.repeat(self.n_fields)
        if var.numel() == 1:
            var = var.repeat(self.n_fields)
        
        if mean.shape != (self.n_fields,):
            raise ValueError(f"Mean shape {mean.shape} doesn't match number of fields {self.n_fields}")
        if var.shape != (self.n_fields,):
            raise ValueError(f"Variance shape {var.shape} doesn't match number of fields {self.n_fields}")
        
        self.mean = mean
        self.std = torch.sqrt(var)
    
    def __getitem__(self, idx):
        norm = lambda x: (x - self.mean.view(1, self.n_fields, 1)) / self.std.view(1, self.n_fields, 1)

        field_data = norm(self.field_data[idx])
        if self.mode == "train" and self.noise > 0:
            field_data = field_data + self.noise * torch.randn_like(field_data)

        pf_y = (
            norm(self.push_forward_y[idx])
            if self.push_forward_y is not None
            else torch.tensor([])
        )

        result = {
            "field_data": field_data,
            "single_step_y": norm(self.single_step_y[idx]),
            "params": self.params[idx],
            "push_forward_y": pf_y,
            "edge_features": self.edge_features,
            "edge_index": self.edge_index,
            "one_hot_node_type": self.one_hot_node_type,
            "mask": self.mask,
            "push_forward_count": self.push_forward_count,
            "dirichlet_masks": self.dirichlet_masks,
            "pos": self.pos,
        }

        # Per-sample data for moving-node problems
        if self.edge_update is not None:
            result["edge_update"] = self.edge_update[idx]

        return result

class ConcatDatasetWithStats(ConcatDataset):
    def __init__(self, datasets):
        super().__init__(datasets)
        if datasets:
            first = datasets[0]
            self.mean = first.mean
            self.std = first.std
            self.fields = first.fields
            self.bundle = first.bundle
            self.n_fields = first.n_fields
            self.n_node_types = first.n_node_types
            self.n_graphs = sum(ds.n_graphs for ds in datasets)
    def reset_statistics(self, mean, var):
        for ds in self.datasets:
            ds.reset_statistics(mean, var)


