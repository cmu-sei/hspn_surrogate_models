import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import ConcatDataset, Dataset

from hspn.dogn.constants import NODE_TYPE_NORMAL, NODE_TYPE_OUTFLOW


class MeshDataset(Dataset):
    """PyTorch dataset for CFD mesh data."""

    def __init__(
        self,
        mode: str,
        root: str,
        subdirs: Optional[List[str]] = None,
        fields: Optional[List[str]] = None,
        noise: float = 0.0,
        tmin: int = None,
        tmax: int = None,
        xmin: Optional[float] = None,
        xmax: Optional[float] = None,
        ymin: Optional[float] = None,
        ymax: Optional[float] = None,
        push_forward_count: int = 0,
        bundle: int = 1,
        bundle_style: str = "non-overlapping",
        **kwargs,  # Ignore extra params
    ):
        if tmin is None or tmax is None:
            raise ValueError("tmin and tmax must be specified")

        self.mode = mode
        self.root = root
        self.subdirs = subdirs or [""]
        self.fields = fields or ["density"]
        self.noise = noise
        self.tmin = tmin
        self.tmax = tmax
        self.bounds = [xmin, xmax, ymin, ymax]
        self.push_forward_count = push_forward_count
        self.bundle = bundle
        self.bundle_style = bundle_style
        self._load_all()

    def _load_all(self):
        """Load everything."""
        # Load graph
        pos_data = np.load(f"{self.root}/nodes.npy")
        x, y = pos_data["X"], pos_data["Y"]
        node_types = pos_data["node_type"]
        edges = np.load(f"{self.root}/edges_cell.npy").T

        # Crop if needed
        if any(b is not None for b in self.bounds):
            mask = np.ones(len(x), dtype=bool)
            if self.bounds[0] is not None:
                mask &= x >= self.bounds[0]
            if self.bounds[1] is not None:
                mask &= x <= self.bounds[1]
            if self.bounds[2] is not None:
                mask &= y >= self.bounds[2]
            if self.bounds[3] is not None:
                mask &= y <= self.bounds[3]

            node_map = np.where(mask)[0]
            o2n = np.full(len(x), -1, dtype=int)
            o2n[node_map] = np.arange(len(node_map))

            edge_mask = (o2n[edges[0]] >= 0) & (o2n[edges[1]] >= 0)
            edges = np.stack([o2n[edges[0, edge_mask]], o2n[edges[1, edge_mask]]])

            x, y = x[node_map], y[node_map]
            node_types = node_types[node_map]
        else:
            node_map = np.arange(len(x))

        # Convert graph to tensors
        self.pos = torch.stack([torch.from_numpy(x), torch.from_numpy(y)], dim=1).float()
        self.edges = torch.from_numpy(edges)
        self.node_types = torch.from_numpy(node_types).long()

        # One-hot and mask
        unique = np.unique(node_types)
        self.n_node_types = len(unique)
        idx_map = {t: i for i, t in enumerate(unique)}
        indices = torch.tensor([idx_map[t] for t in node_types])
        self.one_hot_node_type = torch.nn.functional.one_hot(indices, self.n_node_types)
        self.mask = (self.node_types == NODE_TYPE_NORMAL) | (self.node_types == NODE_TYPE_OUTFLOW)

        # Edge features
        pos_diff = self.pos[self.edges[1]] - self.pos[self.edges[0]]
        self.edge_features = torch.cat([pos_diff, pos_diff.norm(dim=1, keepdim=True)], dim=-1)

        # Load field data
        all_data = []
        for subdir in self.subdirs:
            # Params
            params_path = Path(f"{self.root}/{subdir}{'/' if subdir else ''}params.json")
            params = (
                torch.tensor(list(json.load(open(params_path)).values())) if params_path.exists() else torch.tensor([])
            )

            # Fields
            data_list = []
            for t in range(self.tmin, self.tmax + 1):
                d = np.load(f"{self.root}/{subdir}{'/' if subdir else ''}flow_{t:05d}.npy", allow_pickle=True)
                data_list.append(d if d.shape else d.item())

            fields = torch.stack(
                [torch.from_numpy(np.vstack([d[f] for d in data_list]).T).T for f in self.fields], dim=-1
            ).float()

            # Crop fields
            fields = fields[:, node_map, :]

            # Bundle
            if self.bundle_style == "overlapping":
                # Colleague's style: sliding window with stride 1
                bundled = [
                    torch.stack([fields[i + j] for j in range(self.bundle)], dim=-1)
                    for i in range(len(fields) - self.bundle + 1)
                ]

            else:
                # Your style: non-overlapping windows
                limit = len(fields) - len(fields) % self.bundle
                bundled = [
                    torch.stack([fields[i + j] for j in range(self.bundle)], dim=-1)
                    for i in range(0, limit, self.bundle)
                ]

            # Create samples
            n_samples = len(bundled) - 1 - self.push_forward_count
            for i in range(n_samples):
                if self.bundle_style == "overlapping":
                    time_idx = torch.arange(i, i + self.bundle)
                else:
                    time_idx = torch.arange(i * self.bundle, (i + 1) * self.bundle)

                # Add global time offset
                time_idx = time_idx + self.tmin
                sample = {
                    "x": bundled[i],
                    "y": bundled[i + 1],
                    "y_pf": bundled[i + 1 + self.push_forward_count] if self.push_forward_count > 0 else None,
                    "params": params,
                    "time_idx": time_idx,
                }
                all_data.append(sample)

        pos_bytes = self.pos.cpu().numpy().tobytes()
        onehot_bytes = self.one_hot_node_type.cpu().numpy().tobytes()
        mask_bytes = self.mask.cpu().numpy().tobytes()
        node_types_bytes = self.node_types.cpu().numpy().tobytes()
        sort_idx = torch.argsort(self.edges[0] * 10000 + self.edges[1])
        edges_sorted = self.edges[:, sort_idx]
        features_sorted = self.edge_features[sort_idx]

        print(f"Code B - Pos hash: {hashlib.md5(pos_bytes).hexdigest()}")
        print(f"Code B - One-hot hash: {hashlib.md5(onehot_bytes).hexdigest()}")
        print(f"Code B - Mask hash: {hashlib.md5(mask_bytes).hexdigest()}")
        print(f"Code B - Node types hash: {hashlib.md5(node_types_bytes).hexdigest()}")
        print(f"Code B - Sorted edges hash: {hashlib.md5(edges_sorted.cpu().numpy().tobytes()).hexdigest()}")
        print(f"Code B - Sorted features hash: {hashlib.md5(features_sorted.cpu().numpy().tobytes()).hexdigest()}")
        print(f"Code B - NODE_TYPE_NORMAL={NODE_TYPE_NORMAL}, NODE_TYPE_OUTFLOW={NODE_TYPE_OUTFLOW}")

        # Stack all samples
        self._stack_data(all_data)

    def _stack_data(self, all_data):
        """Stack all data"""
        # Collect for statistics
        all_fields = []
        for d in all_data:
            all_fields.extend([d["x"], d["y"]])
            if d["y_pf"] is not None:
                all_fields.append(d["y_pf"])

        stacked = torch.stack(all_fields)
        self.mean = stacked.mean(dim=(0, 1, 3))  # Average over samples, nodes, bundle
        self.std = stacked.std(dim=(0, 1, 3))

        self.x = torch.stack([d["x"] for d in all_data])
        self.single_step_y = torch.stack([d["y"] for d in all_data])

        if all_data[0]["y_pf"] is not None:
            self.push_forward_y = torch.stack([d["y_pf"] for d in all_data])
        else:
            self.push_forward_y = None

        # Stack params
        self.params = torch.stack([d["params"] if d["params"].numel() > 0 else torch.zeros(2) for d in all_data])

        self.time_indices = torch.stack([d["time_idx"] for d in all_data])

        # Add these prints:
        print(f"Code B - Mean: {self.mean.cpu().numpy()}")
        print(f"Code B - Std: {self.std.cpu().numpy()}")
        print(f"Code B - Var: {self.std**2}")
        print(f"Code B - Shape: mean {self.mean.shape}, std {self.std.shape}")

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        # Normalize on-the-fly using current statistics
        norm = lambda x: (x - self.mean.view(1, -1, 1)) / self.std.view(1, -1, 1)

        # Normalize first
        x_normalized = norm(self.x[idx])
        single_step_y_normalized = norm(self.single_step_y[idx])
        push_forward_y_normalized = norm(self.push_forward_y[idx]) if self.push_forward_y is not None else None

        # Then add noise to normalized data (only for training)
        if self.mode == "train" and self.noise > 0:
            x_normalized = x_normalized + self.noise * torch.randn_like(x_normalized)

        return {
            "field_data": x_normalized,
            "single_step_y": norm(self.single_step_y[idx]),
            "params": self.params[idx],
            "push_forward_y": norm(self.push_forward_y[idx]) if self.push_forward_y is not None else None,
            "edge_features": self.edge_features,
            "edge_index": self.edges,
            "one_hot_node_type": self.one_hot_node_type,
            "time_indices": self.time_indices[idx],
            "mask": self.mask,
            "bundle_size": self.bundle,
            "push_forward_count": self.push_forward_count,
        }

    def reset_statistics(self, mean, var):
        """Reset statistics and handle various input types."""
        # Convert to tensors if needed
        if isinstance(mean, np.ndarray):
            mean = torch.from_numpy(mean).float()
        elif not isinstance(mean, torch.Tensor):
            mean = torch.tensor(mean).float()

        if isinstance(var, np.ndarray):
            var = torch.from_numpy(var).float()
        elif not isinstance(var, torch.Tensor):
            var = torch.tensor(var).float()

        # Handle scalar values - broadcast to number of fields
        if mean.numel() == 1:
            mean = mean.repeat(len(self.fields))
        if var.numel() == 1:
            var = var.repeat(len(self.fields))

        # Ensure correct shape
        if mean.shape != (len(self.fields),):
            raise ValueError(f"Mean shape {mean.shape} doesn't match number of fields {len(self.fields)}")
        if var.shape != (len(self.fields),):
            raise ValueError(f"Variance shape {var.shape} doesn't match number of fields {len(self.fields)}")

        # Update statistics
        self.mean = mean
        self.std = torch.sqrt(var)


class ConcatDatasetWithStats(ConcatDataset):
    def __init__(self, datasets):
        super().__init__(datasets)
        # Pull statistics from first dataset (assuming all have same stats)
        if datasets:
            self.mean = datasets[0].mean
            self.std = datasets[0].std
            self.fields = datasets[0].fields
            self.bundle = datasets[0].bundle
            # Any other attributes you need


def get_dataset(
    mode: str, config: Dict[str, Any], train_stats: Optional[Dict[str, torch.Tensor]] = None
) -> MeshDataset:
    """Load dataset with new config structure."""
    # Check if old structure (has 'dataset' key at root)
    if "dataset" in config and "train_datasets" not in config:
        # Old format - convert to new format
        old_dataset = config["dataset"]

        # Create new structure from old
        new_config = {
            "train_datasets": [{"root": old_dataset["root"], "field_dir": old_dataset.get("train", [])}],
            "valid_datasets": [{"root": old_dataset["root"], "field_dir": old_dataset.get("test", [])}],
            "fields": old_dataset.get("fields", ["density"]),
            "time_minimum": old_dataset.get("tmin", 0),
            "time_maximum": old_dataset.get("tmax", 549),
            "bundle": old_dataset.get("bundle", 1),
            "push_forward": old_dataset.get("pushforward", 0),
            "noise": old_dataset.get("noise", 0.0),
            "bounds": {
                "x_min": old_dataset.get("xmin", -999),
                "x_max": old_dataset.get("xmax", 999),
                "y_min": old_dataset.get("ymin", -999),
                "y_max": old_dataset.get("ymax", 999),
            },
        }
        config = new_config

    # Get the appropriate dataset list
    datasets_key = "train_datasets" if mode == "train" else "valid_datasets"
    dataset_list = config.get(datasets_key, [])

    if not dataset_list:
        raise ValueError(f"No {datasets_key} found in config")

    # Create all datasets
    datasets = []
    for dataset_info in dataset_list:
        dataset_config = {
            "mode": mode,
            "root": dataset_info["root"],
            "subdirs": dataset_info.get("field_dir", []),
            "fields": config.get("fields"),
            "tmin": config.get("time_minimum", 0),
            "tmax": config.get("time_maximum", -1),
            "noise": config.get("noise", 0.0),
            "bundle": config.get("bundle", 1),
            "bundle_style": config.get("bundle_style", "non_overlapping"),
            "push_forward_count": config.get("push_forward", 0),
            "dimensions": config.get("dimensions", 2),
            "n_params": config.get("n_params", 0),
            "max_samples_per_field_dir": config.get("max_samples_per_field_dir", -1),
            "update_node_tags": config.get("update_node_tags", [0]),
            "xmin": config.get("bounds", {}).get("x_min", None),
            "xmax": config.get("bounds", {}).get("x_max", None),
            "ymin": config.get("bounds", {}).get("y_min", None),
            "ymax": config.get("bounds", {}).get("y_max", None),
        }

        dataset = MeshDataset(**dataset_config)

        if train_stats:
            dataset.reset_statistics(train_stats["mean"], train_stats["var"])

        datasets.append(dataset)

    # Return single dataset or concatenated datasets
    if len(datasets) == 1:
        return datasets[0]
    else:
        return ConcatDatasetWithStats(datasets)
