import os
from pathlib import Path
import logging

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Sampler

logger = logging.getLogger(__name__)


class GraphBatchSampler(Sampler):
    def __init__(self, dataset, batch_size, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle

        # Group indices by (push_forward, num_nodes, num_edges)
        self.groups = {}
        for idx in range(len(dataset)):
            item = dataset[idx]
            # Create key from properties that must match
            key = (
                item["push_forward_count"],
                item["field_data"].shape[0],  # num_nodes
                item["edge_index"].shape[1],  # num_edges
            )
            if key not in self.groups:
                self.groups[key] = []
            self.groups[key].append(idx)

        # Log group information
        print(f"Created {len(self.groups)} groups:")
        for key, indices in self.groups.items():
            pf, n_nodes, n_edges = key
            print(f"  PF={pf}, Nodes={n_nodes}, Edges={n_edges}: {len(indices)} samples")

    def __iter__(self):
        # Process each group
        for key, indices in self.groups.items():
            if self.shuffle:
                indices = [indices[i] for i in torch.randperm(len(indices)).tolist()]

            # Yield complete batches from this group
            for i in range(0, len(indices), self.batch_size):
                batch = indices[i : i + self.batch_size]
                if batch:  # Don't yield empty batches
                    yield batch

    def __len__(self):
        total_batches = 0
        for indices in self.groups.values():
            total_batches += (len(indices) + self.batch_size - 1) // self.batch_size
        return total_batches


class GradientMonitor:
    def __init__(self, model):
        self.model = model
        self.gradient_stats = {}
        self.handles = []
        self._register_hooks()

    def _register_hooks(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                handle = param.register_hook(self._make_hook(name))
                self.handles.append(handle)

    def _make_hook(self, name):
        def hook(grad):
            self.gradient_stats[name] = {
                "max": grad.abs().max().item(),
                "mean": grad.abs().mean().item(),
                "variance": grad.var().item(),
                "norm": grad.norm().item(),
            }
            return grad

        return hook

    def get_stats(self):
        return self.gradient_stats

    def get_top_gradients(self, n=10, metric="max"):
        """Get the n parameters with largest gradients

        Args:
            n: number of top parameters to return
            metric: which metric to sort by ('max', 'mean', 'variance', 'norm')
        """
        sorted_params = sorted(self.gradient_stats.items(), key=lambda x: x[1][metric], reverse=True)
        return sorted_params[:n]

    def print_top_gradients(self, n=10, metric="max"):
        """Print the top n parameters by gradient magnitude"""
        top_grads = self.get_top_gradients(n, metric)
        print(f"\nTop {n} parameters by gradient {metric}:")
        print("-" * 60)
        for name, stats in top_grads:
            print(f"{name:40s} | max: {stats['max']:8.4f} | mean: {stats['mean']:8.4f}")

    def check_for_explosion(self, threshold=100.0):
        """Check if any gradients exceed threshold"""
        exploded = []
        for name, stats in self.gradient_stats.items():
            if stats["max"] > threshold:
                exploded.append((name, stats["max"]))
        return exploded

    def remove_hooks(self):
        for handle in self.handles:
            handle.remove()
        self.handles = []


def save_checkpoint(
    model: nn.Module,
    tag: str,
    epoch: str,
    dir_models: str = ".",
    statistics: dict[str, float] | None = None,
):
    """Saves a model checkpoint to the specified directory. Model is saved to
    the file "{model_dir}/model_checkpoint_loss_{loss}.pt" along with its loss.

    Args:
        model (nn.Module): Model to be saved.
        tag (str): Model tag string.
        epoch (int): Training epoch number.
        dir_models (str, optional): Directory in which model checkpoints are
            stored. Defaults to ".".
    """
    path = get_model_path(epoch, tag, dir_models)
    if isinstance(model, DDP):
        model = model.module
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "statistics": statistics,
        },
        path,
    )


def load_checkpoint(
    model: nn.Module,
    tag: str,
    epoch: str,
    dir_models: str = ".",
):
    """Loads a checkpoint model from the specified path."""
    path = get_model_path(epoch, tag, dir_models)
    checkpoint = torch.load(path, weights_only=True)
    logger.info("Loaded model")
    model.load_state_dict(checkpoint["model_state_dict"])


def get_model_path(
    epoch: int,
    tag: str = "model",
    dir_models: str = ".",
    dir_results: str = ".",
) -> str:
    """Provide model checkpoint path."""
    directory = Path(dir_models)
    os.makedirs(directory, exist_ok=True)
    return directory / f"model_{tag}_epoch={str(epoch)}.pt"


def pyg_style_collate_fn(batch):
    """Collate function that separates inputs from targets.
    Returns: (input_dict, single_step_y, push_forward_y)"""

    batch_size = len(batch)

    # Get dimensions from first item (all should be same thanks to sampler)
    first = batch[0]
    bundle_size = first["field_data"].shape[-1]
    num_nodes = first["field_data"].shape[0]
    num_edges = first["edge_index"].shape[1]
    num_node_features = first["field_data"].shape[1]
    num_edge_features = first["edge_features"].shape[1]
    push_forward_count = first["push_forward_count"]

    # Concatenate inputs
    x_concat = torch.cat([data["field_data"] for data in batch], dim=0)

    # Build batched edge index
    edge_list = []
    for i in range(batch_size):
        edge_list.append(batch[i]["edge_index"] + (i * num_nodes))
    edge_index_concat = torch.cat(edge_list, dim=1)

    # Static features from first item
    edge_features = first["edge_features"].repeat(batch_size, 1)
    one_hot_node_type = first["one_hot_node_type"].repeat(batch_size, 1)
    mask_concat = first["mask"].repeat(batch_size)

    # Stack params
    params_concat = torch.stack([data["params"] for data in batch], dim=0)

    # Prepare targets separately
    single_step_y = torch.cat([data["single_step_y"] for data in batch], dim=0)

    pf_list = [
        data["push_forward_y"] for data in batch if "push_forward_y" in data and data["push_forward_y"].numel() > 0
    ]
    push_forward_y = torch.cat(pf_list, dim=0) if pf_list else None

    input_dict = {
        "field_data": x_concat,
        "edge_index": edge_index_concat,
        "edge_features": edge_features,
        "one_hot_node_type": one_hot_node_type,
        "mask": mask_concat,
        "params": params_concat,
        "push_forward_count": push_forward_count,
        "batch_size": batch_size,
        "bundle_size": bundle_size,
        "num_nodes": num_nodes,
        "num_edges": num_edges,
        "num_node_features": num_node_features,
        "num_edge_features": num_edge_features,
    }

    return input_dict, single_step_y, push_forward_y
