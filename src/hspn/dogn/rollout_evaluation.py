#!/bin/sh
#
# HyperSPIN code - hspn_surrogate_models
# 
# Copyright 2025 Carnegie Mellon University.
# 
# NO WARRANTY. THIS CARNEGIE MELLON UNIVERSITY AND SOFTWARE ENGINEERING INSTITUTE MATERIAL IS FURNISHED ON AN "AS-IS" BASIS. CARNEGIE MELLON UNIVERSITY MAKES NO WARRANTIES OF ANY KIND, EITHER EXPRESSED OR IMPLIED, AS TO ANY MATTER INCLUDING, BUT NOT LIMITED TO, WARRANTY OF FITNESS FOR PURPOSE OR MERCHANTABILITY, EXCLUSIVITY, OR RESULTS OBTAINED FROM USE OF THE MATERIAL. CARNEGIE MELLON UNIVERSITY DOES NOT MAKE ANY WARRANTY OF ANY KIND WITH RESPECT TO FREEDOM FROM PATENT, TRADEMARK, OR COPYRIGHT INFRINGEMENT.
# 
# Licensed under a MIT (SEI)-style license, please see license.txt or contact permission@sei.cmu.edu for full terms.
# 
# [DISTRIBUTION STATEMENT A] This material has been approved for public release and unlimited distribution.  Please see Copyright notice for non-US Government use and distribution.
# 
# This Software includes and/or makes use of Third-Party Software each subject to its own license.
# 
# DM25-0396
 

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pyvista as pv
import torch
import yaml
from loguru import logger
from numpy.lib.recfunctions import \
    structured_to_unstructured as to_unstructured
from tqdm import tqdm
from hspn.dogn.graph_utils import (apply_spatial_crop, create_one_hot_encoding, create_update_mask,
                                    create_spatial_mask, create_bc_masks, get_edge_features,
                                    load_node_data)
from hspn.dogn.classes import get_model

def rollout_stay_latent(model, data, n_steps, visualize_every=1):
    """Rollout in latent space with re-encoding to stay on manifold."""
    node_enc, edge_enc, _, latent_ctx = model.encode_to_latent(
        data["field_data"], data["one_hot_node_type"],
        data["edge_index"], data["edge_features"],
        data["params"], data["num_nodes"], data["num_edges"],
        mask=data["mask"],
        dirichlet_masks=data.get("dirichlet_masks"),
    )

    predictions = []
    for step in range(n_steps):
        node_enc, edge_enc = model.message_passing_series(node_enc, edge_enc, latent_ctx)

        # Decode to physical space
        pred = model.decode_from_latent(node_enc, latent_ctx)

        if step % visualize_every == 0:
            predictions.append(pred.detach().cpu())

        # Re-encode to project back onto manifold
        node_enc, edge_enc, _, latent_ctx = model.encode_to_latent(
            pred.detach(), data["one_hot_node_type"],
            data["edge_index"], data["edge_features"],
            data["params"], data["num_nodes"], data["num_edges"],
            mask=data["mask"],
            dirichlet_masks=data.get("dirichlet_masks"),
        )

    return predictions

#def rollout_stay_latent(model, data, n_steps, visualize_every=1):
#    """Rollout in latent space. Decode only for visualization."""
#    node_enc, edge_enc, _, latent_ctx = model.encode_to_latent(
#        data["field_data"], data["one_hot_node_type"],
#        data["edge_index"], data["edge_features"],
#        data["params"], data["num_nodes"], data["num_edges"],
#        mask=data["mask"],
#        dirichlet_masks=data.get("dirichlet_masks"),
#    )
#
#    predictions = []
#    for step in range(n_steps):
#        node_enc, edge_enc = model.message_passing_series(node_enc, edge_enc, latent_ctx)
#        if torch.isnan(node_enc).any():
#            print(f"NaN in node_enc at step {step}")
#            break
#        if step % visualize_every == 0:
#            print(f' rollout stay_latent step {step}')
#            with torch.no_grad():
#                pred = model.decode_from_latent(node_enc, latent_ctx)
#                predictions.append(pred.cpu())
#
#    return predictions


def resolve_baseline_vtu(rollout_cfg, training_cfg):
    """Get baseline VTU path from rollout config."""
    return rollout_cfg.get("baseline_vtu")


def load_model_get_normalization(model_dict):
    if "config" in model_dict:
        # Your original format
        config = model_dict["config"]
        state_dict = model_dict["model_state_dict"]
        statistics = model_dict["statistics"]
    elif "extra" in model_dict and model_dict["extra"] is not None:
        # Colin's format with our additions from Section 1
        config = model_dict["extra"]["config"]
        state_dict = model_dict["model_state_dict"]
        statistics = model_dict["extra"]["statistics"]
    else:
        raise ValueError(
            "Checkpoint missing config/statistics. "
            "Expected either top-level 'config' key or 'extra' dict with 'config' and 'statistics'."
        )

    history = 1 if config["model"]["deeponet_type"] =='colin' else config["bundle"]
    n_inputs = state_dict["node_encoder.0.weight"].shape[1]
    n_fields = len(config["fields"])
    
    if config.get("model",{}).get("time_aggregation") == "conv_compress":
        n_node_types = n_inputs - n_fields
    else:
        n_node_types = n_inputs//history - n_fields
    
    model = get_model(config, n_node_types)
    model.load_state_dict(state_dict)
    return model.eval(), statistics

def calc_energies(data, b, Y, field_idx):
    # will calculate ke, pe, pe_min if data is avaliable (e.g. smoke dataset)
    vx, vy = data[:, field_idx['vx']], data[:, field_idx['vy']]
    u = data[:, field_idx['u']]
    ke = 0.5 * np.sum(vx**2 + vy**2)
    pe = -b * np.sum(u * Y)
    pe_min = -b * np.sum(np.sort(u) * np.sort(Y))
    return ke, pe, pe_min



def save_time_series_as_pvd(
    base_filename,
    baseline_geometry_vtu,
    predictions,
    truth,
    nodes,
    times,
    config,
    statistics,
    param_data,
    crop_bounds=None,
):
    """Save time-dependent 2D surface graphs as PVD format."""
    base_filename = Path(base_filename)

    # Setup paths
    base_path = (
        base_filename.parent if base_filename.parent != Path(".") else Path.cwd()
    )
    base_name = base_filename.stem if base_filename.suffix else base_filename.name
    vtp_dir = base_path / f"{base_name}_files"
    vtp_dir.mkdir(exist_ok=True, parents=True)

    # Initialize PVD content
    pvd_content = [
        '<?xml version="1.0"?>\n',
        '<VTKFile type="Collection" version="0.1">\n',
        "  <Collection>\n",
    ]

    # Load and optionally crop baseline mesh
    baseline_mesh = pv.read(baseline_geometry_vtu)
    if crop_bounds:
        mask = create_spatial_mask(baseline_mesh.points, crop_bounds)
        baseline_mesh = baseline_mesh.extract_points(
            mask, include_cells=True, adjacent_cells=False
        )
        logger.info(f"Cropped baseline mesh to {baseline_mesh.n_points} points")
        assert baseline_mesh.n_points == predictions.size(
            1
        ), f"Points mismatch: {baseline_mesh.n_points} vs {predictions.shape[1]}"
        model._mask = mask
    # Validate field names
    field_names = config.get("fields", [])
    if len(field_names) != predictions.shape[-1]:
        raise ValueError(
            f"Field count mismatch: {len(field_names)} names vs {predictions.shape[-1]} dims"
        )

    # Unnormalize data
    mean = (
        statistics["mean"].cpu()
        if torch.is_tensor(statistics["mean"])
        else statistics["mean"]
    )
    var = (
        statistics["var"].cpu()
        if torch.is_tensor(statistics["var"])
        else statistics["var"]
    )
    std = np.sqrt(var)

    predictions_unnorm = predictions * std + mean
    truth_unnorm = truth * std + mean

    # Process each timestep
    for t in tqdm(range(len(times)), "saving files"):
        mesh = baseline_mesh.copy()
        pred_t, truth_t = predictions_unnorm[t], truth_unnorm[t]

        # Energy calculations (if fields available)
        if all(f in field_names for f in ['vx', 'vy', 'u']):
            field_idx = {f: i for i, f in enumerate(field_names)}
            Y = nodes[:, 1] if nodes.shape[1] >= 2 else nodes[:, 0]  # adjust as needed
            Y = (Y + 0.5) * 32 if np.allclose(max(Y)-min(Y),1) else Y
            b = param_data[0, 0].item() if hasattr(param_data, 'item') else float(param_data[0, 0])
            
            for data, prefix in [(pred_t, 'predicted_'), (truth_t, 'ground_truth_')]:
                din = data.numpy() if hasattr(data, 'numpy') else data
                ke, pe, pe_min = calc_energies(din, b, Y, field_idx) 
                mesh.field_data[f'{prefix}KE'] = [ke]
                mesh.field_data[f'{prefix}PE'] = [pe]
                mesh.field_data[f'{prefix}PE_min'] = [pe_min]

        # Add field data and residuals
        residuals = []
        for i, field in enumerate(field_names):
            mesh.point_data[f"predicted_{field}"] = pred_t[:, i]
            mesh.point_data[f"ground_truth_{field}"] = truth_t[:, i]
            residual = pred_t[:, i] - truth_t[:, i]
            mesh.point_data[f"residual_{field}"] = residual
            residuals.append(residual)

        # Combined residual for multiple fields
        if len(residuals) > 1:
            mesh.point_data["residual_combined"] = np.sqrt(
                np.mean([r**2 for r in residuals], axis=0)
            )

        mesh.point_data["time"] = np.full(len(mesh.points), times[t])
        mesh = mesh.point_data_to_cell_data()

        # Save VTU and update PVD
        vtu_filename = vtp_dir / f"{base_name}_{t:04d}.vtu"
        mesh.save(vtu_filename)
        pvd_content.append(
            f'    <DataSet timestep="{times[t]}" file="{base_name}_files/{vtu_filename.name}"/>\n'
        )

    pvd_content.extend(["  </Collection>\n", "</VTKFile>\n"])

    # Write PVD file
    pvd_filename = base_path / f"{base_name}.pvd"
    pvd_filename.write_text("".join(pvd_content))

    print(f"Time series saved to {pvd_filename}")
    return pvd_filename

def run_evaluation(model_path, rollout_dir, node_file, edge_file, baseline_vtu, output_path, training_cfg, start_idx=0):
    """
    Run model evaluation and save results to PVD.
    
    Args:
        model_path: Path to model checkpoint (.pt file)
        rollout_dir: Directory containing trajectory .npy files and params.json
        node_file: Path to nodes file
        edge_file: Path to edges file
        baseline_vtu: Path to baseline geometry VTU for visualization
        output_path: Where to save the PVD output
        training_cfg: Training configuration dict
        start_idx: Starting timestep index
    
    Returns:
        dict: {"pvd_path": str, "avg_mse": float}
    """
    # Flatten nested config structure if needed.
    # With the new Hydra config groups, keys like bundle/fields/dimensions
    # may be split across dataset and training sections.
    if "dataset" in training_cfg or "training" in training_cfg:
        flat = {}
        flat.update(training_cfg.get("dataset", {}))
        flat.update(training_cfg.get("training", {}))
        # fields and bundle are also set at top level by from_cfg
        for key in ("fields", "bundle"):
            if key in training_cfg:
                flat[key] = training_cfg[key]
        training_cfg = flat
    vtp_dir = Path(rollout_dir)
    node_file = Path(node_file)
    edge_file = Path(edge_file)
    output_path = Path(output_path)

    if not baseline_vtu:
        raise ValueError("baseline_geometry_vtu not specified")

    # Load parameters
    logger.debug("Getting parameters")
    params_path = vtp_dir / "params.json"
    params_dict = json.loads(params_path.read_text())
    param_keys = training_cfg.get("dataset", training_cfg).get("param_keys", [])
    if param_keys:
        missing = [k for k in param_keys if k not in params_dict]
        if missing:
            raise ValueError(
                f"{params_path} missing keys: {missing}. "
                f"Available keys: {list(params_dict.keys())}"
            )
        param_data = torch.tensor([params_dict[k] for k in param_keys], dtype=torch.float32)
    else:
        param_data = torch.tensor(list(params_dict.values()), dtype=torch.float32)
    param_data = param_data.reshape(1, -1)
    # Load model and statistics only
    model_dict = torch.load(model_path, weights_only=False, map_location="cpu")
    model, statistics = load_model_get_normalization(model_dict)

    # Use passed training_cfg
    bundle_size = training_cfg["bundle"]
    fields = training_cfg["fields"]
    dimensions = training_cfg["dimensions"]

    # Load graph data
    pos, node_type, point_id, node_bcs = load_node_data(node_file, dimensions)
    
    # Create dirichlet mask dictionary
    dirichlet_masks = None
    if training_cfg.get('boundary_conditions', {}).get('enabled', False) and node_bcs is not None:
        dirichlet_masks = create_bc_masks(node_bcs, training_cfg.get('fields', []))
    
    edges = np.load(edge_file)
    edges = (
        edges["edges"] if isinstance(edges, np.lib.npyio.NpzFile) else edges
    ).T.astype(np.int64)

    # Apply cropping if configured
    crop_config = training_cfg.get("bounds")
    if crop_config:
        logger.info(f"Applying crop with config: {crop_config}")
        pos, edges, crop_indices = apply_spatial_crop(pos, edges, crop_config)
        node_type = node_type[crop_indices] if node_type is not None else node_type
        logger.info(f"After cropping: {len(pos)} nodes")
    else:
        # Map point IDs for uncropped case
        point_id_to_row = {int(pid.item()): i for i, pid in enumerate(point_id)}
        edges = np.vectorize(point_id_to_row.get)(edges)
        crop_indices = None
        logger.info(f"No cropping: {len(pos)} nodes")

    # Prepare graph components
    one_hot_node_type, _ = create_one_hot_encoding(node_type)
    edge_index = torch.from_numpy(
        np.unique(np.concatenate([edges, edges[[1, 0]]], axis=1), axis=1)
    ).long()
    edge_features = get_edge_features(torch.from_numpy(pos), edge_index).float()

    n_nodes, n_edges, n_fields = pos.shape[0], edge_index.size(1), len(fields)

    # Load model and statistics
    model, statistics = load_model_get_normalization(model_dict)
    stay_latent = getattr(model, 'stay_latent', False)
    model._dirichlet_masks = dirichlet_masks

    # Load ground truth data
    logger.debug("loading ground truth data")
    ground_truth_list = []
    for f in sorted(vtp_dir.glob("*.npy")):
        data = to_unstructured(np.load(f)[fields]).copy()
        if crop_indices is not None:
            data = data[crop_indices]
        ground_truth_list.append(torch.from_numpy(data).float())

    # Reshape ground truth by bundles
    ground_truth = torch.stack(ground_truth_list)[start_idx:]
    if unused := ground_truth.size(0) % bundle_size:
        ground_truth = ground_truth[:-unused]
    ground_truth = ground_truth.view(-1, bundle_size, n_nodes, n_fields).permute(0, 2, 3, 1)

    # Normalize
    ground_truth = (ground_truth - statistics["mean"].view(1, 1, -1, 1)) / torch.sqrt(
        statistics["var"].view(1, 1, -1, 1)
    ).float()
    logger.info(f"Data mean: {ground_truth.mean():.4f}, std: {ground_truth.std():.4f}")

    total_time_steps = ground_truth.size(0) * ground_truth.size(3)

    # Setup device and move to GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    model = model.to(device)
    param_data = param_data.to(device)
    edge_features = edge_features.to(device)
    edge_index = edge_index.to(device)
    one_hot_node_type = one_hot_node_type.to(device)

    # Move dirichlet masks to device
    if dirichlet_masks is not None:
        dirichlet_masks = {k: v.to(device) for k, v in dirichlet_masks.items()}

    update_node_types = training_cfg.get("update_node_tags", [0])
    mask = create_update_mask(node_type, update_node_types)

    # Initialize predictions with Dirichlet applied
    initial = ground_truth[0].detach().cpu()
    if dirichlet_masks is not None:
        for field, dmask in dirichlet_masks.items():
            initial[dmask.cpu(), field, :] = 0
    predictions = [initial]
    
    mean_sqr_err_total = 0


    
    if stay_latent:
        print('here') 
        # Latent-space rollout
        data = {
            "field_data": ground_truth[0].to(device),
            "one_hot_node_type": one_hot_node_type,
            "edge_index": edge_index,
            "edge_features": edge_features,
            "params": param_data,
            "num_nodes": n_nodes,
            "num_edges": n_edges,
            "mask": mask,
            "dirichlet_masks": dirichlet_masks,
        }
        latent_preds = rollout_stay_latent(model, data, ground_truth.size(0) - 1)
        
        predictions = [ground_truth[0].detach().cpu()]
        for bundle_iter, pred in enumerate(latent_preds):
            print(pred.mean(), pred.max())
            truth_step = ground_truth[bundle_iter + 1]
            mse = torch.mean((pred - truth_step.cpu()) ** 2)
            mean_sqr_err_total += mse
            logger.debug(f"Step {bundle_iter + 1} MSE: {mse:.4f}")
            predictions.append(pred)
            
    else:
        # physical space decode/encode rollout
        for bundle_iter in range(1, ground_truth.size(0)):
            data_input = predictions[-1].to(device)
            prediction = model(
                field_data=data_input,
                one_hot_node_type=one_hot_node_type,
                edge_index=edge_index,
                edge_features=edge_features,
                params=param_data,
                mask=mask,
                num_nodes=n_nodes,
                num_edges=n_edges,
                dirichlet_masks=dirichlet_masks,
                pos=pos,
                return_step_enc_dec=False
            )        

            if dirichlet_masks is not None:
                for field, dmask in dirichlet_masks.items():
                    prediction[dmask, field, :] = 0

            truth_step = ground_truth[bundle_iter]
            mse = torch.mean((prediction.detach().cpu() - truth_step) ** 2)
            mean_sqr_err_total += mse
            logger.debug(f"Step {bundle_iter} MSE: {mse:.4f}")

            predictions.append(prediction.detach().cpu())

    # Reshape predictions
    predictions = (
        torch.stack(predictions).permute(0, 3, 1, 2).reshape(-1, n_nodes, n_fields)
    )
    ground_truth_flat = ground_truth.permute(0, 3, 1, 2).reshape(-1, n_nodes, n_fields)

    avg_mse = (mean_sqr_err_total / (ground_truth.size(0) - 1)).item()
    logger.info(f"Average MSE: {avg_mse:.4f}")

    # Prepare positions for visualization
    if pos.shape[1] == 2:
        pos = np.pad(pos, ((0, 0), (0, 1)))

    # Save visualization
    save_time_series_as_pvd(
        output_path,
        baseline_vtu,
        predictions[:ground_truth_flat.size(0)],
        ground_truth_flat,
        pos,
        np.arange(total_time_steps),
        training_cfg,
        statistics,
        param_data.cpu(),
        crop_config,
    )

    return {
        "pvd_path": str(output_path),
        "avg_mse": avg_mse,
    }

def main(config_path):
    """CLI entry point."""
    rollout_config = yaml.safe_load(Path(config_path).read_text())
    logger.info(rollout_config)

    model_path = rollout_config["model"]
    
    # Load training config from checkpoint (handle both formats)
    checkpoint = torch.load(model_path, weights_only=False, map_location="cpu")
    if "config" in checkpoint:
        training_cfg = checkpoint["config"]
    elif "extra" in checkpoint and checkpoint["extra"] is not None:
        training_cfg = checkpoint["extra"]["config"]
    else:
        raise ValueError("Cannot find training config in checkpoint")
     
    start_idx = rollout_config.get("start_idx", training_cfg.get("start_iter", 0))

    result = run_evaluation(
        model_path=model_path,
        rollout_dir=rollout_config["rollout_dir"],
        node_file=rollout_config["node_file"],
        edge_file=rollout_config["edge_file"],
        baseline_vtu=rollout_config.get("baseline_geometry_vtu"),
        output_path=rollout_config.get("save_as_name", "rollout_output.pvd"),
        training_cfg=training_cfg,
        start_idx=start_idx,
    )
    
    logger.info(f"Saved to {result['pvd_path']}")
    logger.info(f"Average MSE: {result['avg_mse']:.4f}")

if __name__ == "__main__":
    if len(sys.argv) <= 1:
        raise ValueError("Config file path required as argument")
    main(sys.argv[1])
