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

"""
Standalone rollout evaluation and movie generation.

Usage:
    python -m hspn.dogn.run_rollout rollout_config.yaml
    python -m hspn.dogn.run_rollout rollout_config.yaml --movie --field Density
"""
import argparse
from pathlib import Path

import torch
import yaml
from loguru import logger

from hspn.dogn.rollout_evaluation import run_evaluation, resolve_baseline_vtu


def main():
    parser = argparse.ArgumentParser(description="Run rollout evaluation")
    parser.add_argument("config", help="Rollout config YAML")
    parser.add_argument("--movie", action="store_true", help="Generate movie from PVD")
    parser.add_argument("--field", default=None, help="Field name for movie (default: all fields)")
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--quality", type=int, default=8)
    parser.add_argument("--cmap", default="viridis")
    args = parser.parse_args()

    rollout_config = yaml.safe_load(Path(args.config).read_text())
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
    baseline_vtu = resolve_baseline_vtu(rollout_config, training_cfg)

    # Resolve node_file and edge_file: rollout config first, then training config
    dataset_cfg = training_cfg.get("dataset", training_cfg)
    first_dataset = (dataset_cfg.get("train_datasets") or [{}])[0]

    node_file = rollout_config.get("node_file", first_dataset.get("node_file"))
    edge_file = rollout_config.get("edge_file", first_dataset.get("edge_file"))
    if not node_file or not edge_file:
        raise ValueError(
            "node_file and edge_file must be specified either in the rollout config "
            "or in the training config's dataset entries."
        )

    result = run_evaluation(
        model_path=model_path,
        rollout_dir=rollout_config["rollout_dir"],
        node_file=node_file,
        edge_file=edge_file,
        baseline_vtu=baseline_vtu,
        output_path=rollout_config.get("save_as_name", "rollout_output.pvd"),
        training_cfg=training_cfg,
        start_idx=start_idx,
    )

    logger.info(f"Average MSE: {result['avg_mse']:.4f}")
    if result["pvd_path"]:
        logger.info(f"Saved to {result['pvd_path']}")

    if args.movie and result["pvd_path"]:
        from hspn.dogn.rollout_movie import create_movie_from_pvd

        dataset_cfg = training_cfg.get("dataset", training_cfg)
        fields = [args.field] if args.field else dataset_cfg.get("fields", [])
        if not fields:
            fields = training_cfg.get("fields", ["field_0"])

        for field in fields:
            movie_path = Path(result["pvd_path"]).parent / f"{Path(result['pvd_path']).stem}_{field}.mp4"
            create_movie_from_pvd(
                pvd_file=result["pvd_path"],
                output_file=str(movie_path),
                prediction_field=f"predicted_{field}",
                truth_field=f"ground_truth_{field}",
                residual_field=f"residual_{field}",
                fps=args.fps,
                quality=args.quality,
                cmap=args.cmap,
            )
            logger.info(f"Movie: {movie_path}")


if __name__ == "__main__":
    main()