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


import logging
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import hydra
import torch
import torch.nn as nn
from aim import Run
from omegaconf import DictConfig, OmegaConf
from ray.train import Checkpoint, ScalingConfig, report
from ray.train.torch import TorchTrainer

from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader

from hspn.context import Context
from hspn.dogn.data_pipeline import get_dataset
from hspn.dogn.classes import get_model
from hspn.dogn.utils import (
    GraphBatchSampler,
    pyg_style_collate_fn,
)

from hspn.dogn.loss import CompositeLoss, EnergyLoss
from hspn.train_utils import save_checkpoint
from hspn.dogn.rollout_evaluation import run_evaluation, resolve_baseline_vtu

logger = logging.getLogger(__name__)


@dataclass
class TrainGNNConfig:
    """Training config for GNN experiments."""

    seed: int
    n_epochs: int
    checkpoint_dir: Path
    learning_rate: float
    batch_size: int
    log_interval: int
    model: nn.Module
    dataloader: DataLoader
    val_dataloader: DataLoader
    optimizer: Optimizer
    scheduler: LRScheduler
    tracker: Optional[Run]
    extra: Optional[Any] = None

    def __post_init__(self) -> None:
        self.checkpoint_dir = Path(self.checkpoint_dir)

    @classmethod
    def from_cfg(cls, cfg: DictConfig) -> "TrainGNNConfig":
        """Contructs from dictconfig."""
        cfg_dict = OmegaConf.to_container(cfg, resolve=True)
        tcfg = cfg.training  # shorthand

        # Merge training hyperparams into dataset config for get_dataset.
        # get_dataset expects bundle, push_forward, noise, etc. in the dict it receives.
        dataset_cfg = OmegaConf.to_container(cfg.dataset, resolve=True)
        for key in ("bundle", "push_forward", "noise", "bundle_style", "max_samples_per_field_dir"):
            if key in cfg_dict.get("training", {}):
                dataset_cfg[key] = cfg_dict["training"][key]

        # build datasets
        train_dataset = get_dataset("train", dataset_cfg)
        train_stats = {"mean": train_dataset.mean, "var": train_dataset.std**2}
        val_dataset = get_dataset("valid", dataset_cfg, train_stats=train_stats)

        # build dataloaders
        dataloader = hydra.utils.instantiate(
            tcfg.dataloader,
            dataset=train_dataset,
            collate_fn=pyg_style_collate_fn,
            batch_sampler=GraphBatchSampler(train_dataset, batch_size=tcfg.batch_size, shuffle=True),
        )
        val_dataloader = hydra.utils.instantiate(
            tcfg.val_dataloader,
            dataset=val_dataset,
            collate_fn=pyg_style_collate_fn,
            batch_sampler=GraphBatchSampler(val_dataset, batch_size=tcfg.batch_size, shuffle=False),
        )

        # build model
        cfg_dict["fields"] = train_dataset.fields
        cfg_dict["bundle"] = train_dataset.bundle
        cfg_dict["dataset"] = dataset_cfg  # save merged config for checkpoint
        model = get_model(cfg_dict, n_types=train_dataset.n_node_types)
        
        optimizer = hydra.utils.instantiate(tcfg.optimizer, params=model.parameters())
        scheduler = hydra.utils.instantiate(tcfg.scheduler, optimizer=optimizer)
        return TrainGNNConfig(
            seed=tcfg.seed,
            n_epochs=tcfg.n_epochs,
            checkpoint_dir=cfg.checkpoint_dir,
            learning_rate=tcfg.learning_rate,
            batch_size=tcfg.batch_size,
            log_interval=tcfg.log_interval,
            model=model,
            dataloader=dataloader,
            val_dataloader=val_dataloader,
            optimizer=optimizer,
            scheduler=scheduler,
            tracker=hydra.utils.instantiate(cfg.tracker) if Context.get().is_main_process else None,
            extra={
                "config": cfg_dict,
                "statistics": train_stats,
            },
        )
        


def train_loop_per_worker(cfg_dict: Dict[str, Any]) -> None:
    ctx = Context()
    cfg = OmegaConf.create(cfg_dict)
    training_dict = cfg_dict.get("training", {})
    loss_cfg = training_dict.get("loss", {})
    OmegaConf.resolve(cfg)

    config = TrainGNNConfig.from_cfg(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = config.model.to(device)
    train_loader = config.dataloader
    val_loader = config.val_dataloader
    optimizer = config.optimizer
    scheduler = config.scheduler
    tracker = config.tracker

    # Setup loss function
    energy_loss_fn = None
    
    # Resume from checkpoint
    resume_from = cfg_dict.get("resume_from")
    
    if resume_from:
        ckpt = torch.load(resume_from, map_location="cpu")
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt.get("epoch", 0)
        logger.info(f"Resumed from {resume_from} at epoch {start_epoch}")
    else:
        start_epoch = 0

    # Energy loss setup
    energy_loss_fn = None
    energy_cfg = loss_cfg.get("energy", {})
    if energy_cfg.get("enabled", False):
        fields = cfg_dict.get("dataset", {}).get("fields", [])
        vel_indices = energy_cfg.get("velocity_indices", [])
        buo_idx = energy_cfg.get("buoyancy_idx")

        if vel_indices and buo_idx is not None:
            node_positions = cfg_dict.get("dataset", {}).get("node_positions", ["x", "y", "z"])
            up_dim = energy_cfg.get("up_dimension", "y")
            up_idx = node_positions.index(up_dim)
            sample = train_loader.dataset[0]
            node_heights = sample[0]["pos"][:, up_idx] if isinstance(sample, tuple) else sample["pos"][:, up_idx]
            energy_loss_fn = EnergyLoss(vel_indices, buo_idx, node_heights).to(device)
            logger.info(f"Energy loss enabled: vel={vel_indices}, buo={buo_idx}, up={up_dim}")
        else:
            logger.warning("Energy loss enabled but missing velocity_indices or buoyancy_idx")

    # Composite loss
    loss_fn = CompositeLoss(
        config=training_dict,
        energy_loss_fn=energy_loss_fn,
        ).to(device)
    
    # Validation is always MSE
    val_criterion = nn.MSELoss()

    if hasattr(optimizer, "train"):
        optimizer.train()

    val_loss = float("inf")
    try:
        if tracker:
            tracker["hparams"] = cfg_dict
            for t in cfg_dict.get("tags", []):
                tracker.add_tag(t)
            tracker.description = cfg_dict.get("description", "")
        
        # Save resolved config alongside checkpoints
        if ctx.is_main_process:
            config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
            config_save_path = config.checkpoint_dir / "config.yaml"
            OmegaConf.save(cfg, config_save_path)
            logger.info(f"Saved config to {config_save_path}")
        
        # Early stopping
        best_val_loss = float("inf")
        patience_counter = 0
        es_cfg = training_dict.get("early_stopping", {})
        patience = es_cfg.get("patience", 0)  # 0 = disabled
        min_delta = es_cfg.get("min_delta", 0.0)

        push_forward_count = training_dict.get("push_forward", 0)
        
        for epoch in range(start_epoch+1, config.n_epochs + 1):
            model.train()
            train_losses: list[float] = []
            single_step_losses = {}
            push_forward_losses = {}
             

            for data, single_step_y, push_forward_y in train_loader:

                data = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in data.items()}
                
                optimizer.zero_grad()
                loss_fn.set_graph(data["edge_index"], data["edge_features"], data["num_nodes"])
                original_field = data["field_data"].clone()
                
                model._dirichlet_masks=data.get("dirichlet_masks")
                model._mask = data.get("mask") 
                
                
                # --- Phase 1: single-step loss (always) ---
                
                # Encode, process, decode
                node_enc, edge_enc, node_enc_dec, latent_ctx = model.encode_to_latent(**data)
                node_enc_mp, edge_enc = model.message_passing_series(node_enc, edge_enc, latent_ctx)
                pred = model.decode_from_latent(node_enc_mp, latent_ctx)

                m = loss_fn(pred, single_step_y.to(device), ae_pred=node_enc_dec,
                            field_data=data["field_data"], edge_index=data["edge_index"],
                            epoch=epoch)
                loss = m["loss"]
                
                for name, value in m.items():
                    v=value.item() if isinstance(value, torch.Tensor) else value
                    single_step_losses[name] = v + single_step_losses.get(name,0)
                
                # --- Phase 2: push-forward loss (additive) ---
                m_pf={}
                if push_forward_count > 0:
                    with torch.inference_mode():
                        for _ in range(push_forward_count):
                            data["field_data"] = model(**data).detach()
                    
                    # Encode, process, decode
                    node_enc, edge_enc, _, latent_ctx = model.encode_to_latent(**data)
                    node_enc_mp, edge_enc = model.message_passing_series(node_enc, edge_enc, latent_ctx)
                    pred_pf = model.decode_from_latent(node_enc_mp, latent_ctx)
                    
                    node_enc_pf,_,_,_ = model.encode_to_latent(**{**data, "field_data":push_forward_y.to(device)})
                    #pred_pf = model(**data)

                    # Build sequences for spectral loss
                    #true_seq = torch.cat([original_field, single_step_y.to(device), push_forward_y.to(device)], dim=-1)
                    #pred_seq = torch.cat([original_field, pred.detach(), pred_pf], dim=-1)
                    
                    m_pf = loss_fn(pred_pf, 
                                    push_forward_y.to(device),
                                    enc_y_pred = node_enc_mp, 
                                    enc_y_true = node_enc_pf.detach(),
                                    field_data=data["field_data"], 
                                    edge_index=data["edge_index"],
                                    epoch=epoch
                                    # energy_data=data, 
                                    # pred_seq=pred_seq, 
                                    # true_seq=true_seq,
                               )
                    loss = loss + m_pf["loss"]
                    
                
                train_losses.append(loss.item())
               
                for name, value in m_pf.items():
                    v = value.item() if isinstance(value, torch.Tensor) else value
                    push_forward_losses[name]=v + push_forward_losses.get(name,0) 
                    
                    
                # --- End: Back Propagation ---
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10)
                optimizer.step()
                
            # --- Post: Tracking ---
            train_loss = sum(train_losses) / len(train_losses)
            single_step_losses = {name: value / len(train_losses) for name, value in single_step_losses.items()}
            push_forward_losses = {name: value / len(train_losses) for name, value in push_forward_losses.items()}
             
            if tracker:
                tracker.track(train_loss, name="loss", step=epoch, context={"subset": "train"})
                tracker.track(single_step_losses.get('loss',0), name="single_step_loss", step=epoch, context={"subset": "train"})
                tracker.track(push_forward_losses.get('loss',0), name="push_forward_loss", step=epoch, context={"subset": "train"})
                 
                for name, loss in single_step_losses.items():
                    tracker.track(loss, f"ss_{name}", step=epoch, context={'subset':'train'})
                for name, loss in push_forward_losses.items():
                    tracker.track(loss, f"pf_{name}", step=epoch, context={'subset':'train'})
                         
            # --- Validation ----
            model.eval()

            if hasattr(optimizer, "eval"):  # schedulefree
                optimizer.eval()

            val_losses: list[float] = []
            
            with torch.no_grad():
                for data, single_step_y, push_forward_y in val_loader:
                    data = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in data.items()}

                    if push_forward_count > 0:
                        for _ in range(push_forward_count):
                            data["field_data"] = model(**data)
                        target = push_forward_y.to(device)
                    else:
                        target = single_step_y.to(device)
                    preds = model(**data)
                        
                    val_loss = val_criterion(preds, target)
                    val_losses.append(val_loss.item())
                
            val_loss = sum(val_losses) / len(val_losses)
            
            
            if tracker:
                tracker.track(val_loss, name="loss", step=epoch, context={"subset": "val"})
            
            
            # output to screen
            logger.info("[%d] train loss: %f  (total/mse/ae/latent): (%f/%f/%f/%f) (%f/%f/%f/%f),   val loss: %f", epoch
                        , train_loss
                        , single_step_losses['loss']
                        , single_step_losses.get('mse',0) 
                        , single_step_losses.get('ae',0)
                        , single_step_losses.get('latent',0)
                        , push_forward_losses['loss']
                        , push_forward_losses.get('mse',0) 
                        , push_forward_losses.get('ae',0)
                        , push_forward_losses.get('latent',0)
                        , val_loss)

            scheduler.step()

            # Stop?
            if patience > 0:
                if val_loss < best_val_loss - min_delta:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        logger.info(f"Early stopping at epoch {epoch} (best val_loss: {best_val_loss:.6f})")
                        break

            # Save?
            if (epoch % 10 == 0  or epoch==config.n_epochs) and ctx.is_main_process:
                metrics = {"loss": train_loss, "val_loss": val_loss}
                if cfg_dict.get("trainer"):
                    with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
                        save_checkpoint(
                            Path(temp_checkpoint_dir, "model.pt"),
                            model=model, optimizer=optimizer,
                            scheduler=scheduler, gradscaler=None,
                            epoch=epoch, metrics={"epoch": epoch},
                            extra=config.extra,
                        )
                        checkpoint = Checkpoint.from_directory(temp_checkpoint_dir)
                        report(metrics, checkpoint=checkpoint)
                else:
                    config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
                    tag = cfg_dict.get("tag", "model")
                    experiment = cfg_dict.get("tracker", {})
                    exp_name = experiment.get("experiment", "") if isinstance(experiment, dict) else ""
                    vl_str = f"_vl{val_loss:.4f}" if val_loss is not None else ""
                    prefix = f"{exp_name}_" if exp_name else ""
                    ckpt_name = f"{prefix}{tag}_epoch{epoch}{vl_str}.pt"
                    save_checkpoint(
                        config.checkpoint_dir / ckpt_name,
                        model=model, optimizer=optimizer,
                        scheduler=scheduler, gradscaler=None,
                        epoch=epoch, metrics=metrics,
                        extra=config.extra,
                    )
            #switch back to training
            if hasattr(optimizer, "train"):  # schedulefree
                optimizer.train()
            
    except Exception:
        logger.exception("Training failed")
        return float("inf")

    finally:
        if tracker:
            tracker.close()
            
        # === Post-training rollout evaluation ===
        rollout_cfg = cfg_dict.get("rollout", {})
        if rollout_cfg.get("enabled", False) and ctx.is_main_process:
            logger.info("Running post-training rollout evaluation...")

            # Validate required keys
            missing = [k for k in ("rollout_dir", "node_file", "edge_file") if k not in rollout_cfg]
            if missing:
                logger.error(f"Rollout config missing required keys: {missing}. Skipping rollout.")
            else:
                # Find the latest checkpoint
                ckpt_dir = config.checkpoint_dir
                ckpt_files = sorted(ckpt_dir.glob("*.pt"), key=lambda p: p.stat().st_mtime)
                if not ckpt_files:
                    logger.error("No checkpoints found. Skipping rollout.")
                else:
                    latest_ckpt = ckpt_files[-1]
                    logger.info(f"Using checkpoint: {latest_ckpt}")

                    output_dir = Path(rollout_cfg.get("output_dir", "./rollouts"))
                    output_dir.mkdir(parents=True, exist_ok=True)
                    output_path = output_dir / f"rollout_epoch{epoch}"
                    dataset_cfg = cfg_dict.get("dataset", {})
                    baseline_vtu = resolve_baseline_vtu(rollout_cfg, cfg_dict)
                    
                    try:
                        result = run_evaluation(
                            model_path=str(latest_ckpt),
                            rollout_dir=rollout_cfg["rollout_dir"],
                            node_file=rollout_cfg["node_file"],
                            edge_file=rollout_cfg["edge_file"],
                            baseline_vtu=baseline_vtu,
                            output_path=str(output_path),
                            training_cfg=dataset_cfg,
                            start_idx=rollout_cfg.get("start_idx", 0),
                        )
                        logger.info(f"Rollout MSE: {result['avg_mse']:.6f}")
                        if result["pvd_path"]:
                            logger.info(f"Rollout PVD: {result['pvd_path']}")

                        if tracker:
                            tracker.track(result["avg_mse"], name="rollout/mse", step=epoch)

                        # Generate movie if configured and PVD was produced
                        movie_cfg = rollout_cfg.get("movie", {})
                        if movie_cfg.get("enabled", False) and result["pvd_path"]:
                            from hspn.dogn.rollout_movie import create_movie_from_pvd

                            fields = dataset_cfg.get("fields", cfg_dict.get("fields", []))
                            for field in fields:
                                movie_path = output_dir / f"rollout_epoch{epoch}_{field}.mp4"
                                create_movie_from_pvd(
                                    pvd_file=result["pvd_path"],
                                    output_file=str(movie_path),
                                    prediction_field=f"predicted_{field}",
                                    truth_field=f"ground_truth_{field}",
                                    residual_field=f"residual_{field}",
                                    fps=movie_cfg.get("fps", 24),
                                    quality=movie_cfg.get("quality", 8),
                                    cmap=movie_cfg.get("cmap", "viridis"),
                                )
                                logger.info(f"Movie saved: {movie_path}")

                    except Exception as e:
                        logger.error(f"Rollout evaluation failed: {e}")
                        logger.exception("Rollout traceback:")
                        
    return val_loss


@hydra.main(config_path="pkg://hspn.conf.dogn", config_name="train_gnn", version_base=None)
def main(cfg: DictConfig) -> None:
    OmegaConf.resolve(cfg)
    if trainer_cfg := cfg.get("trainer"):
        trainer = TorchTrainer(
            train_loop_per_worker,
            train_loop_config=OmegaConf.to_container(cfg),
            scaling_config=ScalingConfig(
                num_workers=trainer_cfg.num_workers,
                use_gpu=trainer_cfg.use_gpu,
                resources_per_worker=trainer_cfg.resources_per_worker,
            ),
        )
        result = trainer.fit()
    else:
        result = train_loop_per_worker(OmegaConf.to_container(cfg))
    print(f"RESULT: {result}")
    return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
