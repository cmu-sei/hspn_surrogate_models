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
from hspn.dogn.data import get_dataset
from hspn.dogn.classes import get_model
from hspn.dogn.utils import (
    GraphBatchSampler,
    pyg_style_collate_fn,
)

from hspn.dogn.loss import LossFunction, EnergyLoss, forward_loss
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
        # build datasets
        train_dataset = get_dataset("train", cfg.dataset)
        train_stats = {"mean": train_dataset.mean, "var": train_dataset.std**2}
        val_dataset = get_dataset("valid", cfg.dataset, train_stats=train_stats)

        # build dataloaders
        dataloader = hydra.utils.instantiate(
            cfg.dataloader,
            dataset=train_dataset,
            collate_fn=pyg_style_collate_fn,
            batch_sampler=GraphBatchSampler(train_dataset, batch_size=cfg.batch_size, shuffle=True),
        )
        val_dataloader = hydra.utils.instantiate(
            cfg.val_dataloader,
            dataset=val_dataset,
            collate_fn=pyg_style_collate_fn,
            batch_sampler=GraphBatchSampler(val_dataset, batch_size=cfg.batch_size, shuffle=False),
        )

        # build model
                
        cfg_dict = OmegaConf.to_container(cfg, resolve=True)
        cfg_dict["fields"] = train_dataset.fields
        cfg_dict["bundle"] = train_dataset.bundle
        model = get_model(cfg_dict, n_types=train_dataset.n_node_types)
        #d_inp = len(train_dataset.fields)
        #d_out = len(train_dataset.fields) * train_dataset.bundle
        #n_typ = train_dataset.n_node_types

        #model = EPDOModel(
        #    d_inp=d_inp,
        #    d_out=d_out,
        #    n_typ=n_typ,
        #    hidden_channels=cfg.model.hidden_channels,
        #    mlp_width=cfg.model.mlp_width,
        #    mlp_depth=cfg.model.mlp_depth,
        #    dropout=cfg.model.dropout,
        #    n_processing_blocks=cfg.model.n_processing_blocks,
        #    checkpointed=cfg.model.checkpointed,
        #    activation=cfg.model.activation,
        #    reweight=cfg.model.reweight,
        #)

        # build optimizer + scheduler
        optimizer = hydra.utils.instantiate(cfg.optimizer, params=model.parameters())
        scheduler = hydra.utils.instantiate(cfg.scheduler, optimizer=optimizer)
        return TrainGNNConfig(
            seed=cfg.seed,
            n_epochs=cfg.n_epochs,
            checkpoint_dir=cfg.checkpoint_dir,
            learning_rate=cfg.learning_rate,
            batch_size=cfg.batch_size,
            log_interval=cfg.log_interval,
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
    loss_cfg = cfg_dict.get("loss", {})
    loss_type = loss_cfg.get("type", "mse")
    OmegaConf.resolve(cfg)

    config = TrainGNNConfig.from_cfg(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = config.model.to(device)
    train_loader = config.dataloader
    val_loader = config.val_dataloader
    optimizer = config.optimizer
    scheduler = config.scheduler
    criterion = nn.MSELoss()
    tracker = config.tracker

    # Setup loss function
    energy_loss_fn = None
    _graph_set = False

    if loss_type == "gradient_weighted":
        gw_cfg = loss_cfg.get("gradient_weighted", {})
        n_fields = len(cfg_dict.get("dataset", {}).get("fields", cfg_dict.get("fields", [])))
        if n_fields < 2:
            logger.error(
                f"gradient_weighted loss requires at least 2 fields (got {n_fields}). "
                "It assumes the last two fields are velocity components. Falling back to MSE."
            )
            loss_type = "mse"
        else:
            criterion = LossFunction(lambda_gradient=gw_cfg.get("lambda_gradient", 2.0))

    elif loss_type == "energy":
        en_cfg = loss_cfg.get("energy", {})
        velocity_indices = en_cfg.get("velocity_indices", [])
        buoyancy_idx = en_cfg.get("buoyancy_idx")
        if velocity_indices and buoyancy_idx is not None:
            logger.warning(
                "EnergyLoss requires node_heights from the mesh. "
                "This needs manual setup based on your geometry."
            )
            energy_loss_fn = None 
        else:
            logger.warning("Energy loss configured but missing velocity_indices or buoyancy_idx. Using MSE.")
            loss_type = "mse"
            
        
    if hasattr(optimizer, "train"):  # schedulefree
        optimizer.train()  # type: ignore

    try:
        if tracker:
            tracker["hparams"] = cfg_dict
        # Save resolved config alongside checkpoints
        if ctx.is_main_process:
            config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
            config_save_path = config.checkpoint_dir / "config.yaml"
            OmegaConf.save(cfg, config_save_path)
            logger.info(f"Saved config to {config_save_path}")
        for epoch in range(1, config.n_epochs + 1):
            model.train()
            train_losses: list[float] = []

            for data, single_step_y, push_forward_y in train_loader:
                data = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in data.items()}

                # pushforward rollout
                push_forward_count = data.get("push_forward_count", 0)
                with torch.inference_mode():
                    for _ in range(push_forward_count):
                                                
                        data["field_data"], _ = model(**data)
                        data["field_data"] = data["field_data"].detach()
                        #data["field_data"] = model(**data).detach()

                target = push_forward_y.to(device) if push_forward_count > 0 else single_step_y.to(device)

                optimizer.zero_grad()                               
                #preds, _ = model(**data)
                ##preds = model(**data)
                #loss = criterion(preds, target)
                
                # Set graph for gradient-weighted loss (first batch only)
                if loss_type == "gradient_weighted" and not _graph_set:
                    criterion.set_graph(
                        data["edge_index"], data["edge_features"], data["num_nodes"]
                    )
                    _graph_set = True

                if energy_loss_fn is not None:
                    energy_cfg = cfg_dict.get("loss", {}).get("energy", {})
                    field_names = cfg_dict.get("dataset", {}).get("fields", cfg_dict.get("fields"))
                    result = forward_loss(
                        model, data, target, nn.MSELoss(),
                        energy_loss_fn, energy_cfg, field_names
                    )
                    loss = result["loss"]
                    if tracker and epoch % config.log_interval == 0:
                        for key in ["mse", "ke", "ape", "pe_min", "mono"]:
                            tracker.track(result[key], name=f"loss/{key}", step=epoch, context="train")
                else:
                    preds, _ = model(**data)
                    loss = criterion(preds, target)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1e8)
                optimizer.step()

                train_losses.append(loss.item())

            train_loss = sum(train_losses) / len(train_losses)
            if tracker:
                tracker.track(train_loss, name="loss/train", step=epoch, context="train")

            model.eval()
            val_losses: list[float] = []
            with torch.no_grad():
                for data, single_step_y, push_forward_y in val_loader:
                    data = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in data.items()}

                    push_forward_count = data.get("push_forward_count", 0)
                    for _ in range(push_forward_count + 1):
                        data["field_data"],_ = model(**data)

                    y_hat = data["field_data"]
                    val_loss = criterion(y_hat, push_forward_y.to(device))
                    val_losses.append(val_loss.item())

            val_loss = sum(val_losses) / len(val_losses)
            if tracker:
                tracker.track(val_loss, name="loss/val", step=epoch, context="train")
            logger.info("[%d] val loss: %f", epoch, val_loss)

            scheduler.step()


            if epoch % 10 == 0 and ctx.is_main_process:
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
            #if epoch % 10 == 0 and ctx.is_main_process:
            #    from ray import train

            #    metric  {"loss": train_loss, "val_loss": val_loss}
            #    with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
            #        from hspn.train_utils import save_checkpoint

            #        save_checkpoint(
            #            Path(temp_checkpoint_dir, "model.pt"),
            #            model=model,
            #            optimizer=optimizer,
            #            scheduler=scheduler,
            #            gradscaler=None,
            #            epoch=epoch,
            #            metrics={"epoch": epoch},
            #        )
            #        checkpoint = Checkpoint.from_directory(temp_checkpoint_dir)
            #        train.report(metrics, checkpoint=checkpoint)
    except:
        logger.exception("Failed")
        raise
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

                    dataset_cfg = cfg_dict.get("dataset", cfg_dict)
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


@hydra.main(config_path="pkg://hspn.conf", config_name="train_gnn", version_base=None)
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
    print(result)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
