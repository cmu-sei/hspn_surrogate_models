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
from ray.train import Checkpoint, ScalingConfig
from ray.train.torch import TorchTrainer
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader

from hspn.context import Context
from hspn.dogn.data import get_dataset
from hspn.dogn.epdo import EPDOModel
from hspn.dogn.utils import (
    GraphBatchSampler,
    pyg_style_collate_fn,
)

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
        d_inp = len(train_dataset.fields)
        d_out = len(train_dataset.fields) * train_dataset.bundle
        n_typ = train_dataset.n_node_types

        model = EPDOModel(
            d_inp=d_inp,
            d_out=d_out,
            n_typ=n_typ,
            hidden_channels=cfg.model.hidden_channels,
            mlp_width=cfg.model.mlp_width,
            mlp_depth=cfg.model.mlp_depth,
            dropout=cfg.model.dropout,
            n_processing_blocks=cfg.model.n_processing_blocks,
            checkpointed=cfg.model.checkpointed,
            activation=cfg.model.activation,
            reweight=cfg.model.reweight,
        )

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
        )


def train_loop_per_worker(cfg_dict: Dict[str, Any]) -> None:
    ctx = Context()
    cfg = OmegaConf.create(cfg_dict)
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

    if hasattr(optimizer, "train"):  # schedulefree
        optimizer.train()  # type: ignore

    try:
        if tracker:
            tracker["hparams"] = cfg_dict
        for epoch in range(1, config.n_epochs + 1):
            model.train()
            train_losses: list[float] = []

            for data, single_step_y, push_forward_y in train_loader:
                data = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in data.items()}

                # pushforward rollout
                push_forward_count = data.get("push_forward_count", 0)
                with torch.inference_mode():
                    for _ in range(push_forward_count):
                        data["field_data"] = model(**data).detach()

                target = push_forward_y.to(device) if push_forward_count > 0 else single_step_y.to(device)

                optimizer.zero_grad()
                preds = model(**data)
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
                        data["field_data"] = model(**data)

                    y_hat = data["field_data"]
                    val_loss = criterion(y_hat, push_forward_y.to(device))
                    val_losses.append(val_loss.item())

            val_loss = sum(val_losses) / len(val_losses)
            if tracker:
                tracker.track(val_loss, name="loss/val", step=epoch, context="train")
            logger.info("[%d] val loss: %f", epoch, val_loss)

            scheduler.step()

            if epoch % 10 == 0 and ctx.is_main_process:
                from ray import train

                metrics = {"loss": train_loss, "val_loss": val_loss}
                with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
                    from hspn.train_utils import save_checkpoint

                    save_checkpoint(
                        Path(temp_checkpoint_dir, "model.pt"),
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        gradscaler=None,
                        epoch=epoch,
                        metrics={"epoch": epoch},
                    )
                    checkpoint = Checkpoint.from_directory(temp_checkpoint_dir)
                    train.report(metrics, checkpoint=checkpoint)
    except:
        logger.exception("Failed")
        raise
    finally:
        if tracker:
            tracker.close()


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
