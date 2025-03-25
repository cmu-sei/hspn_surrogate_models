import logging
import pprint
from pathlib import Path
import time
from typing import Any, Callable, Dict, Literal, Optional
import hydra
from omegaconf.omegaconf import OmegaConf
import torch
import torch.distributed as dist
from omegaconf import DictConfig
from dataclasses import dataclass
from torch import nn
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
from tqdm.contrib.logging import tqdm_logging_redirect

from hspn.dataset import H5Dataset
from hspn.tracker import Tracker
from hspn.train_utils import load_checkpoint, save_checkpoint, setup_distributed

logger = logging.getLogger(__name__)


@dataclass
class TrainConfig:
    seed: int
    n_epochs: int
    checkpoint_dir: Path
    model: nn.Module
    dataloader: DataLoader
    optimizer_factory: Callable[..., Optimizer]
    scheduler_factory: Callable[..., LRScheduler]
    comm_backend: Literal["nccl", "gloo"]
    log_interval: int
    tracker_config: Dict[str, Any]
    extra: Optional[Any] = None

    def __post_init__(self):
        self.checkpoint_dir = Path(self.checkpoint_dir)

    def validate(self):
        """Validate after config is instantiated."""
        if (
            isinstance(getattr(self.dataloader, "dataset"), H5Dataset)
            and self.dataloader.batch_size
            and self.dataloader.batch_size != 1
        ):
            raise ValueError(
                f"Found an invalid value for {self.dataloader.batch_size=} Batching is currently handled by {H5Dataset!s}"
                "Please apply batch settings to the dataset."
            )


@hydra.main(config_path="pkg://hspn.conf", config_name="train", version_base=None)
def main(cfg: DictConfig) -> float:
    OmegaConf.resolve(cfg)
    logger.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")
    torch.manual_seed(cfg.seed)
    rank, world_size = setup_distributed()
    best_epoch_total_loss = float("inf")
    best_epoch = 0
    total_batches = 0
    total_loss = 0
    epoch = 0
    start_time = time.time()
    tracker = None
    try:
        # Make sure we instantiate after setting up distributed since some components are distributed-aware
        config: TrainConfig = TrainConfig(**hydra.utils.instantiate(cfg))
        config.validate()
        model = config.model.train()
        logger.info(
            f"TrainConfig:\n{pprint.pformat(config, width=120, indent=2, sort_dicts=False)}"
        )
        if world_size > 1:
            nn.parallel.DistributedDataParallel(config.model, device_ids=[rank])
        dataloader = config.dataloader
        optimizer = config.optimizer_factory(model.parameters())
        if config.scheduler_factory:
            scheduler = config.scheduler_factory(optimizer)
        else:
            scheduler = None

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using {device}")
        model.to(device)

        epoch = 0
        global_step = 0
        if config.checkpoint_dir.exists() and (
            ckpts := list(config.checkpoint_dir.glob("checkpoint_*.pt"))
        ):
            latest = sorted(ckpts)[-1]
            ckpt = load_checkpoint(
                latest,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                map_location=device,
            )
            epoch = ckpt["epoch"]
            global_step = ckpt.get("global_step", 0)
            if epoch >= config.n_epochs:
                logger.warning(
                    f"Loaded checkpoint has epoch={ckpt['epoch']} >= {config.n_epochs=}"
                )

        tracker = Tracker(**config.tracker_config) if rank == 0 else None

        for epoch in range(epoch, config.n_epochs):
            epoch_total_loss = 0.0
            epoch_batches = 0
            epoch_start_time = time.time()
            epoch_elapsed = 0
            with tqdm_logging_redirect(
                dataloader,
                desc=f"Epoch {epoch}",
                unit="batch",
                disable=(rank != 0),
            ) as progress_bar:
                for (
                    i,
                    (branch_in, trunk_in, output),
                ) in enumerate(progress_bar):
                    model_device = model.curr_device()
                    loss = model.training_step(
                        (
                            branch_in.to(model_device),
                            trunk_in.to(model_device),
                            output.to(model_device),
                        ),
                        i,
                    )

                    optimizer.zero_grad()
                    loss.backward()
                    # TODO: Might want to consider grad clip control later, leaving note to remember.
                    optimizer.step()
                    if scheduler:
                        scheduler.step()
                    loss_val = loss.item()
                    epoch_total_loss += loss_val
                    epoch_batches += 1
                    global_step += 1
                    epoch_elapsed = time.time() - epoch_start_time
                    avg_loss = epoch_total_loss / epoch_batches
                    progress_bar.set_postfix(loss=loss_val, avg=avg_loss)

                    if i > 0 and i % config.log_interval == 0:
                        logger.info(
                            f"Epoch {epoch} [{i}/{len(dataloader)}] "
                            f"Loss: {loss_val:.6f}, "
                            f"Epoch Time Elapsed: {epoch_elapsed:.3f}s"
                        )

                if epoch_total_loss < best_epoch_total_loss:
                    best_epoch_total_loss = epoch_total_loss
                    best_epoch = epoch
                    logger.info(f"New Best Epoch: {epoch}")
                    cfg_dict = OmegaConf.to_container(cfg)
                    assert isinstance(cfg_dict, dict)
                    save_checkpoint(
                        config.checkpoint_dir / f"checkpoint_{epoch}.pt",
                        model.module if hasattr(model, "module") else model,
                        optimizer,
                        scheduler,
                        epoch,
                        metrics={
                            "loss": best_epoch_total_loss,
                            "epoch_time_elapsed": epoch_elapsed,
                            "global_step": global_step,
                        },
                        extra=cfg_dict,
                    )

                if rank == 0 and tracker:
                    avg_loss = epoch_total_loss / epoch_batches
                    print("logging", epoch, global_step)
                    tracker.log_scalar("train/epoch", epoch, global_step)
                    tracker.log_scalar("train/loss", epoch_total_loss, global_step)
                    tracker.log_scalar("train/avg_loss", avg_loss, global_step)

                    current_lr = (
                        scheduler.get_last_lr()[0]
                        if scheduler
                        else optimizer.param_groups[0]["lr"]
                    )
                    tracker.log_scalar("train/learning_rate", current_lr, global_step)

                logger.info(f"{epoch_total_loss=} {epoch_batches=}")
                total_batches += epoch_batches
                total_loss += epoch_total_loss
                avg_loss = epoch_total_loss / epoch_batches
                logger.info(
                    f"Train Epoch: {epoch} completed in {time.time() - epoch_start_time:.3f}s Batches: {epoch_batches}, Avg Batch Total Loss: {avg_loss:.6f}"
                )
    except:
        logger.exception("Failed")
        raise
    finally:
        # We always clean up
        if rank == 0 and tracker:
            tracker.close()
        if dist.is_initialized():
            dist.destroy_process_group()
        logger.info(
            f"{epoch}/{cfg.n_epochs} train epochs completed in {time.time() - start_time:.3f}s"
        )
        logger.info(f"    Batch Avg Loss: {total_loss/total_batches}")
        logger.info(f"    N batches: {total_batches}")
        logger.info(f"    Best Epoch: {best_epoch}")
        logger.info(f"    Best Epoch Total Loss: {best_epoch_total_loss:.6f}")

    return best_epoch_total_loss


if __name__ == "__main__":
    main()
