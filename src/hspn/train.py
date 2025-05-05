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
from torch.nn import MSELoss
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TaskProgressColumn,
    TimeRemainingColumn,
)


from hspn.dataset import H5Dataset
from hspn.tracker import Tracker
from hspn.train_utils import (
    NullProgress,
    load_checkpoint,
    save_checkpoint,
    setup_distributed,
)

logger = logging.getLogger(__name__)


@dataclass
class TrainConfig:
    seed: int
    n_epochs: int
    checkpoint_dir: Path
    model: nn.Module
    dataloader: DataLoader
    val_dataloader: DataLoader
    optimizer_factory: Callable[..., Optimizer]
    scheduler_factory: Callable[..., LRScheduler]
    comm_backend: Literal["nccl", "gloo"]
    log_interval: int
    tracker_config: Optional[Dict[str, Any]]
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


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> float:
    numer_acc = torch.zeros(1, device=device)
    denom_accu = torch.zeros(1, device=device)
    diff_buf = None

    for branch_in, trunk_in, output in dataloader:
        branch_in = branch_in.to(device, non_blocking=True)
        trunk_in = trunk_in.to(device, non_blocking=True)
        output = output.to(device, non_blocking=True)

        pred = model(branch_in, trunk_in)

        if diff_buf is None or diff_buf.shape != pred.shape:
            diff_buf = torch.empty_like(pred)

        torch.sub(pred, output, out=diff_buf)
        numer_acc.add_(torch.dot(diff_buf.flatten(), diff_buf.flatten()))
        denom_accu.add_(torch.dot(output.flatten(), output.flatten()))

    return (
        (numer_acc / denom_accu.clamp_min(1e-12)).item()
        if numer_acc > 0
        else float("inf")
    )


@hydra.main(config_path="pkg://hspn.conf", config_name="train", version_base=None)
def main(cfg: DictConfig) -> float:
    OmegaConf.resolve(cfg)
    logger.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")
    torch.manual_seed(cfg.seed)
    rank, world_size = setup_distributed()
    best_val_loss = float("inf")
    best_epoch = 0
    epoch = 0
    start_time = time.time()
    tracker = None
    try:
        # Make sure we instantiate after setting up distributed since some components are distributed-aware
        config: TrainConfig = TrainConfig(**hydra.utils.instantiate(cfg))
        config.validate()
        logger.info(
            f"TrainConfig:\n{pprint.pformat(config, width=120, indent=2, sort_dicts=False)}"
        )
        device = torch.device(rank)
        logger.info(f"Using {device}")
        model = config.model.train().to(device)
        if world_size > 1:
            model = nn.parallel.DistributedDataParallel(
                model, device_ids=[rank], output_device=rank
            )
        dataloader = config.dataloader
        optimizer = config.optimizer_factory(model.parameters())
        if config.scheduler_factory:
            scheduler = config.scheduler_factory(optimizer)
        else:
            scheduler = None

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
            best_epoch = epoch = ckpt.get("epoch", 0)
            global_step = ckpt.get("global_step", 0)
            best_val_loss = ckpt.get("best_val_loss", best_val_loss)
            if epoch >= config.n_epochs:
                logger.warning(
                    f"Loaded checkpoint has epoch={ckpt['epoch']} >= {config.n_epochs=}"
                )
            logger.info(
                f"After loading checkpoint epoch=best_epoch={best_epoch} {global_step=} {best_val_loss=}"
            )

        cfg_dict = OmegaConf.to_container(cfg)
        assert isinstance(cfg_dict, dict)

        if rank == 0:
            if config.tracker_config:
                tracker = Tracker(**config.tracker_config)
                tracker.log_hparams(cfg_dict)
            progress_bar = Progress(
                "[progress.description]{task.description}",
                MofNCompleteColumn(),
                BarColumn(bar_width=None),
                TaskProgressColumn(show_speed=True),
                TimeRemainingColumn(),
            )
        else:
            progress_bar = NullProgress()
        with progress_bar:
            for epoch in range(epoch, config.n_epochs):
                epoch_total_loss = 0.0
                epoch_batches = 0
                epoch_start_time = time.time()

                task = progress_bar.add_task(
                    f"Train Epoch {epoch}", total=len(dataloader)
                )
                for (
                    i,
                    (branch_in, trunk_in, output),
                ) in enumerate(dataloader):
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
                    global_step += 1

                    loss_val = loss.item()
                    epoch_total_loss += loss_val
                    epoch_batches += 1
                    progress_bar.update(task, advance=1)

                    if i > 0 and i % config.log_interval == 0:
                        logger.info(
                            f"Epoch {epoch} [batch {i}/{len(dataloader)}] "
                            f"Loss: {loss_val:.6f}, "
                            f"Epoch Time Elapsed: {time.time() - epoch_start_time:.3f}s"
                        )

                avg_loss = epoch_total_loss / epoch_batches
                if rank == 0 and tracker:
                    tracker.log_scalar("train/epoch", epoch, global_step)
                    tracker.log_scalar("train/loss", epoch_total_loss, global_step)
                    tracker.log_scalar("train/avg_loss", avg_loss, global_step)

                    current_lr = (
                        scheduler.get_last_lr()[0]
                        if scheduler
                        else optimizer.param_groups[0]["lr"]
                    )
                    tracker.log_scalar("train/learning_rate", current_lr, global_step)

                logger.info(
                    f"Train Epoch: {epoch} completed in {time.time() - epoch_start_time:.3f}s Batches: {epoch_batches}, Avg Batch Total Loss: {avg_loss:.6f}"
                )

                progress_bar.remove_task(task)

                _ = model.eval()
                val_loss = evaluate(model, config.val_dataloader, device)
                _ = model.train()
                logger.info(f"Validation Loss: {val_loss:.6f}")
                if rank == 0 and tracker:
                    tracker.log_scalar("val/loss", val_loss, global_step)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_epoch = epoch
                    logger.info(f"New Best Epoch: {epoch}")
                    save_checkpoint(
                        config.checkpoint_dir / "best_model.pt",
                        model.module if hasattr(model, "module") else model,
                        optimizer,
                        scheduler,
                        epoch,
                        metrics={
                            "val_loss": best_val_loss,
                            "epoch": epoch,
                            "global_step": global_step,
                        },
                        extra=cfg_dict,
                    )

    except:
        logger.exception("Failed")
        raise
    finally:
        if rank == 0 and tracker:
            tracker.close()
            logger.info(
                f"{epoch + 1}/{cfg.n_epochs} train epochs completed in {time.time() - start_time:.3f}s"
            )
            logger.info(f"    Best Epoch: {best_epoch}")
            logger.info(f"    Best Val Loss: {best_val_loss:.6f}")
        if dist.is_initialized():
            dist.destroy_process_group()

    return best_val_loss


if __name__ == "__main__":
    main()
