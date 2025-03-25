import logging
import time
from typing import Any, TypeVar, Optional

import hydra
import torch
import torch.distributed as dist
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig
from dataclasses import dataclass
from torch import nn, optim

from hspn.train_utils import setup_distributed
from hspn.train_utils import save_checkpoint, setup_distributed

logger = logging.getLogger(__name__)

Optimizer = TypeVar("Optimizer", bound=optim.Optimizer)
LRScheduler = TypeVar("LRScheduler", bound=optim.lr_scheduler.LRScheduler)


@dataclass
class TrainConfig:
    seed: int
    n_epochs: int
    # model: DeepOperatorNet
    model: Any
    dataloader: Any  # torch.utils.data.DataLoader
    optimizer_factory: Any
    scheduler_factory: Optional[Any]
    # optimizer_factory: Callable[[...], Optimizer]
    # scheduler_factory: Callable[[...], LRScheduler]
    comm_backend: Any  # Literal["nccl", "gloo"]
    log_interval: int
    extra: Any
    _target_: str = "hspn.train.TrainConfig"

    def validate(self):
        """Validate after config is instantiated."""
        if self.dataloader.batch_size and self.dataloader.batch_size != 1:
            raise ValueError(
                f"Found an invalid value for {self.dataloader.batch_size=} Batching is currently handled by the dataset. "
                "Please apply batch settings to the dataset."
            )


ConfigStore.instance().store("train_spec", TrainConfig)


@hydra.main(config_path="pkg://hspn.conf", config_name="train", version_base=None)
def main(cfg: DictConfig) -> float:
    torch.manual_seed(cfg.seed)
    rank, world_size = setup_distributed()
    best_epoch_total_loss = float("inf")
    best_epoch = 0
    total_batches = 0
    total_loss = 0
    epoch = 0
    start_time = time.time()
    try:
        # Make sure we instantiate after setting up distributed since some components are distributed-aware
        config: TrainConfig = hydra.utils.instantiate(cfg)
        config.validate()
        model = config.model.train()
        logger.info(str(model))
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

        for epoch in range(config.n_epochs):
            epoch_total_loss = 0.0
            epoch_batches = 0
            epoch_start_time = time.time()
            epoch_elapsed = 0
            for (
                i,
                (branch_in, trunk_in, output),
            ) in enumerate(
                dataloader
            ):  # do we have to set the dataloader sampler epoch? We just have an iterable dataset.
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

                if torch.isnan(loss):
                    print(f"NaN detected at batch {i}")
                    print(torch.isnan(branch_in).any(), branch_in.shape)
                    print(torch.isnan(trunk_in).any(), trunk_in.shape)
                    print(torch.isnan(output).any(), output.shape)
                    continue

                optimizer.zero_grad()
                loss.backward()
                # TODO: Might want to consider grad clip control later, leaving note to remember.
                optimizer.step()
                if scheduler:
                    scheduler.step()

                epoch_total_loss += loss.item()
                epoch_batches += 1
                epoch_elapsed = time.time() - epoch_start_time

                if i > 0 and i % config.log_interval == 0:
                    logger.info(
                        f"Epoch {epoch} [{i}/{len(dataloader)}] "
                        f"Loss: {loss.item():.6f}, "
                        f"Epoch Time Elapsed: {epoch_elapsed:.3f}s"
                    )
            if epoch_total_loss < best_epoch_total_loss:
                best_epoch_total_loss = epoch_total_loss
                best_epoch = epoch
                logger.info(f"New Best Epoch: {epoch}")
                cfg_dict = OmegaConf.to_container(cfg)
                assert isinstance(cfg_dict, dict)
                save_checkpoint(
                    config.checkpoint_dir,
                    model.module if hasattr(model, "module") else model,
                    optimizer,
                    scheduler,
                    epoch,
                    metrics={
                        "loss": best_epoch_total_loss,
                        "epoch_time_elapsed": epoch_elapsed,
                    },
                    extra=cfg_dict,
                )

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
