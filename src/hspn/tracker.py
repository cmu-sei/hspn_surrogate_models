import logging
from pathlib import Path
from typing import Dict, Literal, Optional, Union


class Tracker:
    def __init__(
        self,
        log_dir: Union[str, Path],
        backend: Literal["tensorboard", "aim"] = "tensorboard",
        experiment_name: Optional[str] = None,
    ):
        """Initialize metrics tracker with specified backend.

        Args:
            log_dir: Directory to store logs
            backend: Either "tensorboard" or "aim"
            experiment_name: Name of the experiment if using aim (optional)
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.backend = backend
        self.experiment_name = experiment_name or "default"
        self.logger = logging.getLogger(__name__)
        assert backend in ("tensorboard", "aim"), f"Unsupported backend: {backend}"

        if backend == "tensorboard":
            try:
                from torch.utils.tensorboard.writer import SummaryWriter

                self.writer = SummaryWriter(log_dir=str(self.log_dir))
                self.logger.info(f"TensorBoard logging initialized at {self.log_dir}")
            except ImportError:
                self.logger.error(
                    "Could not import SummaryWriter. Install tensorboard."
                )
                raise
        elif backend == "aim":
            try:
                import aim

                self.writer = aim.Run(
                    experiment=self.experiment_name, repo=str(self.log_dir)
                )
                self.logger.info(f"Aim logging initialized at {self.log_dir}")
            except ImportError:
                self.logger.error("Could not import aim. Install aim.")
                raise

    def log_scalar(self, tag: str, value: float, step: int) -> None:
        """Log scalar values."""
        if self.backend == "tensorboard":
            self.writer.add_scalar(tag, value, step)
        elif self.backend == "aim":
            self.writer.track(value, name=tag, step=step)

    def log_scalars(
        self, main_tag: str, tag_scalar_dict: Dict[str, float], step: int
    ) -> None:
        """Log multiple scalars under the same main tag."""
        if self.backend == "tensorboard":
            self.writer.add_scalars(main_tag, tag_scalar_dict, step)
        elif self.backend == "aim":
            for tag, value in tag_scalar_dict.items():
                self.writer.track(value, name=f"{main_tag}/{tag}", step=step)

    def log_histogram(self, tag: str, values, step: int) -> None:
        """Log histogram of values."""
        if self.backend == "tensorboard":
            self.writer.add_histogram(tag, values, step)
        elif self.backend == "aim":
            # do our best to make a histogram
            import numpy as np

            self.writer.track(
                np.histogram(
                    values.detach().cpu().numpy()
                    if hasattr(values, "detach")
                    else values
                ),
                name=f"{tag}_hist",
                step=step,
            )

    def close(self) -> None:
        """Close the logger."""
        if self.backend == "tensorboard":
            self.writer.close()
