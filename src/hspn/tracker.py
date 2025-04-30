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
#

import logging
from pathlib import Path
from typing import Any, Dict, Literal, Optional, Union


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

    def log_hparams(self, hparams: Dict[str, Any]) -> None:
        if self.backend == "tensorboard":
            flattened = self._flatten_dict(hparams)
            self.writer.add_hparams(flattened, {})  # type: ignore
        elif self.backend == "aim":
            self.writer["hparams"] = hparams  # type: ignore

    def _flatten_dict(
        self, d: Dict[str, Any], parent_key: str = "", sep: str = "/"
    ) -> Dict[str, Any]:
        """Flatten a nested dictionary for TensorBoard hparams."""
        items = []
        for k, v in d.items():
            new_key = parent_key + sep + k if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append(
                    (new_key, v if isinstance(v, (bool, int, float, str)) else str(v))
                )
        return dict(items)

    def log_scalar(self, tag: str, value: float, step: int) -> None:
        """Log scalar values."""
        if self.backend == "tensorboard":
            self.writer.add_scalar(tag, value, step)  # type: ignore
        elif self.backend == "aim":
            self.writer.track(value, name=tag, step=step)  # type: ignore

    def log_scalars(
        self, main_tag: str, tag_scalar_dict: Dict[str, float], step: int
    ) -> None:
        """Log multiple scalars under the same main tag."""
        if self.backend == "tensorboard":
            self.writer.add_scalars(main_tag, tag_scalar_dict, step)  # type: ignore
        elif self.backend == "aim":
            for tag, value in tag_scalar_dict.items():
                self.writer.track(value, name=f"{main_tag}/{tag}", step=step)  # type: ignore

    def log_histogram(self, tag: str, values, step: int) -> None:
        """Log histogram of values."""
        if self.backend == "tensorboard":
            self.writer.add_histogram(tag, values, step)  # type: ignore
        elif self.backend == "aim":
            self.writer.track(  # type: ignore
                values.detach().cpu().numpy() if hasattr(values, "detach") else values,
                name=f"{tag}_hist",
                step=step,
            )

    def close(self) -> None:
        """Close the logger."""
        self.writer.close()
