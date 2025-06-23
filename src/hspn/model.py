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

"""Deep Operator Network implementation."""

import logging
from typing import List, Tuple, TypedDict

import torch
import torch.nn as nn
from torch._C import device

logger = logging.getLogger(__name__)


class NetworkSpec(TypedDict):
    """Subnet specification."""

    width: int
    depth: int
    activation: nn.Module


class DeepOperatorNet(nn.Module):
    """DON model implementation.

    This model consists of:
    1. A branch network that processes branch/parameter inputs
    2. A trunk network that processes trunk/spatial inputs
    3. An output layer that combines branch and trunk features using einsum
    """

    def __init__(
        self,
        branch_dim: int,
        trunk_dim: int,
        branch_config: NetworkSpec,
        trunk_config: NetworkSpec,
        latent_dim: int = 20,
        einsum_pattern: str = "ij,kj->ik",
        adapter_layer: Tuple[int, int] | None = None,
    ):
        """
        Args:
            branch_dim: Dimension of branch input features
            trunk_dim: Dimension of trunk input features
            branch_config: Network structure for branch network
            trunk_config: Network structure for trunk network
            latent_dim: Dimension of the latent space where branch and trunk meet
            einsum_pattern: Pattern for combining branch and trunk to produce final prediction.
        """
        super().__init__()

        self.branch_dim = branch_dim
        self.trunk_dim = trunk_dim
        self.latent_dim = latent_dim
        self.einsum_pattern = einsum_pattern

        self.branch_net = self._build_net(branch_config, branch_dim)
        if adapter_layer:
            self.branch_net.insert(0, nn.Linear(adapter_layer[0], adapter_layer[1]))
        self.trunk_net = self._build_net(trunk_config, trunk_dim)
        self._init_weights()

        logger.info(f"DeepONet model created with latent_dim={self.latent_dim}")
        logger.info(f"Model structure:\n{self}")

    def _build_net(self, net_config: NetworkSpec, input_dim: int) -> nn.Sequential:
        """Build a network based on width and depth configuration.

        Args:
            net_config: Network configuration with width, depth, and activation
            input_dim: Input dimension

        Returns:
            Sequential network module
        """
        layers: List[nn.Module] = []
        prev_width = input_dim
        for _ in range(net_config["depth"] - 1):  # We add a final layer
            layers.append(nn.Linear(prev_width, net_config["width"]))
            layers.append(net_config["activation"])
            prev_width = net_config["width"]

        # Final layer to map back to latent dimension
        layers.append(nn.Linear(prev_width, self.latent_dim))

        return nn.Sequential(*layers)

    def _init_weights(self) -> None:
        """Initialize network weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def curr_device(self) -> device:
        return next(self.parameters()).device

    def forward(self, branch_input: torch.Tensor, trunk_input: torch.Tensor) -> torch.Tensor:
        """Forward pass. Inputs must be on the correct device.

        Args:
            branch_input: Tensor of shape (branch_batch_size, branch_dim)
            trunk_input: Tensor of shape (trunk_batch_size, trunk_dim)

        Returns:
            Tensor of shape (branch_batch_size, trunk_batch_size)
        """

        # logger.info(f"{trunk_input.shape=}")
        # logger.info(f"{branch_input.shape=}")
        branch_output = self.branch_net(branch_input)  # (branch_bs, branch_dim) -> (branch_bs, latent_dim)
        trunk_output = self.trunk_net(trunk_input)  # [trunk_bs, trunk_dim] -> (trunk_bs, latent_dim)
        # logger.info(f"{branch_output.shape=}")
        # logger.info(f"{trunk_output.shape=}")
        return torch.einsum(
            self.einsum_pattern,  # e.g. "ij, kj-> ik"
            branch_output,
            trunk_output,
        )  # (branch_bs, trunk_bs)

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Forward pass and compute loss.

        This method doesn't really "train" anything, we adopt this convention from torch lightning.

        Args:
            batch: Branch input, trunk input, output
                Branch shape is (branch_batch_size, branch_dim)
                Trunk shape is (trunk_batch_size, trunk_dim)
                Output shape is (branch_batch_size, trunk_batch_size)
            batch_idx: Tensor of shape (branch_batch_size, branch_dim)

        Returns:
            Computed loss
        """
        # logger.info(f"{batch[-1].shape=}")
        del batch_idx
        preds = self.forward(batch[0], batch[1])
        loss = torch.nn.functional.mse_loss(preds, batch[2], reduction="mean")
        return loss

    def approx_size(self, dtype_size_bytes: int = 4) -> Tuple[float, float, float]:
        """Estimate mode size in GB.

        Args:
            dtype_size_bytes: Number of bytes per parameter element (default 4 for float32).
                            Use 2 for float16, 1 for int8, etc.

        Returns:
            Approximate model size in gibibytes for branch, trunk, and total.
        """

        def estimate(model: nn.Module):
            total_params = sum(p.numel() for p in model.parameters())
            total_buffers = sum(b.numel() for b in model.buffers())
            total_elements = total_params + total_buffers
            total_bytes = total_elements * dtype_size_bytes
            return total_bytes / (1024**3)

        b = estimate(self.branch_net)
        t = estimate(self.trunk_net)
        return (b, t, b + t)
