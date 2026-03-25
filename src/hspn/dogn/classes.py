"""
Graph Neural Network for physics simulation with message passing.

Supports DeepONet conditioning, stream function formulation, and various
time aggregation strategies.

- classes:
    - ReZeroScale
    - GraphNetworkSimulator
- functions:
    - get_model
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

if TYPE_CHECKING:
    from typing import Literal

# =============================================================================
# Constants
# =============================================================================

EPSILON = 1e-10  # For numerical stability in divisions
EDGE_THRESHOLD = 1e-6  # For detecting axis-aligned edges in stream function

# Activation Map
activation_map = {
    'relu': torch.nn.ReLU(),
    'silu': torch.nn.SiLU(),
    'gelu': torch.nn.GELU(),
    'tanh': torch.nn.Tanh(),
    'sigmoid': torch.nn.Sigmoid(),
    'elu': torch.nn.ELU(),
    'leaky_relu': torch.nn.LeakyReLU(),
}

def get_activation(name):
    return activation_map[name.lower()]

# =============================================================================
# ReZero Module
# =============================================================================

class ReZeroScale(nn.Module):
    """
    Learnable residual scaling for stable deep network training.
    
    Uses softplus to ensure positive scaling. Default init_val=-5 gives
    softplus(-5) ≈ 0.007, starting near zero per the ReZero paper.
    
    Reference: https://arxiv.org/abs/2003.04887
    """
    
    def __init__(self, init_val: float = 0.001):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor([init_val]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        #return F.softplus(self.scale) * x
        return torch.abs(self.scale) * x

# =============================================================================
# Main Model
# =============================================================================

class GraphNetworkSimulator(nn.Module):
    """
    Message-passing graph neural network for learning physical simulations.
    
    Architecture:
        1. Encode nodes and edges to latent space
        2. N message-passing steps (edge update → aggregate → node update)
        3. Decode node features to output fields
    
    Supports:
        - DeepONet conditioning via branch network
        - Temporal convolution for bundle processing
        - Edge attention and dropout for regularization
    """
    
    def __init__(
        self,
        edge_encoder: nn.Module,
        node_encoder: nn.Module,
        edge_updaters: nn.ModuleList,
        node_updaters: nn.ModuleList,
        node_decoder: nn.Module,
        branch_net: nn.Module | None = None,
        edge_attention: nn.Module | None = None,
        bundle_size: int = 1,
        node_encoder_input_size: int = -1,
#        num_edge_features: int = -1,
        deeponet_type: Literal["off", "node", "edge", "both", "compatibility", "compat"] = "off",
        time_aggregation: Literal[ "off", "flatten", "conv", "conv_compress"] = "flatten",
        time_integration: Literal["off", "linear", "cumsum"] = "off",
        edge_dropout: float = 0.0,
        sparsity: int | None = None,
        center: bool = False,
    ):
        super().__init__()
        
        # Core network components
        self.edge_encoder = edge_encoder
        self.node_encoder = node_encoder
        self.edge_updaters = edge_updaters
        self.node_updaters = node_updaters
        self.node_decoder = node_decoder
        
        # Optional components
        self.branch_net = branch_net
        self.edge_attention = edge_attention
        self.time_encoder: nn.Conv1d | None = None
        self.time_decoder: nn.ConvTranspose1d | None = None

        # Configuration
        self.bundle_size = bundle_size
        self.node_encoder_input_size = node_encoder_input_size
#        self.num_edge_features = num_edge_features
        self.n_processors = len(edge_updaters)

        # DeepONet flags (require branch_net to be present)
        self.node_deeponet = deeponet_type in ("both", "node", "compat", "compatibility") and branch_net is not None
        self.edge_deeponet = deeponet_type in ("both", "edge") and branch_net is not None
        self.is_compatibility_mode = deeponet_type[:6].lower() == "compat"
        
        #if self.is_compatibility_mode:
        self.no_affine_layer_norm = nn.LayerNorm(self.node_encoder[-1].weight.size(0), elementwise_affine=False)

        # Colin mode overrides certain settings
        if self.is_compatibility_mode:
            edge_dropout = 0.0
            sparsity = None
            center = False

        self.edge_dropout = edge_dropout
        self.sparsity = sparsity
        self.center = center
        
        # Time handling
        self.time_aggregation = time_aggregation
        self.time_integration = time_integration
        self._init_time_convolutions(bundle_size)

        self._init_weights()
        
    def _init_time_convolutions(self, bundle_size: int) -> None:
        """Initialize temporal convolution layers if needed."""
        if not self.time_aggregation.startswith("conv") or bundle_size <= 1:
            if self.time_aggregation.startswith("conv"):
                self.time_aggregation = "off"
            return

        hidden_dim = self.node_encoder[-1].weight.shape[0]
        padding = (bundle_size - 1) // 2 if self.time_aggregation == "conv" else 0
        
        self.time_encoder = nn.Conv1d(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            kernel_size=bundle_size,
            stride=1,
            padding=padding,
        )
        self.time_decoder = nn.ConvTranspose1d(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            kernel_size=bundle_size,
            stride=1,
            padding=padding,
        )

    def _init_weights(self) -> None:
        """Apply Kaiming initialization to all linear layers."""
        self.apply(self._kaiming_init)
        if not hasattr(self,"branch_net") or self.branch_net is None:
            return 
        
        # branch net is initialized to give near-1 values
        last_linear_layer = None
        for module in self.branch_net.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.02, std=0.025)
                if module.bias is not None:
                    torch.nn.init.constant_(module.bias, 1.0)
                last_linear_layer = module
        if isinstance(last_linear_layer, nn.Linear) and  last_linear_layer.bias is None:
            torch.nn.init.normal_(last_linear_layer.weight, mean=1/last_linear_layer.in_features, std=0.05)

    @staticmethod
    def _kaiming_init(module: nn.Module) -> None:
        """Kaiming normal initialization with zero bias."""
        if isinstance(module, nn.Linear):
            # https://arxiv.org/html/2406.02269v2
            #nn.init.normal_(module.weight, mean=0, std=2) 
            #nn.init.kaiming_normal_(module.weight, mode="fan_in", nonlinearity="relu")
            nn.init.orthogonal_(module.weight, gain =1.414) 
            if module.bias is not None:
                nn.init.normal_(module.bias, mean=0, std=0.001) 
                #module.bias.data.zero_()

    # -------------------------------------------------------------------------
    # Edge Dropout
    # -------------------------------------------------------------------------

    @staticmethod
    def compute_edge_mask(
        edge_index: torch.Tensor,
        dropout_prob: float = 0.0,
        training: bool = False,
        sparsity: int | None = None,
    ) -> torch.Tensor:
        """
        Create edge dropout mask guaranteeing each node keeps at least one incoming edge.

        Args:
            edge_index: [2, num_edges] source and destination node indices
            dropout_prob: Probability of dropping each edge (ignored if sparsity set)
            training: Whether in training mode
            sparsity: If set, keep at most this many incoming edges per node

        Returns:
            Boolean mask of shape [num_edges] indicating edges to keep
        """
        num_edges = edge_index.size(1)
        device = edge_index.device

        if not training or dropout_prob == 0:
            return torch.ones(num_edges, dtype=torch.bool, device=device)

        # Assign random priorities with tie-breaking
        priorities = torch.rand(num_edges, device=device)
        priorities += torch.arange(num_edges, device=device) * 1e-10

        if sparsity is not None and sparsity >= 1:
            return GraphNetworkSimulator._sparsity_based_mask(
                edge_index, priorities, sparsity
            )
        else:
            return GraphNetworkSimulator._dropout_based_mask(
                edge_index, priorities, dropout_prob
            )

    @staticmethod
    def _sparsity_based_mask(
        edge_index: torch.Tensor,
        priorities: torch.Tensor,
        sparsity: int,
    ) -> torch.Tensor:
        """Keep top-k incoming edges per node based on priority."""
        device = edge_index.device
        num_edges = edge_index.size(1)
        dst_nodes = edge_index[1]
        num_nodes = dst_nodes.max() + 1

        keep_mask = torch.zeros(num_edges, dtype=torch.bool, device=device)
        remaining_priority = priorities.clone()

        for _ in range(sparsity):
            # Find highest priority edge for each destination node
            max_priority = torch.full((num_nodes,), float("-inf"), device=device)
            max_priority = max_priority.scatter_reduce(
                0, dst_nodes, remaining_priority, reduce="amax"
            )
            
            # Mark those edges and remove from consideration
            is_max = remaining_priority == max_priority[dst_nodes]
            keep_mask |= is_max
            remaining_priority[is_max] = float("-inf")

        return keep_mask

    @staticmethod
    def _dropout_based_mask(
        edge_index: torch.Tensor,
        priorities: torch.Tensor,
        dropout_prob: float,
    ) -> torch.Tensor:
        """Drop edges randomly but guarantee at least one per node."""
        device = edge_index.device
        num_edges = edge_index.size(1)
        num_nodes = edge_index.max() + 1

        # Find highest priority edge per destination (guaranteed kept)
        max_priority = torch.full((num_nodes,), -1.0, device=device)
        max_priority = max_priority.scatter_reduce(
            0, edge_index[1], priorities, reduce="amax"
        )
        guaranteed = priorities == max_priority[edge_index[1]]

        # Keep guaranteed edges plus random sample of others
        random_keep = torch.rand(num_edges, device=device) > dropout_prob
        return guaranteed | (~guaranteed & random_keep)

    # -------------------------------------------------------------------------
    # Time Encoding/Decoding
    # -------------------------------------------------------------------------

    def _time_encode(
        self,
        node_features: torch.Tensor,
        batch_size: int,
        bundle_size: int,
        num_nodes: int,
    ) -> tuple[torch.Tensor, int]:
        """Apply temporal convolution to compress time dimension."""
        # Input: [batch*nodes, features, time]
        encoded = self.time_encoder(node_features)
        new_time_size = encoded.shape[2]
        
        # Flatten time into features: [batch*nodes, features * time]
        encoded = encoded.reshape(batch_size * num_nodes, -1)
        return encoded, new_time_size

    def _time_decode(
        self,
        node_features: torch.Tensor,
        batch_size: int,
        time_size: int,
        num_nodes: int,
    ) -> tuple[torch.Tensor, int]:
        """Apply transpose convolution to expand time dimension."""
        features_per_time = node_features.shape[1] // time_size
        
        # Reshape to [batch*nodes, features, time]
        x = node_features.view(batch_size * num_nodes, features_per_time, time_size)
        decoded = self.time_decoder(x)
        
        return decoded, decoded.shape[2]

    # -------------------------------------------------------------------------
    # Edge Attention
    # -------------------------------------------------------------------------

    def _compute_edge_attention(
        self,
        layer_idx: int,
        edge_enc: torch.Tensor,
    ) -> torch.Tensor:
        """Compute attention weights for edges."""
        if self.edge_attention is None:
            return torch.ones(1, device=edge_enc.device)
        
        # Append layer index as feature
        layer_feature = torch.full_like(edge_enc[:, :1], layer_idx)
        attention_input = torch.cat([edge_enc, layer_feature], dim=-1)
        return torch.sigmoid(self.edge_attention(attention_input))

    # -------------------------------------------------------------------------
    # Forward Methods
    # -------------------------------------------------------------------------

    def _forward_compatibility(
        self,
        field_data: torch.Tensor,
        one_hot_node_type: torch.Tensor,
        edge_index: torch.Tensor,
        edge_features: torch.Tensor,
        params: torch.Tensor,
        num_nodes: int,
        num_edges: int,
    ) -> torch.Tensor:
        """
        Forward pass for Colin-style DeepONet architecture.
        
        Uses only the last timestep and applies linear time integration.
        """
        batch_nodes, n_fields, bundle_size = field_data.shape
        batch_edges = edge_features.shape[0]
        batches = batch_nodes // num_nodes
        edges_per_graph = batch_edges // batches

        # Use only last timestep
        field_data = field_data[:, :, -1]
        edge_index = edge_index[:, :batches * edges_per_graph]

        # Combine fields and node types
        node_data = torch.cat([field_data, one_hot_node_type], dim=-1)

        # Branch network
        branch_out = None
        if self.branch_net is not None:
            branch_out = self.branch_net(params)
            branch_out = branch_out.repeat_interleave(num_nodes, dim=0)

        # Encode
        node_enc = checkpoint(self.node_encoder, node_data, use_reentrant=False)
        edge_enc = checkpoint(self.edge_encoder, edge_features, use_reentrant=False)
        
        # Message passing
        for ex, (edge_updater, node_updater) in enumerate(zip(self.edge_updaters, self.node_updaters)):
            node_enc = self.no_affine_layer_norm(node_enc)
            src_features = node_enc[edge_index[0]]
            dst_features = node_enc[edge_index[1]]

            # Update edges
            edge_input = torch.cat([edge_enc, dst_features, src_features], dim=-1)
            edge_enc = checkpoint(edge_updater, edge_input, use_reentrant=False)

            # Aggregate to nodes (mean)
            aggregated = edge_enc.new_zeros(node_enc.shape)
            aggregated = aggregated.scatter_reduce(
                0,
                edge_index[1].unsqueeze(1).expand_as(edge_enc),
                edge_enc,
                reduce="mean",
            )

            # Update nodes
            node_input = torch.cat([node_enc, aggregated], dim=-1)
            node_delta = node_updater(node_input)
            node_enc = node_enc + node_delta

            # Apply DeepONet modulation
            if branch_out is not None:
                node_enc = node_enc * branch_out
                
        # Decode and apply linear time scaling
        #node_enc = self.no_affine_layer_norm(node_enc)
        output = self.node_decoder(node_enc)
    
        time_scale = torch.arange(1, self.bundle_size + 1, dtype=output.dtype, device=output.device)
        output = output.view(batches, num_nodes, n_fields, self.bundle_size)
        output = torch.einsum("bnft,t->bnft", output, time_scale)
        
        return output.reshape(-1, n_fields, self.bundle_size)

    def _forward_internal(
        self,
        field_data: torch.Tensor,
        one_hot_node_type: torch.Tensor,
        edge_index: torch.Tensor,
        edge_features: torch.Tensor,
        params: torch.Tensor,
        num_nodes: int,
        num_edges: int,
    ) -> torch.Tensor:
        """
        Forward pass for Jasmine-style architecture.
        
        Supports temporal convolution, edge attention, and various DeepONet modes.
        """

        batch_nodes, n_fields, bundle_size = field_data.shape
        batches = batch_nodes // num_nodes

        # Expand node types to match bundle size
        one_hot_expanded = one_hot_node_type.unsqueeze(-1).expand(-1, -1, bundle_size)
        node_data = torch.cat([field_data, one_hot_expanded], dim=1)

        # Branch network (pad to D for passthrough channel)
        branch_out = None
        if self.branch_net is not None:
            branch_out = self.branch_net(params)
            branch_out = F.pad(branch_out, (0, 1), value=1.0)

        # Encode nodes (with optional temporal processing)
        node_enc, time_size = self._encode_nodes(node_data, batches, bundle_size, num_nodes)
        edge_enc = checkpoint(self.edge_encoder, edge_features, use_reentrant=False)
        # Compute edge mask (once per batch, then repeat)
        single_batch_edges = edge_index.reshape(2, batches, num_edges)[:, 0, :]
        edge_mask = self.compute_edge_mask(
            single_batch_edges,
            dropout_prob=self.edge_dropout,
            training=self.training,
            sparsity=self.sparsity,
        )
        edge_mask = edge_mask.repeat(batches)

        # Message passing
        ctx = {
            "edge_index": edge_index,
            "edge_mask": edge_mask,
            "branch_out": branch_out,
            "batches": batches,
            "num_nodes": num_nodes,
            "num_edges": num_edges,
        }
        for layer_idx, (edge_updater, node_updater) in enumerate(zip(self.edge_updaters, self.node_updaters)):
            edge_enc, node_enc = self._message_passing_step(layer_idx, edge_enc, node_enc, edge_updater, node_updater, **ctx)
        
        node_enc = self.no_affine_layer_norm(node_enc) 
        
        # Decode
        output = self._decode_nodes(node_enc, batches, bundle_size, time_size, n_fields, num_nodes)

        # Time integration
        if self.time_integration == "linear":
            n_steps = output.size(-1)
            time_scale = torch.arange(1, n_steps + 1, dtype=output.dtype, device=output.device)
            output = torch.einsum("bft,t->bft", output, time_scale)
        elif self.time_integration == "cumsum":
            output = torch.cumsum(output, dim=2)

        return output

    def _encode_nodes(
        self,
        node_data: torch.Tensor,
        batches: int,
        bundle_size: int,
        num_nodes: int,
    ) -> tuple[torch.Tensor, int | None]:
        """Encode node features with optional temporal processing."""
        batch_nodes = node_data.shape[0]
        if self.time_aggregation == "flatten" or bundle_size == 1:
            node_data = node_data.view(batch_nodes, self.node_encoder_input_size) #batch_nodes, features
            node_enc = checkpoint(self.node_encoder, node_data, use_reentrant=False)
            return node_enc, None
            
        elif self.time_aggregation.startswith("conv"):
            # Process each timestep independently, then apply temporal conv
            node_data = node_data.transpose(1, 2).reshape(batch_nodes * bundle_size, self.node_encoder_input_size)
            node_enc = checkpoint(self.node_encoder, node_data, use_reentrant=False)
            node_enc = node_enc.reshape(batch_nodes, bundle_size, -1).transpose(1, 2) #batch_nodes, features, time
            node_enc, time_size = self._time_encode(node_enc, batches, bundle_size, num_nodes) #batch_nodes, new_features
            return node_enc, time_size
        elif self.time_aggregation == "off":
            
            node_data = node_data.transpose(1, 2).reshape(batch_nodes * bundle_size, self.node_encoder_input_size)
            node_enc = checkpoint(self.node_encoder, node_data, use_reentrant=False)
            return node_enc, None
        else:
            raise NotImplementedError(f"Unknown time_aggregation: {self.time_aggregation}")

    def _decode_nodes(
        self,
        node_enc: torch.Tensor,
        batches: int,
        bundle_size: int,
        time_size: int | None,
        n_fields: int,
        num_nodes: int,
    ) -> torch.Tensor:
        """Decode node features to output fields."""
        batch_nodes = batches * num_nodes
        
        if self.time_aggregation == "flatten":
            output = self.node_decoder(node_enc)
            return output.view(batch_nodes, n_fields, -1)
            
        elif self.time_aggregation.startswith("conv"):
            decoded, _ = self._time_decode(node_enc, batches, time_size, num_nodes)
            features = decoded.shape[1]  # Save before transpose
            decoded = decoded.transpose(1, 2).reshape(batch_nodes * bundle_size, features)
            output = self.node_decoder(decoded)
            return output.reshape(batch_nodes, bundle_size, n_fields).transpose(1, 2)
        
        elif self.time_aggregation == "off":
            output = self.node_decoder(node_enc)
            output = output.reshape(batch_nodes, -1, n_fields).transpose(2,1)
            return output
        else:
            raise NotImplementedError(f"Unknown time aggregation {self.time_aggregation}")
    
    def _message_passing_step(
        self,
        layer_idx: int,
        edge_enc: torch.Tensor,
        node_enc: torch.Tensor,
        edge_updater: nn.Module,
        node_updater: nn.Module,
        edge_index: torch.Tensor,
        edge_mask: torch.Tensor,
        branch_out: torch.Tensor | None,
        batches: int,
        num_nodes: int,
        num_edges: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Single message passing step: edge update → aggregate → node update."""
        
        # Compute edge attention
        attention_weights = self._compute_edge_attention(layer_idx, edge_enc)

        # === Edge Update ===
        masked_edges = edge_mask
        edge_input = torch.cat([
            edge_enc[masked_edges],
            node_enc[edge_index[0, masked_edges]],
            node_enc[edge_index[1, masked_edges]],
        ], dim=-1)
        
        edge_update = checkpoint(edge_updater, edge_input, use_reentrant=False)

        # Apply edge DeepONet modulation
        if self.edge_deeponet and branch_out is not None:
            edge_batch_idx = torch.arange(batches, device=edge_update.device).repeat_interleave(num_edges)[masked_edges]
            edge_update = edge_update * branch_out[edge_batch_idx]

        # Scatter updates back (no residual - full replacement)
        new_edge_enc = torch.zeros(edge_enc.size(0), edge_update.size(1), device=edge_enc.device)
        new_edge_enc[masked_edges] = edge_update
        edge_enc = edge_enc + new_edge_enc
        
        # === Aggregate Edges to Nodes ===
        weighted_edges = edge_enc * attention_weights
        
        aggregated = torch.zeros_like(node_enc)
        aggregated = aggregated.scatter_reduce(
            0,
            edge_index[1, masked_edges].unsqueeze(1).expand(-1, weighted_edges.size(1)),
            weighted_edges[masked_edges],
            reduce="sum",
        )

        # Normalize by attention weight sums if using attention
        if attention_weights.numel() > 1:
            weight_sums = torch.zeros(node_enc.size(0), 1, device=edge_enc.device)
            weight_sums = weight_sums.scatter_reduce(
                0,
                edge_index[1, masked_edges].unsqueeze(1),
                attention_weights[masked_edges],
                reduce="sum",
            )
            aggregated = aggregated / (weight_sums + EPSILON)

        # === Node Update ===
        node_input = torch.cat([node_enc, aggregated], dim=-1)
        node_update = node_updater(node_input)

        # Apply node DeepONet modulation
        if self.node_deeponet and branch_out is not None:
            node_update = node_update.view(batches, num_nodes, -1)
            node_update = torch.einsum("ijk,ik->ijk", node_update, branch_out)
            node_update = node_update.reshape(batches * num_nodes, -1)

        node_enc = node_enc + node_update

        return edge_enc, node_enc

    def forward(
        self,
        field_data,
        one_hot_node_type,
        edge_index,
        edge_features,
        params,
        mask,
        num_nodes,
        num_edges,
        dirichlet_masks=None,
        pos=None, 
        **kwargs,
        ) -> tuple[torch.Tensor, None]:
        """
        Forward pass of the graph network.

        Args:
            field_data: [batch*nodes, n_fields, bundle_size] input fields
            pos: [batch*nodes, dims] node positions
            one_hot_node_type: [batch*nodes, n_types] node type encoding
            edge_index: [2, batch*edges] edge connectivity
            edge_features: [batch*edges, edge_dim] edge features (dx, dy, dist)
            params: [batch, n_params] simulation parameters for DeepONet
            mask: [batch*nodes] boolean mask for updateable nodes
            dirichlet_masks: {field_idx: node_mask} for Dirichlet boundary conditions
            num_nodes: Number of nodes per graph
            num_edges: Number of edges per graph

        Returns:
            residual: [batch*nodes, n_fields, bundle_size] predicted next state
            None: Placeholder for compatibility
        """
            
        residual = field_data[:, :, -1:].clone().repeat(1, 1, field_data.size(2))
          
        # Select architecture1
        arch = self._forward_compatibility if self.is_compatibility_mode else self._forward_internal
        update = arch(
            field_data, one_hot_node_type, edge_index, edge_features,
            params, num_nodes, num_edges,
        )

        # Center updates if enabled
        if self.center:
            update = update - update.mean(dim=0, keepdim=True)
        
        # Apply Dirichlet boundary conditions
        if dirichlet_masks is not None:
            for field_idx, node_mask in dirichlet_masks.items():
                update[node_mask, field_idx, :] = 0.0
                
        # Apply residual connection
        residual[mask] = residual[mask] + update[mask]
        return residual, None
    
    @staticmethod
    def build_mlp(
        in_size: int,
        out_size: int,
        hidden_width: int,
        hidden_layers: int,
        bottleneck_width: int = 0,
        initial_bias: bool = True,
        final_bias: bool = True,
        final_activation: bool = False,
        initial_norm: Literal["batch", "layer"]| None = None,
        hidden_norm: Literal["batch", "layer"] | None = None,
        final_norm: Literal["batch", "layer"] | None = None,
        hidden_norm_cadence: int = 2, 
        dropout: float = 0.0,
        init_val: float = 0.01,
        activation: str = 'silu',
        ReZero: bool = False,
    ) -> nn.Sequential:
        """
        Build a multi-layer perceptron with configurable architecture.

        Args:
            in_size: Input dimension
            out_size: Output dimension
            hidden_width: Width of hidden layers
            hidden_layers: Number of hidden layers
            bottleneck_width: If > 0, alternate layers use this width
            initial_bias: Whether first layer has bias
            final_bias: Whether last layer has bias
            final_activation: Whether to apply activation after last layer
            initial_norm:
            hidden_norm:
            final_norm:
            dropout: Dropout probability (applied before last layer)
            init_val: Initial value for ReZero scaling
            activation: Activation function
            ReZero: Add ReZero scaling at the end

        Returns:
            Sequential module containing the MLP
        """
        # Build layer sizes
        in_channels = [in_size] + [hidden_width] * hidden_layers
        out_channels = [hidden_width] * hidden_layers + [out_size]
        biases = [initial_bias] + [True] * (hidden_layers - 1) + [final_bias]

        # Apply bottleneck pattern if specified
        if bottleneck_width > 0:
            if hidden_layers < 3 or hidden_layers % 2 == 0:
                warnings.warn(
                    "Bottleneck requires at least 3 odd hidden layers",
                    RuntimeWarning,
                )
            else:
                in_channels[2:-1:2] = [bottleneck_width] * (hidden_layers // 2)
                out_channels[1:-2:2] = [bottleneck_width] * (hidden_layers // 2)

        # Configure normalization
        norm = lambda x:  nn.BatchNorm1d  if x=="batch" else (nn.LayerNorm if x=="layer" else None)
        hnorm = [norm(hidden_norm) if layer % hidden_norm_cadence==0 else None for layer in range(1,hidden_layers)] 
        norms = [norm(initial_norm)] + hnorm + [norm(final_norm)]
        
        # Configure activations
        activations = [get_activation(activation)] * hidden_layers
        activations.append(get_activation(activation) if final_activation else None)

        # Build layers
        layers = []
        for in_ch, out_ch, bias, norm, act in zip(
            in_channels, out_channels, biases, norms, activations
        ):
            layers.append(nn.Linear(in_ch, out_ch, bias=bias))
            if norm is not None:
                layers.append(norm(out_ch))
            if act is not None:
                layers.append(act)
        # Add regularization
        if ReZero:
            layers.append(ReZeroScale(init_val))
        if dropout > 1e-5:
            layers.insert(-2, nn.Dropout(dropout))

        return nn.Sequential(*layers)
# =============================================================================
# Model Factory
# =============================================================================

def get_model(config: dict, n_types: int) -> GraphNetworkSimulator:
    """
    Construct a GraphNetworkSimulator from configuration dictionary.

    Args:
        config: Training configuration dictionary
        n_types: Number of node types (for one-hot encoding)

    Returns:
        Configured GraphNetworkSimulator instance
    """
    # Extract configuration
    n_fields = len(config.get("fields", []))
    n_params = len(config.get("dataset", {}).get("param_keys", []))
    dimensions = config.get("dimensions", 2)
    bundle = config.get("bundle", 1)
    
    model_cfg = config.get("model", {})
    latent_dim = model_cfg.get("latent_dimension", 32)
    n_processors = model_cfg.get("n_processors", 3)
    edge_dropout = model_cfg.get("edge_dropout", 0.0)
    sparsity = model_cfg.get("sparsity")
    center = model_cfg.get("center", False)
    
    deeponet_type = model_cfg.get("deeponet_type", "off")
    time_integration = model_cfg.get("time_integration", "off")
    time_aggregation = model_cfg.get("time_aggregation", "flatten")

    # Network configurations
    edge_enc_cfg = model_cfg.get("edge_encoder", {})
    node_enc_cfg = model_cfg.get("node_encoder", {})
    edge_upd_cfg = model_cfg.get("edge_updater", {})
    node_upd_cfg = model_cfg.get("node_updater", {})
    node_dec_cfg = model_cfg.get("node_decoder", {})
    branch_cfg = model_cfg.get("branch_net")
    attention_cfg = model_cfg.get("edge_attention")

    # Colin mode overrides
    if deeponet_type[:6].lower() == "compat":
        attention_cfg = None
        time_aggregation = "flatten"
        time_integration = "linear"
        edge_dropout = 0.0
        sparsity = None
        branch_net_output_size = latent_dim
    else:
        branch_net_output_size = latent_dim - 1 

    # Build branch network (optional for all modes)
    branch_net = None
    if deeponet_type != "off" and n_params > 0 and branch_cfg is not None:

        print("BUILD BRANCH NET")
        branch_net = GraphNetworkSimulator.build_mlp(
            in_size=n_params, out_size=branch_net_output_size, **branch_cfg
        )
    else:
        print("NO BRANCH NET")

    # Build edge attention
    edge_attention = None
    if attention_cfg is not None:
        edge_attention = GraphNetworkSimulator.build_mlp(
            in_size=latent_dim + 1, out_size=1, **attention_cfg
        )

    # Compute sizes based on time aggregation
    encoder_input_size = n_fields + n_types
    decoder_output_size = n_fields
    node_enc_timesteps = 1

    if time_aggregation == "flatten":
        time_multiplier = 1 if deeponet_type[:6].lower() == "compat" else bundle
        encoder_input_size *= time_multiplier
        decoder_output_size *= bundle
    elif time_aggregation == "conv":
        node_enc_timesteps = bundle

    edge_upd_in = (1 + 2 * node_enc_timesteps) * latent_dim
    node_upd_in = 2 * node_enc_timesteps * latent_dim
    node_upd_out = node_enc_timesteps * latent_dim

    # Build encoders/decoders
    edge_encoder = GraphNetworkSimulator.build_mlp(
        in_size=dimensions + 1, out_size=latent_dim, **edge_enc_cfg
    )
    node_encoder = GraphNetworkSimulator.build_mlp(
        in_size=encoder_input_size, out_size=latent_dim, **node_enc_cfg
    )
    node_decoder = GraphNetworkSimulator.build_mlp(
        in_size=latent_dim, out_size=decoder_output_size, **node_dec_cfg
    )

    # Build processor networks with per-layer ReZero initialization
    edge_updaters = nn.ModuleList([
        GraphNetworkSimulator.build_mlp(
            in_size=edge_upd_in,
            out_size=latent_dim,
            init_val=0.5 ** ((i / 2) + 1),
            **edge_upd_cfg,
        )
        for i in range(n_processors)
    ])
    node_updaters = nn.ModuleList([
        GraphNetworkSimulator.build_mlp(
            in_size=node_upd_in,
            out_size=node_upd_out,
            init_val=0.1 ** ((i / 2) + 1),
            **node_upd_cfg,
        )
        for i in range(n_processors)
    ])

    return GraphNetworkSimulator(
        edge_encoder=edge_encoder,
        node_encoder=node_encoder,
        edge_updaters=edge_updaters,
        node_updaters=node_updaters,
        node_decoder=node_decoder,
        branch_net=branch_net,
        edge_attention=edge_attention,
        bundle_size=bundle,
        node_encoder_input_size=encoder_input_size,
#        num_edge_features=dimensions + 1,
        deeponet_type=deeponet_type,
        time_aggregation=time_aggregation,
        time_integration=time_integration,
        edge_dropout=edge_dropout,
        sparsity=sparsity,
        center=center,
    )