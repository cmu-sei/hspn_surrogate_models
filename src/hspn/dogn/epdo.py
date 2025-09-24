import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def build_mlp(
    in_channels: int,
    out_channels: int,
    mlp_width: int = 128,
    mlp_depth: int = 3,
    layer_norm: bool = True,
    activation: str = "gelu",
    dropout: int | list[int] = 0,
    plain_last: bool = True,
) -> nn.Sequential:
    if not isinstance(dropout, list):
        dropout = [dropout] * mlp_depth

    activation_map = {
        "relu": nn.ReLU(),
        "gelu": nn.GELU(),
        "silu": nn.SiLU(),
        "swish": nn.SiLU(),
        "elu": nn.ELU(),
        "leakyrelu": nn.LeakyReLU(),
        "tanh": nn.Tanh(),
        "sigmoid": nn.Sigmoid(),
    }

    layers = []

    # Build layers following PyG pattern
    for i in range(mlp_depth):
        in_dim = in_channels if i == 0 else mlp_width
        out_dim = out_channels if i == mlp_depth - 1 else mlp_width

        # Add linear layer
        layers.append(nn.Linear(in_dim, out_dim))

        # After each linear (except the last if plain_last=True)
        if i < mlp_depth - 1:
            # Add BatchNorm
            layers.append(nn.BatchNorm1d(mlp_width))
            # Add activation
            layers.append(activation_map.get(activation.lower(), nn.GELU()))
            # Add dropout if specified
            if dropout[i] > 0:
                layers.append(nn.Dropout(dropout[i]))

    # Add LayerNorm at the end if requested
    if layer_norm:
        layers.append(nn.LayerNorm(out_channels))

    return nn.Sequential(*layers)


def get_updaters(
    in_channels: int,
    out_channels: int,
    hidden_channels: int,
    mlp_width: int = 128,
    mlp_depth: int = 3,
    dropout: int | list[int] = 0,
    activation: str = "gelu",
):
    edge_mlp = build_mlp(
        in_channels * 3,
        hidden_channels,
        mlp_width=mlp_width,
        mlp_depth=mlp_depth,
        activation=activation,
        dropout=dropout,
    )
    node_mlp = build_mlp(
        in_channels + hidden_channels,
        out_channels,
        mlp_width=mlp_width,
        mlp_depth=mlp_depth,
        activation=activation,
        dropout=dropout,
    )
    return node_mlp, edge_mlp


class EPDOModel(torch.nn.Module):
    def __init__(
        self,
        d_inp: int,
        d_out: int,
        n_typ: int,
        hidden_channels: int = 128,
        mlp_width: int = 128,
        mlp_depth: int = 3,
        dropout: int | list[int] = 0,
        n_processing_blocks: int = 15,
        checkpointed: bool = False,
        activation: str = "gelu",
        smoothing: int = 0,  # legacy, ignored.
        skip: bool = True,  # legacy, ignored.
        reweight: bool = False,  # legacy, ignored
        **kwargs,
    ):
        """PyGNN implementation of the Encode-Process-Decode model.

        See https://arxiv.org/pdf/2010.03409.
        """

        super(EPDOModel, self).__init__()

        self.n_processing_blocks = n_processing_blocks
        self.checkpointed = checkpointed
        self.n_typ = n_typ
        self.activation = activation

        self.node_encoder = build_mlp(
            d_inp + n_typ,
            hidden_channels,
            mlp_width=mlp_width,
            mlp_depth=mlp_depth,
            activation=self.activation,
        )
        self.edge_encoder = build_mlp(
            3,
            hidden_channels,
            mlp_width=mlp_width,
            mlp_depth=mlp_depth,
            activation=self.activation,
        )
        self.node_decoder = build_mlp(
            hidden_channels,
            d_out,
            mlp_width=mlp_width,
            mlp_depth=mlp_depth,
            layer_norm=False,
            activation=self.activation,
        )

        node_ups = []
        edge_ups = []
        for _ in range(n_processing_blocks):
            node_up, edge_up = get_updaters(
                hidden_channels,
                hidden_channels,
                hidden_channels,
                mlp_width=mlp_width,
                mlp_depth=mlp_depth,
                activation=self.activation,
                dropout=dropout,
            )
            node_ups.append(node_up)
            edge_ups.append(edge_up)

        self.node_updaters = nn.ModuleList(node_ups)
        self.edge_updaters = nn.ModuleList(edge_ups)

        # kludge
        self.branch_net = torch.nn.Sequential(
            torch.nn.Linear(
                2, 40
            ),  # back to 2 dimensions for simple demo# for extra dimension re, 16-32 for  extra dimension re
            torch.nn.SiLU(),
            # torch.nn.Linear(16,mlp_width),
            torch.nn.Linear(40, mlp_width),  # 16-32 for  extra dimension re
            torch.nn.SiLU(),
            torch.nn.Linear(mlp_width, mlp_width),  # added when adding extra dimension re
            # torch.nn.Dropout(0.0625),# added when adding extra dimension re
            torch.nn.SiLU(),  # added when adding extra dimension re
            torch.nn.Linear(mlp_width, hidden_channels),
        )  # get rid of the -3

        self.init_weights()
        # print(self)

    def init_weights(self) -> None:
        """Initialize weights via Kaiming initialization."""

        # anecdotally switching from xavier to kaiming made a huge difference in startup
        def kaiming(X):
            if isinstance(X, torch.nn.Linear):
                torch.nn.init.kaiming_normal_(X.weight, nonlinearity="relu")
                X.bias.data.fill_(0.0)

        self.apply(kaiming)

    @staticmethod
    def _edge_features(pos: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        diff = pos[edge_index[1]] - pos[edge_index[0]]
        dist = torch.norm(diff, dim=1, keepdim=True)
        return torch.cat([diff, dist], dim=-1)

    def forward_internal(self, field_data, one_hot_node_type, edge_index, edge_features, params) -> torch.Tensor:
        _, n_fields = field_data.shape

        x = torch.cat([field_data, one_hot_node_type], dim=-1)

        node_enc = self.node_encoder(x)
        edge_enc = self.edge_encoder(edge_features)
        branch_out = self.branch_net(params)
        enum = 0
        for node_up, edge_up in zip(self.node_updaters, self.edge_updaters):
            # for debugging
            # if enum>0:
            #    break
            # enum+=1
            # Get source and destination node features
            src, dst = node_enc[edge_index[0]], node_enc[edge_index[1]]

            # Update edge features
            edge_enc = edge_up(torch.cat([edge_enc, dst, src], dim=-1))

            # Aggregate edge features to destination nodes (out-of-place)
            update = edge_enc.new_zeros(node_enc.size(0), edge_enc.size(1))
            update = update.scatter_reduce(0, edge_index[1].view(-1, 1).expand_as(edge_enc), edge_enc, "sum")

            # Update node features
            features = [node_enc, update]
            node_update = node_up(torch.cat(features, dim=-1))

            node_enc = node_enc + node_update

            node_enc = torch.einsum(
                "ijk,ik->ijk", node_enc.view(branch_out.size(0), -1, node_enc.size(1)), branch_out
            ).reshape(-1, node_enc.size(1))

        update = self.node_decoder(node_enc)
        L2, field_bundles = update.shape
        bundle_size = field_bundles // n_fields
        multiplier = torch.arange(1, bundle_size + 1, dtype=update.dtype, device=update.device)
        update = torch.einsum("nft,t->nft", update.view(L2, n_fields, bundle_size), multiplier)
        return update.view(L2, n_fields, bundle_size)

    def forward(
        self, field_data, one_hot_node_type, edge_index, edge_features, params, bundle_size, **kwargs
    ) -> torch.Tensor:
        update = self.forward_internal(field_data[:, :, -1], one_hot_node_type, edge_index, edge_features, params)

        residual = field_data[:, :, -1:].repeat(1, 1, bundle_size)  # Now [nodes, fields, bundle_size]

        if hasattr(kwargs, "mask") and kwargs["mask"] is not None:
            residual[mask] = residual[mask] + update[mask]
        else:
            residual = residual + update

        return residual


class EPDOOnnx(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x, one_hot, edge_index, edge_features, params):
        return self.model.forward_internal(x, one_hot_node_type, edge_index, edge_features, params)
