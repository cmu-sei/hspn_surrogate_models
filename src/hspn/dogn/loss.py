import numpy as np
import torch
import torch.nn as nn

class LossFunction(nn.Module):
    """ Returns a custom loss
    """
    def __init__(self, lambda_gradient=2.0):
        super().__init__()
        self.lambda_gradient = lambda_gradient

    def set_graph(self, edge_index, edge_features, num_nodes):
        self.edge_index = edge_index
        self.edge_features = edge_features
        self.num_nodes = num_nodes

    def forward(self, prediction, truth):
        
        # MSE
        mse_per_node = ((prediction - truth) ** 2).mean(dim=1)
        
        # ASSUMES this structure
        u_true, v_true = truth[:, -2, :], truth[:, -1, :]
        non_self = self.edge_index[0] != self.edge_index[1]
        source, target = self.edge_index[:, non_self]

        edge_length = self.edge_features[non_self, 2].unsqueeze(1)
        du = (u_true[target] - u_true[source]) / edge_length
        dv = (v_true[target] - v_true[source]) / edge_length
        grad_magnitude = torch.sqrt(du**2 + dv**2)

        node_gradient = torch.zeros_like(mse_per_node)
        idx = lambda x: x.unsqueeze(0).expand(grad_magnitude.size(1), -1).T
        node_gradient.scatter_reduce_(0, idx(source), grad_magnitude, reduce="max")
        node_gradient.scatter_reduce_(0, idx(target), grad_magnitude, reduce="max")

        # weights areas associated with a large gradient in velocity
        weights = 1 + (self.lambda_gradient - 1) * (
            node_gradient / (node_gradient.max() + 1e-8)
        )
        
        #weighted MSE
        weighted = (mse_per_node * weights).reshape(-1)

        return weighted.mean()


class EnergyLoss(nn.Module):
    """ Returns a loss based on the velocity and buoyancy of a field (2D). Includes APE
    """
    def __init__(self, velocity_indices, buoyancy_idx, node_heights):
        super().__init__()
        self.velocity_indices = velocity_indices
        self.buoyancy_idx = buoyancy_idx
        self.register_buffer("node_heights", node_heights)
        self.register_buffer("heights_sorted", torch.sort(node_heights).values)

    def _reshape(self, field_data, num_nodes):
        batch_size = field_data.shape[0] // num_nodes
        t = field_data.shape[-1]
        return field_data.reshape(batch_size, num_nodes, -1, t)

    def kinetic_energy(self, field_data, num_nodes):
        f = self._reshape(field_data, num_nodes)
        return 0.5 * sum(
            (f[:, :, i, :] ** 2).mean(dim=1) for i in self.velocity_indices
        )

    def potential_energies(self, field_data, buo_coeff, num_nodes):
        """Returns APE, PE_min"""
        f = self._reshape(field_data, num_nodes)

        u = f[:, :, self.buoyancy_idx, :]
        h = self.node_heights.view(1, -1, 1)
        pe = -buo_coeff.view(-1, 1) * (u * h).mean(dim=1)

        u_sorted = torch.sort(u.detach(), dim=1).values # sharp gradients 
        h_sorted = self.heights_sorted.view(1, -1, 1)
        pe_min = -buo_coeff.view(-1, 1) * (u_sorted * h_sorted).mean(dim=1)

        return pe - pe_min, pe_min

    def forward(
        self,
        pred,
        target,
        buo_coeff,
        num_nodes,
        ke_weight=1.0,
        ape_weight=1.0,
        pe_min_weight=1.0,
        monotonic_weight=0.1,
    ):
        pred_ke = self.kinetic_energy(pred, num_nodes)
        pred_ape, pred_pe_min = self.potential_energies(pred, buo_coeff, num_nodes)

        target_ke = self.kinetic_energy(target, num_nodes)
        target_ape, target_pe_min = self.potential_energies(
            target, buo_coeff, num_nodes
        )

        ke_loss = ((pred_ke - target_ke) ** 2).mean()
        ape_loss = ((pred_ape - target_ape) ** 2).mean()
        pe_min_loss = ((pred_pe_min - target_pe_min) ** 2).mean()

        total_E = pred_ke + pred_ape + pred_pe_min
        mono_loss = 0.0
        if total_E.shape[-1] > 1:
            dE = total_E[:, 1:] - total_E[:, :-1]
            mono_loss = torch.relu(dE).mean()

        total = (
            ke_weight * ke_loss
            + ape_weight * ape_loss
            + pe_min_weight * pe_min_loss
            + monotonic_weight * mono_loss
        )

        return (
            total,
            ke_loss.item(),
            ape_loss.item(),
            pe_min_loss.item(),
            mono_loss if isinstance(mono_loss, float) else mono_loss.item(),
        )



def forward_loss(
    model, data, target, criterion, energy_loss_fn, energy_config,field_names=None
):
    pred, orig = model(**data)
    mse_loss = criterion( pred, target)
    energy_loss, ke, ape, pe_min, mono = 0.0, 0.0, 0.0, 0.0, 0.0
    if energy_loss_fn:
        buo_coeff = data["params"][:, 0]
        energy_loss, ke, ape, pe_min, mono = energy_loss_fn(
            pred,
            target,
            buo_coeff,
            data["num_nodes"],
            energy_config.get("ke_weight", 1.0),
            energy_config.get("ape_weight", 1.0),
            energy_config.get("pe_min_weight", 1.0),
            energy_config.get("monotonic_weight", 1.0),
        )
    loss = (
        mse_loss
        + energy_loss
        + (nn.MSELoss()(orig, data["field_data"]) if orig is not None else 0)
    )
    mse_val = mse_loss.item() if isinstance(mse_loss, torch.Tensor) else mse_loss
    
    result = {
        "loss": loss,
        "mse": mse_val,
        "ke": ke,
        "ape": ape,
        "pe_min": pe_min,
        "mono": mono,
    }
    
    # Compute mean of each field for both pred and target
    pred_means = pred.mean(dim=(0, 2))
    target_means = target.mean(dim=(0, 2))
    
    for i, (pm, tm) in enumerate(zip(pred_means, target_means)):
        name = field_names[i] if field_names else f"field_{i}"
        result[f"{name}_pred_mean"] = pm.item()
        result[f"{name}_true_mean"] = tm.item()
    
    return result