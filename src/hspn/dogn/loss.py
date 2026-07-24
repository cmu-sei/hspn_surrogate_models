#!/bin/sh
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
#!/bin/bash

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def diffuse_weights(weights, edge_index, iterations=2, alpha=0.5):
    """Spread importance weights to graph neighbors."""
    for _ in range(iterations):
        neighbor_sum = torch.zeros_like(weights)
        neighbor_sum.scatter_add_(
            0,
            edge_index[1].unsqueeze(-1).expand_as(weights[edge_index[0]]),
            weights[edge_index[0]],
        )
        degree = torch.zeros(weights.shape[0], 1, device=weights.device)
        degree.scatter_add_(
            0,
            edge_index[1].unsqueeze(-1),
            torch.ones(edge_index.shape[1], 1, device=weights.device),
        )
        neighbor_avg = neighbor_sum / degree.clamp(min=1)
        weights = alpha * weights + (1 - alpha) * neighbor_avg
    return weights

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

class CompositeLoss(nn.Module):
    def __init__(self, config, energy_loss_fn=None):
        super().__init__()
        loss_cfg = config.get("loss", {})
        energy_cfg = loss_cfg.get("energy", {})
        
        self.persistence_weight = loss_cfg.get("persistence_weight", 0)
        self.persistence_eps = loss_cfg.get("persistence_eps", 1e-2)
        
        self.change_weight = loss_cfg.get("change_weight", 0)
        self.change_fields = loss_cfg.get("change_fields", [])
        self.change_diffuse_iterations = loss_cfg.get("change_diffuse_iterations", 2)
        self.change_diffuse_alpha = loss_cfg.get("change_diffuse_alpha", 0.5)
        
        self.ae_weight = loss_cfg.get("ae_weight", 0)
        self.latent_weight = loss_cfg.get("latent_weight",0)
        
        
        lg = energy_cfg.get("lambda_gradient", 0)
        if lg > 0 and energy_loss_fn is not None:
            self.criterion = LossFunction(lambda_gradient=lg)
        else:
            self.criterion = nn.MSELoss()
            
        self.energy_loss_fn = energy_loss_fn
        self.energy_weights = {
            k: energy_cfg.get(k, w)
            for k, w in [("ke_weight", 1.0), ("ape_weight", 1.0),
                        ("pe_min_weight", 1.0), ("monotonic_weight", 0.1)]
        }
        
        
        self.spectral_weight = loss_cfg.get("spectral_weight", 0)

    def set_graph(self, edge_index, edge_features, num_nodes):
        if hasattr(self.criterion, "set_graph"):
            self.criterion.set_graph(edge_index, edge_features, num_nodes)

    def forward(self, pred, target, ae_pred=None, 
            enc_y_pred=None, enc_y_true=None,
            field_data=None, edge_index=None, 
            energy_data=None,
            pred_seq=None, true_seq=None,epoch=0):
        """
        Args:
            pred:        model prediction (batch_nodes, fields, bundle)
            target:      ground truth
            num_nodes:   nodes per graph (for per-graph spectral loss)
            ae_pred:        pre-message-passing decoder output (AE term)
            energy_data: dict with params, num_nodes (for energy loss)
            pred_seq:    predicted sequence for spectral loss
            true_seq:    true sequence for spectral loss
            field_data:  field data
            edge_index:  edge index
            epoch:       Epoch, in case of any decaying weights.

        Returns:
            dict with "loss" and individual metrics
        """
        #loss = self.criterion(pred, target)
        m = {}#"mse": loss.item()}
        if self.change_weight > 0 and field_data is not None and edge_index is not None:
            if self.change_fields:
                fd = field_data[:, self.change_fields, :]
                tgt = target[:, self.change_fields, :]
            else:
                fd = field_data
                tgt = target
                
            delta = (tgt - fd[..., -1:]).abs().mean(dim=(1, 2))  # (nodes,)
            weights = 1 + self.change_weight * (delta / (delta.max() + 1e-8))
            weights = weights.unsqueeze(-1)  # (nodes, 1) for diffuse_weights
            weights = diffuse_weights(weights, edge_index,
                                    iterations=self.change_diffuse_iterations,
                                    alpha=self.change_diffuse_alpha)
            weights = weights.view(-1, 1, 1)  # (nodes, 1, 1) for broadcasting with (nodes, fields, bundle)
            loss = (weights * (pred - target) ** 2).mean()     
        else:
            loss = self.criterion(pred, target)
        

        m["mse"] = loss.item()
        
        

        if self.ae_weight > 0 and ae_pred is not None:
            ae = (ae_pred ** 2).mean()
            loss = loss + self.ae_weight * ae
            m["ae"] = ae.item()
            
        if self.latent_weight > 0 and enc_y_pred is not None and enc_y_true is not None:
            enc_ae = self.criterion(enc_y_pred, enc_y_true)
            loss = loss + self.latent_weight * enc_ae
            m["latent"] = enc_ae.item()
            
            
        if self.persistence_weight > 0 and field_data is not None:
            pred_delta = (pred - field_data[..., -1:]).abs()
            true_delta = (target - field_data[..., -1:]).abs()
            shortfall = torch.relu(true_delta - pred_delta - self.persistence_eps)
            persistence_loss = shortfall.mean()
            loss = loss + self.persistence_weight * persistence_loss
            m["persistence"] = persistence_loss.item()
            

        if self.energy_loss_fn is not None and energy_data is not None:
            eloss, ke, ape, pe_min, mono = self.energy_loss_fn(
                pred, target, energy_data["params"][:, 0],
                energy_data["num_nodes"], **self.energy_weights,
            )
            loss = loss + eloss
            m.update(ke=ke, ape=ape, pe_min=pe_min, mono=mono)
            
        if self.spectral_weight > 0 and pred_seq is not None and true_seq is not None:
            # Per-graph SVD since nodes are concatenated across graphs
            n = num_nodes or pred_seq.shape[0]
            losses = []
            for i in range(pred_seq.shape[0] // n):
                p = pred_seq[i*n:(i+1)*n].reshape(-1, pred_seq.shape[-1])
                t = true_seq[i*n:(i+1)*n].reshape(-1, true_seq.shape[-1])
                losses.append(F.mse_loss(torch.linalg.svdvals(p), torch.linalg.svdvals(t)))
            spec = torch.stack(losses).mean()
            loss = loss + self.spectral_weight * spec
            m["spectral"] = spec.item()
   
        m['loss'] = loss 
        return m
