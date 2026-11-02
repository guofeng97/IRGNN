from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

PHYSICS_ALPHA = 0.4


DEFAULT_SWEEPS = (1, 4, 16, 64)
DEFAULT_OMEGA = 2.0 / 3.0
DEFAULT_COARSE_BINS = (4, 8, 16, 32)

NUM_PHYSICS_FEATURES = len(DEFAULT_SWEEPS) + len(DEFAULT_COARSE_BINS) + 2


def num_physics_features(
    sweeps: tuple[int, ...] = DEFAULT_SWEEPS,
    coarse_bins: tuple[int, ...] = DEFAULT_COARSE_BINS,
) -> int:
    return len(sweeps) + len(coarse_bins) + 2


@dataclass(frozen=True)
class PhysicsSystem:


    edge_index: Tensor
    conductance: Tensor
    rhs: Tensor
    diag: Tensor
    pad_mask: Tensor

    def to(self, device: torch.device | str) -> "PhysicsSystem":
        return PhysicsSystem(
            edge_index=self.edge_index.to(device),
            conductance=self.conductance.to(device),
            rhs=self.rhs.to(device),
            diag=self.diag.to(device),
            pad_mask=self.pad_mask.to(device),
        )


def build_physics_system(
    node_feat: Tensor,
    edge_index: Tensor,
    edge_attr: Tensor,
    raw_resistor_edge_count: int | None = None,
    alpha: float = PHYSICS_ALPHA,
    eps: float = 1.0e-12,
) -> PhysicsSystem:


    resistance = edge_attr[:, 0].to(torch.float64).clamp_min(eps)
    conductance = 1.0 / resistance
    duplicated = (
        raw_resistor_edge_count is not None
        and int(edge_index.size(1)) == 2 * int(raw_resistor_edge_count)
    )
    conductance = conductance * (alpha / (2.0 if duplicated else 1.0))

    num_nodes = int(node_feat.size(0))
    src = edge_index[0]
    diag = torch.zeros(num_nodes, dtype=torch.float64, device=node_feat.device)
    diag.scatter_add_(0, src, conductance)

    rough = node_feat[:, 6].to(torch.float64)


    pad_mask = rough == 0.0
    return PhysicsSystem(
        edge_index=edge_index,
        conductance=conductance,
        rhs=node_feat[:, 4].to(torch.float64),
        diag=diag,
        pad_mask=pad_mask,
    )


def apply_operator(system: PhysicsSystem, values: Tensor) -> Tensor:


    src, dst = system.edge_index[0], system.edge_index[1]
    values = values.to(torch.float64)
    flow = system.conductance * (values[src] - values[dst])
    out = torch.zeros_like(values)
    out.scatter_add_(0, src, flow)
    return out


def physics_residual(system: PhysicsSystem, values: Tensor) -> Tensor:


    residual = system.rhs - apply_operator(system, values)
    return residual.masked_fill(system.pad_mask, 0.0)


def jacobi_refine(
    system: PhysicsSystem,
    residual: Tensor,
    sweeps: tuple[int, ...] = DEFAULT_SWEEPS,
    omega: float = DEFAULT_OMEGA,
) -> list[Tensor]:


    inv_diag = 1.0 / system.diag.clamp_min(1.0e-30)
    error = torch.zeros_like(residual)
    outputs: list[Tensor] = []
    total = max(sweeps)
    wanted = set(sweeps)
    for step in range(1, total + 1):
        update = residual - apply_operator(system, error)
        error = error + omega * inv_diag * update
        error = error.masked_fill(system.pad_mask, 0.0)
        if step in wanted:
            outputs.append(error.clone())
    return outputs


def _coarse_bin_mean(
    values: Tensor, node_feat: Tensor, num_bins: int
) -> Tensor:


    coords = node_feat[:, :2].to(torch.float64)
    lo = coords.min(dim=0).values
    span = (coords.max(dim=0).values - lo).clamp_min(1.0e-9)
    idx = ((coords - lo) / span * num_bins).floor().clamp(0, num_bins - 1).long()
    flat = idx[:, 0] * num_bins + idx[:, 1]
    sums = torch.zeros(num_bins * num_bins, dtype=values.dtype, device=values.device)
    counts = torch.zeros_like(sums)
    sums.scatter_add_(0, flat, values)
    counts.scatter_add_(0, flat, torch.ones_like(values))
    means = sums / counts.clamp_min(1.0)
    return means[flat]


def physics_node_features(
    system: PhysicsSystem,
    node_feat: Tensor,
    values: Tensor | None = None,
    sweeps: tuple[int, ...] = DEFAULT_SWEEPS,
    omega: float = DEFAULT_OMEGA,
    coarse_bins: tuple[int, ...] = DEFAULT_COARSE_BINS,
    eps: float = 1.0e-30,
) -> Tensor:


    if values is None:
        values = node_feat[:, 6]
    values = values.to(torch.float64)
    residual = physics_residual(system, values)
    free = ~system.pad_mask
    scale = values[free].abs().mean().clamp_min(1.0e-12)

    errors = jacobi_refine(system, residual, sweeps=sweeps, omega=omega)
    columns = [error / scale for error in errors]
    for num_bins in coarse_bins:
        columns.append(_coarse_bin_mean(errors[-1], node_feat, num_bins) / scale)
    columns.append(torch.log1p(system.diag / system.diag.mean().clamp_min(eps)))

    rhs_norm = system.rhs[free].norm().clamp_min(eps)
    rel_residual = residual[free].norm() / rhs_norm
    columns.append(torch.full_like(values, float(torch.log1p(rel_residual))))
    return torch.stack(columns, dim=-1).to(node_feat.dtype)
