from __future__ import annotations

from typing import Literal

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch_scatter import scatter_max, scatter_mean

from .layers import GraphTransformerStack, MLP, NDABlock


def _ensure_batch(x: Tensor, batch: Tensor | None) -> Tensor:
    if batch is None:
        return x.new_zeros(x.size(0), dtype=torch.long)
    return batch


class CoarseFieldHead(nn.Module):
\
\
\
\
\
\


    def __init__(self, channels: int, bins: int = 32, dropout: float = 0.0) -> None:
        super().__init__()
        self.bins = bins
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(1, channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1, stride=2),
            nn.GroupNorm(1, channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(1, channels),
            nn.GELU(),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, coords: Tensor, batch: Tensor) -> Tensor:
        num_graphs = int(batch.max().item()) + 1 if batch.numel() else 1
        out = torch.zeros_like(x)
        for graph in range(num_graphs):
            mask = batch == graph
            if not bool(mask.any()):
                continue
            feats = x[mask]
            pos = coords[mask]
            lo = pos.min(dim=0).values
            span = (pos.max(dim=0).values - lo).clamp_min(1.0e-9)
            norm = (pos - lo) / span
            idx = (norm * self.bins).floor().clamp(0, self.bins - 1).long()
            flat = idx[:, 0] * self.bins + idx[:, 1]

            cells = self.bins * self.bins
            sums = feats.new_zeros(cells, feats.size(-1))
            counts = feats.new_zeros(cells, 1)
            sums.index_add_(0, flat, feats)
            counts.index_add_(0, flat, torch.ones_like(feats[:, :1]))
            grid = (sums / counts.clamp_min(1.0)).view(
                self.bins, self.bins, -1
            ).permute(2, 0, 1)[None]

            field = self.net(grid)
            sample = (norm * 2.0 - 1.0).flip(-1)[None, :, None, :]
            sampled = F.grid_sample(
                field, sample.to(field.dtype), mode="bilinear",
                padding_mode="border", align_corners=False,
            )[0, :, :, 0].transpose(0, 1)
            out[mask] = self.dropout(sampled)
        return out


def _graph_feature_max(values: Tensor, batch: Tensor, eps: float) -> Tensor:
    num_graphs = int(batch.max().item()) + 1 if batch.numel() else 1
    max_values = scatter_max(values.abs(), batch, dim=0, dim_size=num_graphs)[0]
    max_values = torch.where(torch.isfinite(max_values), max_values, values.new_ones(()))
    return max_values.clamp_min(eps)


class IRGraphFeatureAdapter(nn.Module):
\
\
\
\
\
\
\
\


    def __init__(
        self,
        node_in_channels: int = 7,
        edge_in_channels: int = 4,
        eps: float = 1.0e-12,
    ) -> None:
        super().__init__()
        self.node_in_channels = node_in_channels
        self.edge_in_channels = edge_in_channels
        self.eps = eps

    def forward(
        self,
        node_feat: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor | None,
        batch: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        batch = _ensure_batch(node_feat, batch)
        node_feat = node_feat.float()
        node_norm = self.normalize_nodes(node_feat, batch)

        if edge_attr is None:
            edge_attr = node_feat.new_zeros(edge_index.size(1), self.edge_in_channels)
        else:
            edge_attr = edge_attr.float()
        edge_norm = self.normalize_edges(node_feat, edge_index, edge_attr, batch)
        return node_norm, edge_norm

    def normalize_nodes(self, node_feat: Tensor, batch: Tensor) -> Tensor:
        pieces: list[Tensor] = []
        if self.node_in_channels >= 2:
            xy_scale = _graph_feature_max(node_feat[:, :2], batch, self.eps)
            pieces.append(node_feat[:, :2] / xy_scale[batch])
        if self.node_in_channels >= 3:
            layer_scale = _graph_feature_max(node_feat[:, 2:3], batch, self.eps)
            pieces.append(node_feat[:, 2:3] / layer_scale[batch])
        if self.node_in_channels >= 4:
            pieces.append(torch.log1p(node_feat[:, 3:4].clamp_min(0.0)))
        if self.node_in_channels >= 5:
            pieces.append(node_feat[:, 4:5] * 1.0e6)
        if self.node_in_channels >= 6:
            pieces.append(torch.log1p(node_feat[:, 5:6].clamp_min(0.0)))
        if self.node_in_channels >= 7:
            pieces.append(node_feat[:, 6:7] * 1.0e4)
        if self.node_in_channels > 7:
            pieces.append(node_feat[:, 7:])
        return torch.cat(pieces, dim=-1)

    def normalize_edges(
        self,
        node_feat: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor,
        batch: Tensor,
    ) -> Tensor:
        pieces: list[Tensor] = []
        if self.edge_in_channels >= 1:
            pieces.append(torch.log1p(edge_attr[:, 0:1].clamp_min(0.0)))

        edge_batch = batch[edge_index[1]]
        if self.edge_in_channels >= 3:
            xy_scale = _graph_feature_max(node_feat[:, :2], batch, self.eps)
            pieces.append(edge_attr[:, 1:3] / xy_scale[edge_batch])
        elif self.edge_in_channels >= 2:
            xy_scale = _graph_feature_max(node_feat[:, :1], batch, self.eps)
            pieces.append(edge_attr[:, 1:2] / xy_scale[edge_batch])

        if self.edge_in_channels >= 4:
            layer_scale = _graph_feature_max(node_feat[:, 2:3], batch, self.eps)
            pieces.append(edge_attr[:, 3:4] / layer_scale[edge_batch])
        if self.edge_in_channels > 4:
            pieces.append(edge_attr[:, 4:])
        return torch.cat(pieces, dim=-1)


class IRGNN(nn.Module):
\
\
\
\
\


    def __init__(
        self,
        node_in_channels: int = 7,
        edge_in_channels: int = 4,
        hidden_channels: int = 128,
        nda_layers: int = 2,
        gt_layers: int = 1,
        heads: int = 4,
        dropout: float = 0.1,
        nda_beta: float = 1.0,
        nda_distance_threshold: float | None = None,
        gt_mode: Literal["auto", "full", "landmark"] = "auto",
        full_attention_max_nodes: int = 4096,
        num_landmarks: int = 64,
        landmark_key_bins: int | None = None,
        readout_layers: int = 3,
        residual_to_numerical: bool = True,
        numerical_feature_index: int = 6,
        residual_scale: float = 1.0e-4,
        compose: Literal["parallel", "serial"] = "parallel",
        split_layer_edges: bool = False,
        peak_head: bool = False,
        residual_scale_mode: Literal["const", "design"] = "const",
        residual_scale_ratio: float = 0.05,
        coarse_head: bool = False,
        coarse_bins: int = 32,
        unroll_steps: int = 1,
        residual_zero_init: bool = True,
    ) -> None:
        super().__init__()
        if hidden_channels % heads != 0:
            raise ValueError("hidden_channels must be divisible by heads")
        if nda_layers < 1:
            raise ValueError("nda_layers must be >= 1")
        if gt_layers < 1:
            raise ValueError("gt_layers must be >= 1")

        self.node_in_channels = node_in_channels
        self.edge_in_channels = edge_in_channels
        self.hidden_channels = hidden_channels
        self.residual_to_numerical = residual_to_numerical
        self.numerical_feature_index = numerical_feature_index
        self.residual_scale = residual_scale
        self.compose = compose
        self.split_layer_edges = split_layer_edges
        self.peak_head = peak_head
        self.residual_scale_mode = residual_scale_mode
        self.residual_scale_ratio = residual_scale_ratio
        self.unroll_steps = max(int(unroll_steps), 1)
        self.residual_zero_init = residual_zero_init
        self.landmark_key_bins = landmark_key_bins

        self.feature_adapter = IRGraphFeatureAdapter(
            node_in_channels=node_in_channels,
            edge_in_channels=edge_in_channels,
        )
        self.node_encoder = MLP(
            node_in_channels,
            hidden_channels,
            hidden_channels,
            num_layers=2,
            dropout=dropout,
        )
        self.edge_encoder = MLP(
            edge_in_channels,
            hidden_channels,
            hidden_channels,
            num_layers=2,
            dropout=dropout,
        )

        def make_nda_stack() -> nn.ModuleList:
            return nn.ModuleList(
                [
                    NDABlock(
                        hidden_channels=hidden_channels,
                        edge_channels=hidden_channels,
                        heads=heads,
                        dropout=dropout,
                        beta=nda_beta,
                        distance_threshold=nda_distance_threshold,
                    )
                    for _ in range(nda_layers)
                ]
            )

        self.nda_blocks = make_nda_stack()
        self.nda_blocks_via = make_nda_stack() if split_layer_edges else None
        self.gt_stack = GraphTransformerStack(
            num_layers=gt_layers,
            hidden_channels=hidden_channels,
            heads=heads,
            dropout=dropout,
            mode=gt_mode,
            full_attention_max_nodes=full_attention_max_nodes,
            num_landmarks=num_landmarks,
            landmark_key_bins=landmark_key_bins,
        )
        readout_in = hidden_channels if compose == "serial" else 2 * hidden_channels
        self.coarse_field = (
            CoarseFieldHead(hidden_channels, bins=coarse_bins, dropout=dropout)
            if coarse_head
            else None
        )
        if self.coarse_field is not None:
            readout_in += hidden_channels
        self.readout = MLP(
            readout_in,
            hidden_channels,
            1,
            num_layers=readout_layers,
            dropout=dropout,
        )
        self.peak_readout = (
            MLP(
                readout_in,
                hidden_channels,
                1,
                num_layers=readout_layers,
                dropout=dropout,
            )
            if peak_head
            else None
        )


        self.step_sizes = nn.Parameter(torch.ones(self.unroll_steps))
        if self.residual_to_numerical and self.residual_zero_init:
            last_linear = self.readout.net[-1]
            nn.init.zeros_(last_linear.weight)
            nn.init.zeros_(last_linear.bias)
            if self.peak_readout is not None:
                nn.init.zeros_(self.peak_readout.net[-1].weight)
                nn.init.zeros_(self.peak_readout.net[-1].bias)

    def _apply_nda_stack(
        self,
        x: Tensor,
        blocks: nn.ModuleList,
        edge_index: Tensor,
        edge_hidden: Tensor,
        edge_direction: Tensor | None,
        distance_attr: Tensor,
    ) -> Tensor:
        for block in blocks:
            x = block(
                x,
                edge_index=edge_index,
                edge_attr=edge_hidden,
                edge_direction=edge_direction,
                distance_attr=distance_attr,
            )
        return x

    def _run_nda(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_hidden: Tensor,
        edge_direction: Tensor | None,
        distance_attr: Tensor,
        raw_edge_attr: Tensor | None,
    ) -> Tensor:
        if (
            not self.split_layer_edges
            or self.nda_blocks_via is None
            or raw_edge_attr is None
            or raw_edge_attr.size(-1) < 4
        ):
            return self._apply_nda_stack(
                x,
                self.nda_blocks,
                edge_index,
                edge_hidden,
                edge_direction,
                distance_attr,
            )


        intra = raw_edge_attr[:, 3].abs() < 0.5
        via = ~intra
        h = x
        intra_dir = None if edge_direction is None else edge_direction[intra]
        via_dir = None if edge_direction is None else edge_direction[via]
        has_intra = bool(intra.any())
        has_via = bool(via.any())


        for metal_block, via_block in zip(self.nda_blocks, self.nda_blocks_via):
            if has_intra:
                h = metal_block(
                    h,
                    edge_index=edge_index[:, intra],
                    edge_attr=edge_hidden[intra],
                    edge_direction=intra_dir,
                    distance_attr=distance_attr[intra],
                )
            if has_via:
                h = via_block(
                    h,
                    edge_index=edge_index[:, via],
                    edge_attr=edge_hidden[via],
                    edge_direction=via_dir,
                    distance_attr=distance_attr[via],
                )
        return h

    def _residual_scale(self, node_feat: Tensor, batch: Tensor) -> Tensor:

        if self.residual_scale_mode != "design":
            return torch.as_tensor(
                self.residual_scale, device=node_feat.device, dtype=node_feat.dtype
            )
        rough = node_feat[:, self.numerical_feature_index].abs().float()
        per_graph = scatter_mean(rough, batch, dim=0)
        return (per_graph[batch] * self.residual_scale_ratio).clamp_min(1.0e-9)

    def forward(
        self,
        node_feat: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor | None = None,
        batch: Tensor | None = None,
        edge_direction: Tensor | None = None,
        return_dict: bool = False,
    ) -> Tensor | dict[str, Tensor]:
        batch = _ensure_batch(node_feat, batch)
        node_norm, edge_norm = self.feature_adapter(
            node_feat, edge_index, edge_attr, batch=batch
        )
        x0 = self.node_encoder(node_norm)
        edge_hidden = self.edge_encoder(edge_norm)

        local = self._run_nda(
            x0,
            edge_index=edge_index,
            edge_hidden=edge_hidden,
            edge_direction=edge_direction,
            distance_attr=edge_norm,
            raw_edge_attr=edge_attr,
        )

        if self.compose == "serial":
            global_context = self.gt_stack(local, batch=batch)
            fused = global_context
        else:
            global_context = self.gt_stack(x0, batch=batch)
            fused = torch.cat([local, global_context], dim=-1)
        if self.coarse_field is not None:
            coarse = self.coarse_field(fused, node_feat[:, :2].float(), batch)
            fused = torch.cat([fused, coarse], dim=-1)
        delta = self.readout(fused).squeeze(-1)
        if self.peak_readout is not None:
            delta = delta + self.peak_readout(fused).squeeze(-1)

        if self.residual_to_numerical and node_feat.size(1) > self.numerical_feature_index:
            pred = node_feat[:, self.numerical_feature_index].float()
            pred = pred + delta * self._residual_scale(node_feat, batch)
        else:
            pred = delta

        if return_dict:
            return {
                "pred": pred,
                "delta": delta,
                "local": local,
                "global": global_context,
                "fused": fused,
            }
        return pred

    def forward_data(self, data, return_dict: bool = False):


        steps = self.unroll_steps
        system = getattr(data, "physics", None)
        if steps == 1 or system is None:
            return self(
                node_feat=data.x,
                edge_index=data.edge_index,
                edge_attr=getattr(data, "edge_attr", None),
                batch=getattr(data, "batch", None),
                edge_direction=getattr(data, "edge_direction", None),
                return_dict=return_dict,
            )
        return self._forward_unrolled(data, system, return_dict=return_dict)

    def _forward_unrolled(self, data, system, return_dict: bool = False):


        from .physics import physics_node_features

        base = getattr(data, "base_x", None)
        if base is None:
            raise ValueError("unrolled refinement needs physics features on the graph")
        if system.rhs.device != data.x.device:

            system = system.to(data.x.device)
            data.physics = system
        sweeps = tuple(getattr(data, "physics_sweeps", ())) or None
        batch = _ensure_batch(data.x, getattr(data, "batch", None))
        values = base[:, self.numerical_feature_index].float()
        preds: list[Tensor] = []
        for step in range(self.unroll_steps):
            if step == 0:
                node_feat = data.x
            else:
                with torch.no_grad():
                    kwargs = {"sweeps": sweeps} if sweeps else {}
                    feats = physics_node_features(
                        system, base, values=values.detach(), **kwargs
                    )
                updated = base.clone()
                updated[:, self.numerical_feature_index] = values
                node_feat = torch.cat([updated, feats.to(updated.dtype)], dim=-1)
            out = self(
                node_feat=node_feat,
                edge_index=data.edge_index,
                edge_attr=getattr(data, "edge_attr", None),
                batch=batch,
                edge_direction=getattr(data, "edge_direction", None),
                return_dict=True,
            )
            delta = out["delta"] * self._residual_scale(node_feat, batch)
            values = values + self.step_sizes[step] * delta
            preds.append(values)

        if return_dict:
            return {"pred": preds[-1], "preds": preds}
        return preds[-1]
