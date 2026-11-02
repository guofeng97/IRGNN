\
\
\
\
\


from __future__ import annotations

from typing import Literal, Optional

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax, to_dense_batch


class MLP(nn.Module):


    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int = 2,
        dropout: float = 0.0,
        use_norm: bool = True,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")

        layers: list[nn.Module] = []
        last_channels = in_channels
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(last_channels, hidden_channels))
            if use_norm:
                layers.append(nn.LayerNorm(hidden_channels))
            layers.append(nn.GELU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            last_channels = hidden_channels
        layers.append(nn.Linear(last_channels, out_channels))
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class NeighborDistanceAttentionConv(MessagePassing):
\
\
\
\
\
\
\
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
        hidden_channels: int,
        edge_channels: int,
        heads: int = 4,
        concat: bool = True,
        beta: float = 1.0,
        distance_threshold: float | None = None,
        max_distance_weight: float = 16.0,
        dropout: float = 0.0,
        negative_slope: float = 0.2,
        eps: float = 1.0e-12,
    ) -> None:
        super().__init__(aggr="add", node_dim=0)
        if hidden_channels % heads != 0:
            raise ValueError("hidden_channels must be divisible by heads")
        if distance_threshold is not None and distance_threshold <= 0:
            raise ValueError("distance_threshold must be positive")

        self.hidden_channels = hidden_channels
        self.edge_channels = edge_channels
        self.heads = heads
        self.head_channels = hidden_channels // heads
        self.concat = concat
        self.beta = beta
        self.distance_threshold = distance_threshold
        self.max_distance_weight = max_distance_weight
        self.dropout = dropout
        self.negative_slope = negative_slope
        self.eps = eps

        self.node_lin = nn.Linear(hidden_channels, hidden_channels, bias=False)
        self.edge_lin = nn.Linear(edge_channels, hidden_channels, bias=False)
        self.att = nn.Parameter(torch.empty(heads, 3 * self.head_channels))


        self.register_buffer("running_threshold", torch.zeros(()))
        self.register_buffer("threshold_initialised", torch.zeros((), dtype=torch.bool))

        out_channels = hidden_channels if concat else self.head_channels
        self.out_lin = nn.Linear(out_channels, hidden_channels)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.node_lin.weight)
        nn.init.xavier_uniform_(self.edge_lin.weight)
        nn.init.xavier_uniform_(self.att)
        nn.init.xavier_uniform_(self.out_lin.weight)
        nn.init.zeros_(self.out_lin.bias)

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor,
        edge_direction: Tensor | None = None,
        distance_attr: Tensor | None = None,
    ) -> Tensor:
        if edge_attr is None:
            raise ValueError("NeighborDistanceAttentionConv requires edge_attr")
        if edge_attr.size(-1) != self.edge_channels:
            raise ValueError(
                f"edge_attr has {edge_attr.size(-1)} channels, expected "
                f"{self.edge_channels}"
            )

        num_edges = edge_index.size(1)
        if edge_direction is None:
            edge_direction = x.new_ones(num_edges)
        else:
            edge_direction = edge_direction.to(device=x.device, dtype=x.dtype)

        x_proj = self.node_lin(x).view(-1, self.heads, self.head_channels)
        edge_proj = self.edge_lin(edge_attr).view(
            -1, self.heads, self.head_channels
        )
        if distance_attr is None:
            distance_attr = edge_attr
        distance_weight = self._distance_weight(distance_attr)

        out = self.propagate(
            edge_index,
            x=x_proj,
            edge_attr=edge_proj,
            edge_direction=edge_direction,
            distance_weight=distance_weight,
            size=None,
        )
        if self.concat:
            out = out.reshape(-1, self.hidden_channels)
        else:
            out = out.mean(dim=1)
        return self.out_lin(out)

    def _distance_weight(self, edge_attr: Tensor) -> Tensor:
        if edge_attr.size(-1) >= 4:
            offset = edge_attr[:, 1:4]
        elif edge_attr.size(-1) >= 2:
            offset = edge_attr[:, 1:]
        else:
            offset = edge_attr

        distance = torch.linalg.vector_norm(offset, dim=-1).clamp_min(self.eps)
        if self.distance_threshold is not None:
            threshold = edge_attr.new_tensor(self.distance_threshold)
        else:
            threshold = self._running_threshold(distance)
        weight = (distance / threshold).clamp_min(self.eps).pow(-self.beta)
        return weight.clamp(max=self.max_distance_weight)

    def _running_threshold(self, distance: Tensor) -> Tensor:
        if self.training:
            batch_median = distance.detach().median().clamp_min(self.eps)
            if bool(self.threshold_initialised):
                new_value = 0.9 * self.running_threshold + 0.1 * batch_median
            else:
                new_value = batch_median
                self.threshold_initialised.fill_(True)
            self.running_threshold.copy_(new_value)
            return batch_median
        if bool(self.threshold_initialised):
            return self.running_threshold.clamp_min(self.eps)
        return distance.detach().median().clamp_min(self.eps)

    def message(
        self,
        x_i: Tensor,
        x_j: Tensor,
        edge_attr: Tensor,
        edge_direction: Tensor,
        distance_weight: Tensor,
        index: Tensor,
        ptr: Optional[Tensor],
        size_i: Optional[int],
    ) -> Tensor:
        att_input = torch.cat([x_i, edge_attr, x_j], dim=-1)
        score = (att_input * self.att.unsqueeze(0)).sum(dim=-1)
        score = score * edge_direction.view(-1, 1)
        score = F.leaky_relu(score, negative_slope=self.negative_slope)
        score = score + distance_weight.log().view(-1, 1)

        alpha = softmax(score, index, ptr, size_i)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return (x_j + edge_attr) * alpha.unsqueeze(-1)


class NDABlock(nn.Module):


    def __init__(
        self,
        hidden_channels: int,
        edge_channels: int,
        heads: int = 4,
        dropout: float = 0.0,
        beta: float = 1.0,
        distance_threshold: float | None = None,
    ) -> None:
        super().__init__()
        self.conv = NeighborDistanceAttentionConv(
            hidden_channels=hidden_channels,
            edge_channels=edge_channels,
            heads=heads,
            beta=beta,
            distance_threshold=distance_threshold,
            dropout=dropout,
        )
        self.norm1 = nn.LayerNorm(hidden_channels)
        self.norm2 = nn.LayerNorm(hidden_channels)
        self.ffn = MLP(
            hidden_channels,
            hidden_channels * 4,
            hidden_channels,
            num_layers=2,
            dropout=dropout,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor,
        edge_direction: Tensor | None = None,
        distance_attr: Tensor | None = None,
    ) -> Tensor:
        h = self.conv(
            x,
            edge_index,
            edge_attr,
            edge_direction=edge_direction,
            distance_attr=distance_attr,
        )
        x = self.norm1(x + self.dropout(h))
        x = self.norm2(x + self.dropout(self.ffn(x)))
        return x


class GraphTransformerLayer(nn.Module):
\
\
\
\
\
\


    def __init__(
        self,
        hidden_channels: int,
        heads: int = 4,
        dropout: float = 0.0,
        mode: Literal["auto", "full", "landmark"] = "auto",
        full_attention_max_nodes: int = 4096,
        num_landmarks: int = 64,
        landmark_key_bins: int | None = None,
    ) -> None:
        super().__init__()
        if hidden_channels % heads != 0:
            raise ValueError("hidden_channels must be divisible by heads")
        if num_landmarks < 1:
            raise ValueError("num_landmarks must be >= 1")

        self.hidden_channels = hidden_channels
        self.mode = mode
        self.full_attention_max_nodes = full_attention_max_nodes
        self.num_landmarks = num_landmarks
        self.landmark_key_bins = landmark_key_bins

        self.full_attn = nn.MultiheadAttention(
            hidden_channels, heads, dropout=dropout, batch_first=True
        )
        self.full_norm1 = nn.LayerNorm(hidden_channels)
        self.full_norm2 = nn.LayerNorm(hidden_channels)
        self.full_ffn = MLP(
            hidden_channels,
            hidden_channels * 4,
            hidden_channels,
            num_layers=2,
            dropout=dropout,
        )

        self.landmark_seed = nn.Parameter(
            torch.empty(num_landmarks, hidden_channels)
        )
        self.graph_to_landmarks = nn.Linear(
            hidden_channels, num_landmarks * hidden_channels
        )
        self.token_from_node_attn = nn.MultiheadAttention(
            hidden_channels, heads, dropout=dropout, batch_first=True
        )
        self.node_from_token_attn = nn.MultiheadAttention(
            hidden_channels, heads, dropout=dropout, batch_first=True
        )
        self.token_norm1 = nn.LayerNorm(hidden_channels)
        self.token_norm2 = nn.LayerNorm(hidden_channels)
        self.token_ffn = MLP(
            hidden_channels,
            hidden_channels * 4,
            hidden_channels,
            num_layers=2,
            dropout=dropout,
        )
        self.node_norm1 = nn.LayerNorm(hidden_channels)
        self.node_norm2 = nn.LayerNorm(hidden_channels)
        self.node_ffn = MLP(
            hidden_channels,
            hidden_channels * 4,
            hidden_channels,
            num_layers=2,
            dropout=dropout,
        )
        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.landmark_seed, std=0.02)
        nn.init.xavier_uniform_(self.graph_to_landmarks.weight)
        nn.init.zeros_(self.graph_to_landmarks.bias)

    def forward(self, x: Tensor, batch: Tensor | None = None) -> Tensor:
        if batch is None:
            batch = x.new_zeros(x.size(0), dtype=torch.long)
        max_nodes = int(torch.bincount(batch).max().item())
        use_full = self.mode == "full" or (
            self.mode == "auto" and max_nodes <= self.full_attention_max_nodes
        )
        if use_full:
            if (
                self.mode == "full"
                and self.full_attention_max_nodes is not None
                and max_nodes > self.full_attention_max_nodes
            ):
                raise RuntimeError(
                    f"full graph attention requested for {max_nodes} nodes; "
                    f"increase full_attention_max_nodes or use mode='auto'"
                )
            return self._full_attention(x, batch)
        return self._landmark_attention(x, batch)

    def _full_attention(self, x: Tensor, batch: Tensor) -> Tensor:
        dense, mask = to_dense_batch(x, batch)
        attn_out, _ = self.full_attn(
            dense,
            dense,
            dense,
            key_padding_mask=~mask,
            need_weights=False,
        )
        dense = self.full_norm1(dense + self.dropout(attn_out))
        dense = self.full_norm2(dense + self.dropout(self.full_ffn(dense)))
        return dense[mask]

    def _landmark_attention(self, x: Tensor, batch: Tensor) -> Tensor:
        dense, mask = to_dense_batch(x, batch)
        graph_sum = (dense * mask.unsqueeze(-1)).sum(dim=1)
        graph_count = mask.sum(dim=1).clamp_min(1).unsqueeze(-1)
        graph_mean = graph_sum / graph_count

        batch_size = dense.size(0)
        token_bias = self.graph_to_landmarks(graph_mean).view(
            batch_size, self.num_landmarks, self.hidden_channels
        )
        tokens = self.landmark_seed.unsqueeze(0) + token_bias


        keys = dense
        key_padding = None if bool(mask.all()) else ~mask
        bins = self.landmark_key_bins
        if bins and int(dense.size(1)) > int(bins) and key_padding is None:


            n = int(dense.size(1))
            width = (n + int(bins) - 1) // int(bins)
            pad_n = width * int(bins) - n
            if pad_n:
                keys = F.pad(dense, (0, 0, 0, pad_n))
            keys = keys.view(batch_size, int(bins), width, dense.size(-1)).mean(dim=2)
        token_update, _ = self.token_from_node_attn(
            tokens,
            keys,
            keys,
            key_padding_mask=key_padding,
            need_weights=False,
        )
        tokens = self.token_norm1(tokens + self.dropout(token_update))
        tokens = self.token_norm2(tokens + self.dropout(self.token_ffn(tokens)))

        node_update, _ = self.node_from_token_attn(
            dense,
            tokens,
            tokens,
            need_weights=False,
        )
        dense = self.node_norm1(dense + self.dropout(node_update))
        dense = self.node_norm2(dense + self.dropout(self.node_ffn(dense)))
        return dense[mask]


class GraphTransformerStack(nn.Module):


    def __init__(
        self,
        num_layers: int,
        hidden_channels: int,
        heads: int = 4,
        dropout: float = 0.0,
        mode: Literal["auto", "full", "landmark"] = "auto",
        full_attention_max_nodes: int = 4096,
        num_landmarks: int = 64,
        landmark_key_bins: int | None = None,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                GraphTransformerLayer(
                    hidden_channels=hidden_channels,
                    heads=heads,
                    dropout=dropout,
                    mode=mode,
                    full_attention_max_nodes=full_attention_max_nodes,
                    num_landmarks=num_landmarks,
                    landmark_key_bins=landmark_key_bins,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, x: Tensor, batch: Tensor | None = None) -> Tensor:
        for layer in self.layers:
            x = layer(x, batch=batch)
        return x
