from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable, Literal

import numpy as np
import torch
from torch import Tensor
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch_geometric.data import Data


DEFAULT_GRAPH_DIR = Path("IRGNN/graph")
DEFAULT_RAW_DATA_DIR = Path(
    "/dataset/raw_data"
)


@dataclass(frozen=True)
class LayerGridSpec:


    indices: Tensor
    grid_shape: tuple[int, int]
    x_values: Tensor
    y_values: Tensor


@dataclass(frozen=True)
class IRDropMetricResult:


    mae: float
    f1: float
    cc: float
    mirde: float
    max_abs_error: float
    hotspot_threshold: float

    def as_dict(self, voltage_scale: float = 1.0) -> dict[str, float]:
        return {
            "mae": self.mae * voltage_scale,
            "f1": self.f1,
            "cc": self.cc,
            "mirde": self.mirde * voltage_scale,
            "max_abs_error": self.max_abs_error * voltage_scale,
            "hotspot_threshold": self.hotspot_threshold * voltage_scale,
        }


def design_from_graph_path(path: str | Path) -> str:
    return Path(path).stem


def load_npz_graph(
    path: str | Path,
    include_node_names: bool = False,
    physics: bool = False,
    physics_cache_dir: str | Path | None = None,
    physics_sweeps: tuple[int, ...] | None = None,
) -> Data:
\
\
\
\
\


    payload = np.load(path, allow_pickle=True)
    node_feat = np.nan_to_num(
        payload["node_feat"], nan=0.0, posinf=0.0, neginf=0.0
    )
    edge_attr = np.nan_to_num(
        payload["edge_attr"], nan=0.0, posinf=0.0, neginf=0.0
    )
    data = Data(
        x=torch.from_numpy(node_feat).float(),
        edge_index=torch.from_numpy(payload["edge_index"]).long(),
        edge_attr=torch.from_numpy(edge_attr).float(),
    )
    data.design = str(np.asarray(payload["design"]).item())
    data.node_feature_names = [str(v) for v in payload["node_feature_names"]]
    data.edge_feature_names = [str(v) for v in payload["edge_feature_names"]]
    data.raw_resistor_edge_count = int(np.asarray(payload["raw_resistor_edge_count"]))
    data.skipped_edge_count = int(np.asarray(payload["skipped_edge_count"]))
    if include_node_names:
        data.node_names = [str(v) for v in payload["node_names"]]
    if physics:
        attach_physics(
            data, path=path, cache_dir=physics_cache_dir, sweeps=physics_sweeps
        )
    return data


def attach_physics(
    data: Data,
    path: str | Path | None = None,
    cache_dir: str | Path | None = None,
    sweeps: tuple[int, ...] | None = None,
    coarse_bins: tuple[int, ...] | None = None,
) -> Data:


    from .physics import (
        DEFAULT_COARSE_BINS,
        DEFAULT_SWEEPS,
        build_physics_system,
        num_physics_features,
        physics_node_features,
    )

    sweeps = tuple(sweeps) if sweeps else DEFAULT_SWEEPS
    coarse_bins = tuple(coarse_bins) if coarse_bins else DEFAULT_COARSE_BINS
    system = build_physics_system(
        data.x, data.edge_index, data.edge_attr, data.raw_resistor_edge_count
    )
    features = None
    cache_path = None
    if cache_dir is not None and path is not None:
        tag = "s" + "-".join(str(s) for s in sweeps) + "_b" + "-".join(
            str(b) for b in coarse_bins
        )
        cache_path = Path(cache_dir) / tag / f"{Path(path).stem}.npy"
        if cache_path.exists():
            cached = np.load(cache_path)
            if cached.shape == (
                int(data.x.size(0)),
                num_physics_features(sweeps, coarse_bins),
            ):
                features = torch.from_numpy(cached).to(data.x.dtype)
    if features is None:
        features = physics_node_features(
            system, data.x, sweeps=sweeps, coarse_bins=coarse_bins
        )
        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(cache_path, features.numpy())

    data.base_x = data.x
    data.x = torch.cat([data.x, features], dim=-1)
    data.physics = system
    data.physics_sweeps = sweeps
    data.num_physics_features = int(features.size(-1))
    return data


def list_graph_paths(
    graph_dir: str | Path = DEFAULT_GRAPH_DIR,
    designs: Iterable[str] | None = None,
    include_skipped: bool = True,
) -> list[Path]:
    graph_dir = Path(graph_dir)
    wanted = set(designs) if designs is not None else None
    manifest_path = graph_dir / "manifest.json"

    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        paths: list[Path] = []
        for item in manifest.get("graphs", []):
            design = item.get("design")
            if wanted is not None and design not in wanted:
                continue
            path = Path(item["path"])
            if path.exists():
                paths.append(path)
        return paths

    paths = sorted(graph_dir.glob("*.npz"))
    if wanted is not None:
        paths = [p for p in paths if p.stem in wanted]
    return paths


class ICCADGraphDataset(Dataset):
\
\
\
\
\


    def __init__(
        self,
        graph_dir: str | Path = DEFAULT_GRAPH_DIR,
        raw_data_dir: str | Path = DEFAULT_RAW_DATA_DIR,
        designs: Iterable[str] | None = None,
        load_target_grid: bool = False,
        include_node_names: bool = False,
    ) -> None:
        self.graph_paths = list_graph_paths(graph_dir, designs=designs)
        self.raw_data_dir = Path(raw_data_dir)
        self.load_target_grid = load_target_grid
        self.include_node_names = include_node_names

    def __len__(self) -> int:
        return len(self.graph_paths)

    def __getitem__(self, index: int) -> Data:
        data = load_npz_graph(
            self.graph_paths[index], include_node_names=self.include_node_names
        )
        if self.load_target_grid:
            target = load_iccad_ir_drop_grid(data.design, self.raw_data_dir)
            data.target_grid = torch.from_numpy(target).float()
            data.target_shape = torch.tensor(target.shape, dtype=torch.long)
        return data


def load_iccad_ir_drop_grid(
    design: str,
    raw_data_dir: str | Path = DEFAULT_RAW_DATA_DIR,
) -> np.ndarray:


    path = Path(raw_data_dir) / f"{design}_ir_drop.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    return np.loadtxt(path, delimiter=",").astype(np.float32)


def compute_layer_grid_spec(
    node_feat: Tensor,
    layer: float = 1.0,
    layer_col: int = 2,
    x_col: int = 0,
    y_col: int = 1,
    atol: float = 1.0e-4,
) -> LayerGridSpec:


    feat = node_feat.detach()
    mask = torch.isclose(
        feat[:, layer_col],
        feat.new_tensor(layer),
        atol=atol,
        rtol=0.0,
    )
    indices = torch.nonzero(mask, as_tuple=False).flatten()
    if indices.numel() == 0:
        raise ValueError(f"no nodes found on layer {layer}")

    coords = feat[indices][:, [x_col, y_col]]
    x_values = torch.unique(coords[:, 0], sorted=True)
    y_values = torch.unique(coords[:, 1], sorted=True)
    grid_shape = (int(y_values.numel()), int(x_values.numel()))
    if grid_shape[0] * grid_shape[1] != int(indices.numel()):
        raise ValueError(
            "layer nodes do not form a dense rectangular coordinate grid: "
            f"{indices.numel()} nodes vs shape {grid_shape}"
        )

    x_first = torch.argsort(coords[:, 0], stable=True)
    indices = indices[x_first]
    coords = coords[x_first]
    y_then_x = torch.argsort(coords[:, 1], stable=True)
    sorted_indices = indices[y_then_x]
    return LayerGridSpec(
        indices=sorted_indices,
        grid_shape=grid_shape,
        x_values=x_values,
        y_values=y_values,
    )


def node_values_to_layer_grid(
    values: Tensor,
    node_feat: Tensor,
    layer: float = 1.0,
    spec: LayerGridSpec | None = None,
) -> Tensor:


    if values.dim() == 2 and values.size(-1) == 1:
        values = values.squeeze(-1)
    if spec is None:
        spec = compute_layer_grid_spec(node_feat, layer=layer)
    return values[spec.indices].reshape(spec.grid_shape)


def resize_grid(
    grid: Tensor,
    target_shape: tuple[int, int] | torch.Size,
    mode: Literal["bilinear", "bicubic", "nearest"] = "bilinear",
) -> Tensor:


    align_corners = False if mode in {"bilinear", "bicubic"} else None
    grid_4d = grid
    squeeze_dims = 0
    if grid_4d.dim() == 2:
        grid_4d = grid_4d.unsqueeze(0).unsqueeze(0)
        squeeze_dims = 2
    elif grid_4d.dim() == 3:
        grid_4d = grid_4d.unsqueeze(1)
        squeeze_dims = 1
    elif grid_4d.dim() != 4:
        raise ValueError("grid must have shape [H,W], [B,H,W], or [B,C,H,W]")

    out = F.interpolate(
        grid_4d.float(),
        size=tuple(int(v) for v in target_shape),
        mode=mode,
        align_corners=align_corners,
    )
    if squeeze_dims == 2:
        return out[0, 0]
    if squeeze_dims == 1:
        return out[:, 0]
    return out


CSV_GRID_PITCH = 2000.0
\
\
\
\
\
\
\


def _fractional_index(values: Tensor, query: Tensor) -> Tensor:
\
\
\
\


    values = values.to(dtype=torch.float64)
    query = query.to(dtype=torch.float64)
    n = int(values.numel())
    if n == 1:
        return torch.zeros_like(query)
    right = torch.searchsorted(values, query.contiguous()).clamp(1, n - 1)
    left = right - 1
    span = (values[right] - values[left]).clamp_min(1.0e-12)
    frac = (query - values[left]) / span
    return (left.to(query.dtype) + frac).clamp(0.0, float(n - 1))


def _sample_layer_grid(
    grid: Tensor,
    spec: LayerGridSpec,
    target_shape: tuple[int, int] | torch.Size,
    pitch: float,
    mode: Literal["bilinear", "bicubic", "nearest"],
) -> Tensor:


    rows, cols = (int(v) for v in target_shape)
    device = grid.device
    x_query = torch.arange(rows, dtype=torch.float64, device=device) * pitch
    y_query = torch.arange(cols, dtype=torch.float64, device=device) * pitch
    x_idx = _fractional_index(spec.x_values.to(device), x_query)
    y_idx = _fractional_index(spec.y_values.to(device), y_query)

    def to_norm(idx: Tensor, size: int) -> Tensor:
        if size == 1:
            return torch.zeros_like(idx)
        return 2.0 * idx / (size - 1) - 1.0


    u = to_norm(x_idx, int(spec.x_values.numel()))
    v = to_norm(y_idx, int(spec.y_values.numel()))
    sample = torch.stack(
        [u[:, None].expand(rows, cols), v[None, :].expand(rows, cols)], dim=-1
    )[None].to(grid.dtype)
    out = F.grid_sample(
        grid[None, None].float(),
        sample.float(),
        mode="nearest" if mode == "nearest" else "bilinear",
        padding_mode="border",
        align_corners=True,
    )
    return out[0, 0].to(grid.dtype)


def iccad_prediction_grid(
    pred_node: Tensor,
    node_feat: Tensor,
    target_shape: tuple[int, int] | torch.Size | None = None,
    layer: float = 1.0,
    resize_mode: Literal["bilinear", "bicubic", "nearest"] = "bilinear",
    spec: LayerGridSpec | None = None,
    align: Literal["physical", "stretch"] = "physical",
    pitch: float = CSV_GRID_PITCH,
) -> Tensor:


    if spec is None:
        spec = compute_layer_grid_spec(node_feat, layer=layer)
    pred_grid = node_values_to_layer_grid(
        pred_node, node_feat=node_feat, layer=layer, spec=spec
    )
    if target_shape is None:


        return pred_grid.transpose(-2, -1)
    if align == "physical":
        return _sample_layer_grid(pred_grid, spec, target_shape, pitch, resize_mode)
    return resize_grid(
        pred_grid.transpose(-2, -1), target_shape=target_shape, mode=resize_mode
    )


def iccad_target_on_nodes(
    target_grid: Tensor,
    node_feat: Tensor,
    layer: float = 1.0,
    spec: LayerGridSpec | None = None,
    pitch: float = CSV_GRID_PITCH,
) -> tuple[Tensor, Tensor]:
\
\
\
\
\


    if spec is None:
        spec = compute_layer_grid_spec(node_feat, layer=layer)
    grid = target_grid.to(node_feat.device)
    rows, cols = int(grid.size(-2)), int(grid.size(-1))
    idx = spec.indices.to(node_feat.device)
    x = node_feat[idx, 0].to(torch.float64) / pitch
    y = node_feat[idx, 1].to(torch.float64) / pitch

    def to_norm(pos: Tensor, size: int) -> Tensor:
        if size == 1:
            return torch.zeros_like(pos)
        return (2.0 * pos.clamp(0.0, float(size - 1)) / (size - 1) - 1.0)

    sample = torch.stack([to_norm(y, cols), to_norm(x, rows)], dim=-1)
    sample = sample[None, :, None, :].float()
    values = F.grid_sample(
        grid[None, None].float(),
        sample,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )[0, 0, :, 0]

    out = node_feat.new_full((node_feat.size(0),), float("nan"))
    out[idx] = values.to(out.dtype)
    mask = torch.zeros_like(out, dtype=torch.bool)
    mask[idx] = True
    return out, mask


def _elementwise_loss(
    pred: Tensor,
    target: Tensor,
    loss: Literal["l1", "mse", "smooth_l1"],
) -> Tensor:
    if loss == "l1":
        return F.l1_loss(pred, target, reduction="none")
    if loss == "mse":
        return F.mse_loss(pred, target, reduction="none")
    if loss == "smooth_l1":
        return F.smooth_l1_loss(pred, target, reduction="none")
    raise ValueError(f"unknown loss: {loss}")


def _aggregate_map_loss(
    pred_grid: Tensor,
    target_grid: Tensor,
    loss: Literal["l1", "mse", "smooth_l1"],
    hotspot_ratio: float,
    hotspot_weight: float,
    max_weight: float,
    relative: bool,
    eps: float,
) -> Tensor:
    finite = torch.isfinite(target_grid) & torch.isfinite(pred_grid)
    if not bool(finite.any()):
        raise ValueError("no finite prediction/target values")

    errors = _elementwise_loss(pred_grid, target_grid, loss)
    total = errors[finite].mean()
    finite_target = target_grid[finite]

    if hotspot_weight > 0.0 or max_weight > 0.0:
        peak = finite_target.max()

        if hotspot_weight > 0.0:
            hot = finite & (target_grid >= hotspot_ratio * peak)
            if bool(hot.any()):
                total = total + hotspot_weight * errors[hot].mean()

        if max_weight > 0.0:
            total = total + max_weight * (pred_grid[finite].max() - peak).abs()

    if relative:
        scale = finite_target.abs().mean().clamp_min(eps)
        if loss == "mse":
            scale = scale * scale
        total = total / scale

    return total


def iccad_grid_loss(
    pred_node: Tensor,
    node_feat: Tensor,
    target_grid: Tensor | np.ndarray,
    layer: float = 1.0,
    loss: Literal["l1", "mse", "smooth_l1"] = "l1",
    resize_mode: Literal["bilinear", "bicubic", "nearest"] = "bilinear",
    hotspot_ratio: float = 0.9,
    hotspot_weight: float = 0.0,
    max_weight: float = 0.0,
    relative: bool = False,
    align: Literal["csv", "native"] = "csv",
    eps: float = 1.0e-12,
) -> Tensor:
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
\
\
\
\
\
\


    target = torch.as_tensor(target_grid, device=pred_node.device, dtype=pred_node.dtype)
    if align == "csv":
        pred_grid = iccad_prediction_grid(
            pred_node,
            node_feat=node_feat,
            target_shape=target.shape[-2:],
            layer=layer,
            resize_mode=resize_mode,
        )
        target_grid_t = target
    elif align == "native":


        node_target, node_mask = iccad_target_on_nodes(target, node_feat, layer=layer)
        pred_grid = torch.where(node_mask, pred_node, pred_node.new_tensor(float("nan")))
        target_grid_t = node_target
    else:
        raise ValueError(f"unknown align mode: {align}")

    return _aggregate_map_loss(
        pred_grid,
        target_grid_t,
        loss=loss,
        hotspot_ratio=hotspot_ratio,
        hotspot_weight=hotspot_weight,
        max_weight=max_weight,
        relative=relative,
        eps=eps,
    )


DEFAULT_POWERRUSH_IR_DIR = Path(
    "/PowerRush_final/iccad_dataset/ir_drop"
)


def load_powerrush_node_ir_drop(
    design: str,
    ir_dir: str | Path = DEFAULT_POWERRUSH_IR_DIR,
) -> dict[str, float]:


    path = Path(ir_dir) / f"{design}.ir_drop.txt"
    if not path.exists():
        raise FileNotFoundError(path)
    values: dict[str, float] = {}
    with path.open("r", encoding="utf-8", errors="replace") as fp:
        next(fp, None)
        for line in fp:
            fields = line.split()
            if len(fields) < 3:
                continue
            try:
                values[fields[0]] = float(fields[2])
            except ValueError:
                continue
    if not values:
        raise ValueError(f"no IR-drop values found in {path}")
    return values


def iccad_node_loss(
    pred_node: Tensor,
    node_names: list[str] | np.ndarray,
    target_by_name: dict[str, float],
    loss: Literal["l1", "mse", "smooth_l1"] = "l1",
    layer_mask: Tensor | None = None,
    hotspot_ratio: float = 0.9,
    hotspot_weight: float = 0.0,
    max_weight: float = 0.0,
    relative: bool = False,
    target_tensor: Tensor | None = None,
    valid_mask: Tensor | None = None,
    eps: float = 1.0e-12,
) -> Tensor:
\
\
\
\


    if pred_node.dim() == 2 and pred_node.size(-1) == 1:
        pred_node = pred_node.squeeze(-1)

    if target_tensor is None or valid_mask is None:
        names = [str(v) for v in node_names]
        if len(names) != int(pred_node.size(0)):
            raise ValueError(
                f"node_names length {len(names)} != pred_node length {pred_node.size(0)}"
            )
        target_np = np.fromiter(
            (
                target_by_name[name]
                if name in target_by_name and np.isfinite(target_by_name[name])
                else np.nan
                for name in names
            ),
            dtype=np.float32,
            count=len(names),
        )
        target_tensor = torch.from_numpy(target_np).to(
            device=pred_node.device, dtype=pred_node.dtype
        )
        valid_mask = torch.isfinite(target_tensor)
    else:
        target_tensor = target_tensor.to(device=pred_node.device, dtype=pred_node.dtype)
        valid_mask = valid_mask.to(device=pred_node.device, dtype=torch.bool)

    valid = valid_mask
    if layer_mask is not None:
        valid = valid & layer_mask.to(device=pred_node.device, dtype=torch.bool)

    if not bool(valid.any()):
        raise ValueError("no overlapping nodes between prediction and node targets")


    nan = pred_node.new_tensor(float("nan"))
    target = torch.where(valid, target_tensor, nan)
    pred = torch.where(valid, pred_node, nan)
    return _aggregate_map_loss(
        pred,
        target,
        loss=loss,
        hotspot_ratio=hotspot_ratio,
        hotspot_weight=hotspot_weight,
        max_weight=max_weight,
        relative=relative,
        eps=eps,
    )


def iccad_train_loss(
    pred_node: Tensor,
    node_feat: Tensor,
    target_grid: Tensor | np.ndarray,
    *,
    node_names: list[str] | np.ndarray | None = None,
    node_target_by_name: dict[str, float] | None = None,
    node_target_tensor: Tensor | None = None,
    node_valid_mask: Tensor | None = None,
    supervise: Literal["csv", "native", "nodes", "native_nodes"] = "native",
    layer: float = 1.0,
    loss: Literal["l1", "mse", "smooth_l1"] = "l1",
    resize_mode: Literal["bilinear", "bicubic", "nearest"] = "bilinear",
    hotspot_ratio: float = 0.9,
    hotspot_weight: float = 0.0,
    max_weight: float = 0.0,
    relative: bool = False,
    node_loss_weight: float = 1.0,
    grid_loss_weight: float = 1.0,
    node_layers: Literal["m1", "all"] = "all",
    kcl_weight: float = 0.0,
    edge_index: Tensor | None = None,
    resistance: Tensor | None = None,
    eps: float = 1.0e-12,
) -> Tensor:


    terms: list[Tensor] = []

    if supervise in {"csv", "native", "native_nodes"}:
        align: Literal["csv", "native"] = "csv" if supervise == "csv" else "native"
        terms.append(
            grid_loss_weight
            * iccad_grid_loss(
                pred_node,
                node_feat,
                target_grid,
                layer=layer,
                loss=loss,
                resize_mode=resize_mode,
                hotspot_ratio=hotspot_ratio,
                hotspot_weight=hotspot_weight,
                max_weight=max_weight,
                relative=relative,
                align=align,
                eps=eps,
            )
        )

    if supervise in {"nodes", "native_nodes"}:
        if node_target_tensor is None and (
            node_names is None or node_target_by_name is None
        ):
            raise ValueError("node supervision requires node targets")
        layer_mask = None
        if node_layers == "m1":
            layer_mask = torch.isclose(
                node_feat[:, 2],
                node_feat.new_tensor(layer),
                atol=1.0e-4,
                rtol=0.0,
            )
        terms.append(
            node_loss_weight
            * iccad_node_loss(
                pred_node,
                node_names if node_names is not None else [],
                node_target_by_name or {},
                loss=loss,
                layer_mask=layer_mask,
                hotspot_ratio=hotspot_ratio,
                hotspot_weight=hotspot_weight,
                max_weight=max_weight,
                relative=relative,
                target_tensor=node_target_tensor,
                valid_mask=node_valid_mask,
                eps=eps,
            )
        )

    if not terms:
        raise ValueError(f"unknown supervise mode: {supervise}")
    total = terms[0]
    for term in terms[1:]:
        total = total + term

    if kcl_weight > 0.0:
        if edge_index is None or resistance is None:
            raise ValueError("kcl_weight>0 requires edge_index and resistance")
        kcl_target = node_target_tensor
        kcl_mask = node_valid_mask
        if kcl_target is not None:
            kcl_target = kcl_target.to(device=pred_node.device, dtype=pred_node.dtype)
            padish = kcl_target.abs() <= 1.0e-12
            if kcl_mask is None:
                kcl_mask = ~padish
            else:
                kcl_mask = kcl_mask.to(device=pred_node.device, dtype=torch.bool) & ~padish
        total = total + kcl_weight * graph_kcl_loss(
            pred_node,
            edge_index,
            resistance,
            target_drop=kcl_target,
            valid_mask=kcl_mask,
            eps=eps,
        )
    return total


def ir_drop_kcl_residual(
    drop: Tensor,
    edge_index: Tensor,
    resistance: Tensor,
    eps: float = 1.0e-12,
) -> Tensor:
\
\
\
\


    if drop.dim() == 2 and drop.size(-1) == 1:
        drop = drop.squeeze(-1)
    src = edge_index[0]
    dst = edge_index[1]
    cond = resistance.clamp_min(eps).reciprocal()
    flow = (drop[dst] - drop[src]) * cond
    residual = drop.new_zeros(drop.size(0))
    residual.scatter_add_(0, src, flow)
    residual.scatter_add_(0, dst, -flow)
    return residual


def graph_kcl_loss(
    pred_drop: Tensor,
    edge_index: Tensor,
    resistance: Tensor,
    target_drop: Tensor | None = None,
    valid_mask: Tensor | None = None,
    eps: float = 1.0e-12,
) -> Tensor:


    pred_r = ir_drop_kcl_residual(pred_drop, edge_index, resistance, eps=eps)
    if target_drop is None:
        residual = pred_r
    else:
        target_r = ir_drop_kcl_residual(target_drop, edge_index, resistance, eps=eps)
        residual = pred_r - target_r
    if valid_mask is not None:
        residual = residual[valid_mask]
        if residual.numel() == 0:
            return pred_drop.new_zeros(())
    return residual.abs().mean()


def compute_ir_drop_metrics(
    pred_grid: Tensor | np.ndarray,
    target_grid: Tensor | np.ndarray,
    hotspot_ratio: float = 0.9,
    f1_threshold: Literal["target", "separate"] = "target",
    eps: float = 1.0e-12,
) -> IRDropMetricResult:
\
\
\
\
\


    pred = torch.as_tensor(pred_grid).float().flatten()
    target = torch.as_tensor(target_grid).float().flatten()
    finite = torch.isfinite(pred) & torch.isfinite(target)
    pred = pred[finite]
    target = target[finite]
    if pred.numel() == 0:
        raise ValueError("no finite prediction/target values")

    abs_err = (pred - target).abs()
    mae = float(abs_err.mean().item())

    pred_centered = pred - pred.mean()
    target_centered = target - target.mean()
    denom = pred_centered.norm() * target_centered.norm()
    cc = float((pred_centered * target_centered).sum().div(denom + eps).item())

    target_threshold = hotspot_ratio * target.max()
    if f1_threshold == "target":
        pred_threshold = target_threshold
    elif f1_threshold == "separate":
        pred_threshold = hotspot_ratio * pred.max()
    else:
        raise ValueError(f"unknown f1_threshold: {f1_threshold}")

    pred_hot = pred >= pred_threshold
    target_hot = target >= target_threshold
    tp = (pred_hot & target_hot).sum().float()
    fp = (pred_hot & ~target_hot).sum().float()
    fn = (~pred_hot & target_hot).sum().float()
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = float((2.0 * precision * recall / (precision + recall + eps)).item())

    mirde = float((pred.max() - target.max()).abs().item())
    max_abs_error = float(abs_err.max().item())
    return IRDropMetricResult(
        mae=mae,
        f1=f1,
        cc=cc,
        mirde=mirde,
        max_abs_error=max_abs_error,
        hotspot_threshold=float(target_threshold.item()),
    )


def evaluate_iccad_node_prediction(
    pred_node: Tensor,
    node_feat: Tensor,
    design: str,
    raw_data_dir: str | Path = DEFAULT_RAW_DATA_DIR,
    layer: float = 1.0,
    hotspot_ratio: float = 0.9,
    f1_threshold: Literal["target", "separate"] = "target",
    resize_mode: Literal["bilinear", "bicubic", "nearest"] = "bilinear",
) -> tuple[IRDropMetricResult, Tensor, Tensor]:


    target_np = load_iccad_ir_drop_grid(design, raw_data_dir=raw_data_dir)
    target = torch.as_tensor(target_np, device=pred_node.device, dtype=pred_node.dtype)
    pred_grid = iccad_prediction_grid(
        pred_node,
        node_feat=node_feat,
        target_shape=target.shape,
        layer=layer,
        resize_mode=resize_mode,
    )
    metrics = compute_ir_drop_metrics(
        pred_grid,
        target,
        hotspot_ratio=hotspot_ratio,
        f1_threshold=f1_threshold,
    )
    return metrics, pred_grid, target
