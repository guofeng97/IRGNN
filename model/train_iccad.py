from __future__ import annotations

import argparse
import contextlib
from pathlib import Path
import sys
import time

import numpy as np
import torch

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from model.iccad import (
        DEFAULT_GRAPH_DIR,
        DEFAULT_POWERRUSH_IR_DIR,
        DEFAULT_RAW_DATA_DIR,
        compute_ir_drop_metrics,
        evaluate_iccad_node_prediction,
        iccad_prediction_grid,
        iccad_train_loss,
        list_graph_paths,
        load_iccad_ir_drop_grid,
        load_powerrush_node_ir_drop,
        load_npz_graph,
    )
    from model.irgnn import IRGNN
else:
    from .iccad import (
        DEFAULT_GRAPH_DIR,
        DEFAULT_POWERRUSH_IR_DIR,
        DEFAULT_RAW_DATA_DIR,
        compute_ir_drop_metrics,
        evaluate_iccad_node_prediction,
        iccad_prediction_grid,
        iccad_train_loss,
        list_graph_paths,
        load_iccad_ir_drop_grid,
        load_powerrush_node_ir_drop,
        load_npz_graph,
    )
    from .irgnn import IRGNN


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph-dir", type=Path, default=DEFAULT_GRAPH_DIR)
    parser.add_argument(
        "--val-graph-dir",
        type=Path,
        default=None,
        help="Validation graph directory. Defaults to --graph-dir.",
    )
    parser.add_argument("--raw-data-dir", type=Path, default=DEFAULT_RAW_DATA_DIR)
    parser.add_argument(
        "--train-design",
        action="append",
        default=[],
        help="Design name for training. Defaults to every graph in --graph-dir.",
    )
    parser.add_argument(
        "--val-design",
        action="append",
        default=[],
        help="Design name for validation. Defaults to every graph in --val-graph-dir.",
    )
    parser.add_argument("--limit-train", type=int, default=None)
    parser.add_argument("--limit-val", type=int, default=None)
    parser.add_argument(
        "--limit-mode",
        choices=["stride", "head"],
        default="stride",
        help=(
            "How --limit-* truncates the sorted design list. head keeps the "
            "first N, which silently dropped every realdesign when N matched "
            "the fakedesign count; stride samples evenly across the list."
        ),
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--hidden-channels", type=int, default=128)
    parser.add_argument("--nda-layers", type=int, default=2)
    parser.add_argument("--gt-layers", type=int, default=1)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument(
        "--gt-mode",
        choices=["auto", "full", "landmark"],
        default="auto",
        help="Use landmark mode for large ICCAD graphs.",
    )
    parser.add_argument("--full-attention-max-nodes", type=int, default=4096)
    parser.add_argument("--num-landmarks", type=int, default=64)
    parser.add_argument(
        "--loss", choices=["l1", "mse", "smooth_l1"], default="l1"
    )
    parser.add_argument(
        "--hotspot-weight",
        type=float,
        default=0.0,
        help="Weight of the extra loss term restricted to hotspot pixels.",
    )
    parser.add_argument(
        "--hotspot-ratio",
        type=float,
        default=0.9,
        help="Hotspot pixels are those above this fraction of max(target).",
    )
    parser.add_argument(
        "--bilinear-hinge-weight",
        type=float,
        default=0.0,
        help="Weight of max(0, threshold-pred) on the bilinear eval grid, only where the target is already a hotspot.",
    )
    parser.add_argument(
        "--contour-band",
        choices=["none", "gauss", "shoulder"],
        default="none",
        help="Hotspot-contour residual. shoulder edits 90-97%% of the predicted peak; gauss edits a narrow ring at 90%%.",
    )
    parser.add_argument("--contour-cap", type=float, default=0.06)
    parser.add_argument("--contour-loss-weight", type=float, default=10.0)
    parser.add_argument(
        "--max-weight",
        type=float,
        default=0.0,
        help="Weight of the peak error term abs(max(pred) - max(target)).",
    )
    parser.add_argument(
        "--relative-loss",
        action="store_true",
        help="Divide each design's loss by its mean target IR drop.",
    )
    parser.add_argument(
        "--supervise",
        choices=["csv", "native", "nodes", "native_nodes"],
        default="native_nodes",
        help=(
            "csv: legacy resize-pred-to-CSV loss; "
            "native: compare on the M1 node grid (downsample CSV); "
            "nodes: PowerRush converged node IR drop; "
            "native_nodes: native grid + PowerRush nodes (recommended)."
        ),
    )
    parser.add_argument(
        "--powerrush-ir-dir",
        type=Path,
        default=DEFAULT_POWERRUSH_IR_DIR,
        help="Directory of converged {design}.ir_drop.txt files for node supervision.",
    )
    parser.add_argument(
        "--node-loss-weight",
        type=float,
        default=1.0,
        help="Weight of the PowerRush node-level term.",
    )
    parser.add_argument(
        "--grid-loss-weight",
        type=float,
        default=1.0,
        help="Weight of the LaRED grid term (csv/native).",
    )
    parser.add_argument(
        "--node-layers",
        choices=["m1", "all"],
        default="all",
        help="Which graph nodes receive PowerRush supervision.",
    )
    parser.add_argument(
        "--lr-schedule",
        choices=["none", "cosine"],
        default="none",
        help="Decay the learning rate to --lr-min over the remaining epochs.",
    )
    parser.add_argument("--lr-min", type=float, default=1.0e-5)
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument(
        "--skip-val",
        action="store_true",
        help="Train and save weights without running validation.",
    )
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("irgnn_runs"))
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument(
        "--compose",
        choices=["parallel", "serial"],
        default="parallel",
        help="serial: Graph Transformer reads NDA output. parallel: original concat.",
    )
    parser.add_argument(
        "--no-residual-zero-init",
        action="store_true",
        help="Leave the residual readout randomly initialized, as in the original IRGNN.",
    )
    parser.add_argument(
        "--split-layer-edges",
        action="store_true",
        help="Separate NDA stacks for same-layer metal vs via/inter-layer edges.",
    )
    parser.add_argument(
        "--peak-head",
        action="store_true",
        help="Extra zero-init residual head for peak/hotspot corrections.",
    )
    parser.add_argument(
        "--kcl-weight",
        type=float,
        default=0.0,
        help="Weight of the graph Kirchhoff residual loss vs PowerRush nodes.",
    )
    parser.add_argument(
        "--target-scale",
        choices=["none", "drop", "residual"],
        default="none",
        help=(
            "Per-design loss normalisation. residual divides by the design's "
            "own mean |converged - rough|, so easy fake designs stop "
            "dominating the gradient (their error is ~4x smaller)."
        ),
    )
    parser.add_argument(
        "--predict-direct",
        action="store_true",
        help="Network output is the IR drop itself, not a correction added to the rough solution.",
    )
    parser.add_argument(
        "--residual-scale-mode",
        choices=["const", "design"],
        default="const",
        help="design ties the readout's volt scale to each design's IR-drop amplitude.",
    )
    parser.add_argument("--residual-scale-ratio", type=float, default=0.05)
    parser.add_argument(
        "--physics-features",
        action="store_true",
        help="Append residual-derived node features (needs the PG operator).",
    )
    parser.add_argument(
        "--physics-sweeps",
        default="1,4,16,64",
        help="Weighted-Jacobi sweep counts used for the residual features.",
    )
    parser.add_argument("--physics-cache-dir", type=Path, default=None)
    parser.add_argument(
        "--coarse-head",
        action="store_true",
        help="Add the coarse spatial field head for the smooth part of the error.",
    )
    parser.add_argument("--coarse-bins", type=int, default=32)
    parser.add_argument(
        "--unroll-steps",
        type=int,
        default=1,
        help="Unrolled refinement steps; the residual is recomputed each step.",
    )
    parser.add_argument(
        "--landmark-key-bins",
        type=int,
        default=0,
        help=(
            "If >0, mean-pool landmark-attention keys into this many chunks. "
            "2048 matches the P3 checkpoint's MAE/F1 and skips the full-node matmul. "
            "0 keeps exact attention."
        ),
    )
    parser.add_argument(
        "--infer-dtype",
        choices=["fp32", "fp16", "bf16"],
        default="fp32",
        help="Autocast dtype for evaluation only. fp16 keeps the P3 metrics.",
    )
    return parser.parse_args()


def select_paths(
    graph_dir: Path,
    raw_data_dir: Path,
    explicit_designs: list[str],
    limit: int | None,
    limit_mode: str = "stride",
) -> list[Path]:
    if explicit_designs:
        paths = list_graph_paths(graph_dir, designs=explicit_designs)
    else:
        paths = sorted(graph_dir.glob("*.npz"))
    paths = [p for p in paths if (raw_data_dir / f"{p.stem}_ir_drop.csv").exists()]
    paths = sorted(paths, key=lambda p: p.stem)
    if limit is not None and limit < len(paths):
        if limit_mode == "head":
            paths = paths[:limit]
        else:
            picks = np.linspace(0, len(paths) - 1, limit).round().astype(int)
            paths = [paths[i] for i in sorted(set(int(i) for i in picks))]
    return paths


def describe_designs(label: str, paths: list[Path]) -> None:
    families: dict[str, int] = {}
    for path in paths:
        family = path.stem.rstrip("0123456789") or path.stem
        families[family] = families.get(family, 0) + 1
    summary = ", ".join(f"{name}={count}" for name, count in sorted(families.items()))
    print(f"{label}: {len(paths)} graphs ({summary})")


def physics_sweeps(args: argparse.Namespace) -> tuple[int, ...] | None:
    if not args.physics_features:
        return None
    return tuple(int(v) for v in str(args.physics_sweeps).split(",") if v.strip())


def build_model(args: argparse.Namespace) -> IRGNN:
    node_in_channels = 7
    if args.physics_features:
        from model.physics import num_physics_features

        node_in_channels += num_physics_features(sweeps=physics_sweeps(args))
    return IRGNN(
        node_in_channels=node_in_channels,
        hidden_channels=args.hidden_channels,
        nda_layers=args.nda_layers,
        gt_layers=args.gt_layers,
        heads=args.heads,
        dropout=args.dropout,
        gt_mode=args.gt_mode,
        full_attention_max_nodes=args.full_attention_max_nodes,
        num_landmarks=args.num_landmarks,
        compose=args.compose,
        split_layer_edges=args.split_layer_edges,
        peak_head=args.peak_head,
        residual_to_numerical=not args.predict_direct,
        residual_scale_mode=args.residual_scale_mode,
        residual_scale_ratio=args.residual_scale_ratio,
        coarse_head=args.coarse_head,
        coarse_bins=args.coarse_bins,
        unroll_steps=args.unroll_steps,
        landmark_key_bins=args.landmark_key_bins or None,
        residual_zero_init=not args.no_residual_zero_init,
    )


def load_checkpoint(model: IRGNN, path: Path, device: torch.device) -> int:
    payload = torch.load(path, map_location=device)
    state_dict = payload["model"] if isinstance(payload, dict) and "model" in payload else payload
    model.load_state_dict(state_dict)
    if isinstance(payload, dict):
        return int(payload.get("epoch", 0))
    return 0


def load_best_mae(path: Path, device: torch.device) -> float:
    if not path.exists():
        return float("inf")
    payload = torch.load(path, map_location=device)
    if not isinstance(payload, dict):
        return float("inf")
    metrics = payload.get("metrics")
    if not isinstance(metrics, dict):
        return float("inf")
    return float(metrics.get("mae_x1e4", float("inf")))


def serializable_args(args: argparse.Namespace) -> dict[str, object]:
    values: dict[str, object] = {}
    for key, value in vars(args).items():
        values[key] = str(value) if isinstance(value, Path) else value
    return values


def bilinear_hotspot_hinge(
    pred_node: torch.Tensor,
    node_feat: torch.Tensor,
    target_grid: torch.Tensor,
    hotspot_ratio: float,
) -> torch.Tensor:


    pred_grid = iccad_prediction_grid(
        pred_node,
        node_feat,
        target_shape=target_grid.shape,
        resize_mode="bilinear",
    )
    finite = torch.isfinite(pred_grid) & torch.isfinite(target_grid)
    if not bool(finite.any()):
        return pred_node.new_zeros(())
    threshold = target_grid[finite].max() * hotspot_ratio
    missed = finite & (target_grid >= threshold) & (pred_grid < threshold)
    if not bool(missed.any()):
        return pred_node.new_zeros(())
    return (threshold - pred_grid[missed]).mean()


def train_one_epoch(
    model: IRGNN,
    paths: list[Path],
    raw_data_dir: Path,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    loss_name: str,
    grad_clip: float,
    hotspot_ratio: float = 0.9,
    hotspot_weight: float = 0.0,
    max_weight: float = 0.0,
    relative: bool = False,
    supervise: str = "native_nodes",
    powerrush_ir_dir: Path | None = None,
    node_loss_weight: float = 1.0,
    grid_loss_weight: float = 1.0,
    node_layers: str = "all",
    kcl_weight: float = 0.0,
    node_align_cache: dict[str, tuple[torch.Tensor, torch.Tensor]] | None = None,
    physics: bool = False,
    physics_cache_dir: Path | None = None,
    physics_sweep_spec: tuple[int, ...] | None = None,
    target_scale: str = "none",
    scale_cache: dict[str, float] | None = None,
    bilinear_hinge_weight: float = 0.0,
    contour: torch.nn.Module | None = None,
    contour_loss_weight: float = 0.0,
) -> float:
    model.train()
    if contour is not None:
        contour.train()
    total = 0.0
    need_nodes = supervise in {"nodes", "native_nodes"}
    if node_align_cache is None:
        node_align_cache = {}
    if scale_cache is None:
        scale_cache = {}
    for path in paths:
        data = load_npz_graph(
            path,
            include_node_names=need_nodes,
            physics=physics,
            physics_cache_dir=physics_cache_dir,
            physics_sweeps=physics_sweep_spec,
        ).to(device)
        target = torch.from_numpy(load_iccad_ir_drop_grid(data.design, raw_data_dir)).to(device)

        node_target_tensor = None
        node_valid_mask = None
        if need_nodes:
            if data.design not in node_align_cache:
                if powerrush_ir_dir is None:
                    raise ValueError("powerrush_ir_dir is required for node supervision")
                by_name = load_powerrush_node_ir_drop(data.design, powerrush_ir_dir)
                names = [str(v) for v in data.node_names]
                values = np.fromiter(
                    (
                        by_name[name]
                        if name in by_name and np.isfinite(by_name[name])
                        else np.nan
                        for name in names
                    ),
                    dtype=np.float32,
                    count=len(names),
                )
                aligned = torch.from_numpy(values)
                node_align_cache[data.design] = (aligned, torch.isfinite(aligned))
            node_target_tensor, node_valid_mask = node_align_cache[data.design]

        if target_scale != "none" and data.design not in scale_cache:
            scale_cache[data.design] = design_loss_scale(
                data, node_target_tensor, node_valid_mask, target_scale
            )

        optimizer.zero_grad(set_to_none=True)
        out = model.forward_data(data, return_dict=model.unroll_steps > 1)
        preds = out["preds"] if isinstance(out, dict) else [out]

        def step_loss(pred: torch.Tensor) -> torch.Tensor:
            loss = iccad_train_loss(
                pred,
                data.x,
                target,
                node_target_tensor=node_target_tensor,
                node_valid_mask=node_valid_mask,
                supervise=supervise,
                loss=loss_name,
                hotspot_ratio=hotspot_ratio,
                hotspot_weight=hotspot_weight,
                max_weight=max_weight,
                relative=relative,
                node_loss_weight=node_loss_weight,
                grid_loss_weight=grid_loss_weight,
                node_layers=node_layers,
                kcl_weight=kcl_weight,
                edge_index=data.edge_index if kcl_weight > 0 else None,
                resistance=data.edge_attr[:, 0] if kcl_weight > 0 else None,
            )
            if bilinear_hinge_weight > 0.0:
                loss = loss + bilinear_hinge_weight * bilinear_hotspot_hinge(
                    pred, data.x, target, hotspot_ratio
                )
            return loss


        loss = sum(step_loss(pred) for pred in preds) / len(preds)
        if target_scale != "none":
            loss = loss / scale_cache[data.design]
        if contour is not None and contour_loss_weight > 0.0:
            from model.contour import boundary_loss

            final = preds[-1]
            pred_grid = iccad_prediction_grid(
                final, data.x, target_shape=target.shape, resize_mode="bilinear"
            )
            loss = loss + contour_loss_weight * boundary_loss(contour(pred_grid), target)
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            if contour is not None:
                torch.nn.utils.clip_grad_norm_(contour.parameters(), grad_clip)
        optimizer.step()

        total += float(loss.detach().cpu())
        del data, target, out, preds, loss
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return total / max(len(paths), 1)


def design_loss_scale(
    data,
    node_target: torch.Tensor | None,
    node_mask: torch.Tensor | None,
    mode: str,
) -> float:


    rough = data.x[:, 6].float()
    if mode == "drop":
        return max(float(rough.abs().mean()), 1.0e-9)
    if node_target is None or node_mask is None:
        raise ValueError("--target-scale residual needs node supervision targets")
    target = node_target.to(rough.device)
    mask = node_mask.to(rough.device)
    return max(float((target[mask] - rough[mask]).abs().mean()), 1.0e-9)


@torch.no_grad()
def evaluate(
    model: IRGNN,
    paths: list[Path],
    raw_data_dir: Path,
    device: torch.device,
    physics: bool = False,
    physics_cache_dir: Path | None = None,
    physics_sweep_spec: tuple[int, ...] | None = None,
    infer_dtype: str = "fp32",
    contour: torch.nn.Module | None = None,
) -> dict[str, float]:
    model.eval()
    if contour is not None:
        contour.eval()
    sums = {
        "mae_x1e4": 0.0,
        "f1": 0.0,
        "cc": 0.0,
        "mirde_x1e4": 0.0,
        "max_abs_error_x1e4": 0.0,
        "runtime_s": 0.0,
        "forward_s": 0.0,
        "raster_s": 0.0,
    }
    for path in paths:
        data = load_npz_graph(
            path,
            physics=physics,
            physics_cache_dir=physics_cache_dir,
            physics_sweeps=physics_sweep_spec,
        ).to(device)
        start = time.perf_counter()
        autocast_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}.get(infer_dtype)
        if autocast_dtype is not None and device.type == "cuda":
            cast = torch.autocast(device_type="cuda", dtype=autocast_dtype)
        else:
            cast = contextlib.nullcontext()
        with cast:
            pred = model.forward_data(data)
        if pred.dtype != torch.float32:
            pred = pred.float()
        if device.type == "cuda":
            torch.cuda.synchronize()
        forward_elapsed = time.perf_counter() - start
        raster_start = time.perf_counter()
        metrics, pred_grid, target = evaluate_iccad_node_prediction(
            pred, data.x, data.design, raw_data_dir=raw_data_dir
        )
        if contour is not None:
            with torch.no_grad():
                pred_grid = contour(pred_grid)
            metrics = compute_ir_drop_metrics(pred_grid, target, hotspot_ratio=0.9)
        raster_elapsed = time.perf_counter() - raster_start
        values = metrics.as_dict(voltage_scale=1.0e4)
        sums["mae_x1e4"] += values["mae"]
        sums["f1"] += values["f1"]
        sums["cc"] += values["cc"]
        sums["mirde_x1e4"] += values["mirde"]
        sums["max_abs_error_x1e4"] += values["max_abs_error"]
        sums["forward_s"] += forward_elapsed
        sums["raster_s"] += raster_elapsed
        sums["runtime_s"] += forward_elapsed
        del data, pred
        if device.type == "cuda":
            torch.cuda.empty_cache()

    count = max(len(paths), 1)
    return {key: value / count for key, value in sums.items()}


def format_metrics(metrics: dict[str, float]) -> str:
    return (
        f"MAE={metrics['mae_x1e4']:.4f}e-4V "
        f"F1={metrics['f1']:.4f} "
        f"CC={metrics['cc']:.4f} "
        f"MIRDE={metrics['mirde_x1e4']:.4f}e-4V "
        f"MaxErr={metrics['max_abs_error_x1e4']:.4f}e-4V "
        f"Forward={metrics.get('forward_s', metrics['runtime_s']):.3f}s/design "
        f"Raster={metrics.get('raster_s', 0.0):.3f}s/design"
    )


def main() -> int:
    args = parse_args()
    device = torch.device(args.device)
    val_graph_dir = args.val_graph_dir or args.graph_dir
    train_paths = select_paths(
        args.graph_dir,
        args.raw_data_dir,
        args.train_design,
        args.limit_train,
        args.limit_mode,
    )
    val_paths = select_paths(
        val_graph_dir,
        args.raw_data_dir,
        args.val_design,
        args.limit_val,
        args.limit_mode,
    )
    overlap = {p.stem for p in train_paths} & {p.stem for p in val_paths}
    if overlap:
        raise SystemExit(f"designs appear in both train and val: {sorted(overlap)}")
    if not train_paths and not args.eval_only:
        raise SystemExit("no training graphs found")
    if not val_paths and not args.skip_val:
        raise SystemExit("no validation graphs found")
    describe_designs("train", train_paths)
    describe_designs("val", val_paths)

    sweep_spec = physics_sweeps(args)
    cache_dir = args.physics_cache_dir
    if args.physics_features and cache_dir is None:
        cache_dir = args.graph_dir.parent / "physics_cache"

    model = build_model(args).to(device)
    start_epoch = 0
    if args.checkpoint is not None:
        start_epoch = load_checkpoint(model, args.checkpoint, device)

    if args.eval_only:
        metrics = evaluate(
            model,
            val_paths,
            args.raw_data_dir,
            device,
            physics=args.physics_features,
            physics_cache_dir=cache_dir,
            physics_sweep_spec=sweep_spec,
            infer_dtype=args.infer_dtype,
        )
        print(format_metrics(metrics))
        return 0

    args.output_dir.mkdir(parents=True, exist_ok=True)
    contour = None
    if args.contour_band != "none":
        from model.contour import ContourResidual

        contour = ContourResidual(cap=args.contour_cap, band=args.contour_band).to(device)
    parameters = list(model.parameters())
    if contour is not None:
        parameters += list(contour.parameters())
    optimizer = torch.optim.AdamW(
        parameters, lr=args.lr, weight_decay=args.weight_decay
    )
    remaining = max(args.epochs - start_epoch, 1)
    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=remaining, eta_min=args.lr_min
        )
        if args.lr_schedule == "cosine"
        else None
    )
    best_mae = load_best_mae(args.output_dir / "best.pt", device)
    best_f1 = 0.0
    f1_path = args.output_dir / "best_f1.pt"
    if f1_path.exists():
        f1_payload = torch.load(f1_path, map_location=device)
        if isinstance(f1_payload, dict) and isinstance(f1_payload.get("metrics"), dict):
            best_f1 = float(f1_payload["metrics"].get("f1", 0.0))
    node_align_cache: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    scale_cache: dict[str, float] = {}
    for epoch in range(start_epoch + 1, args.epochs + 1):
        train_loss = train_one_epoch(
            model,
            train_paths,
            args.raw_data_dir,
            optimizer,
            device,
            args.loss,
            args.grad_clip,
            hotspot_ratio=args.hotspot_ratio,
            hotspot_weight=args.hotspot_weight,
            max_weight=args.max_weight,
            relative=args.relative_loss,
            supervise=args.supervise,
            powerrush_ir_dir=args.powerrush_ir_dir,
            node_loss_weight=args.node_loss_weight,
            grid_loss_weight=args.grid_loss_weight,
            node_layers=args.node_layers,
            kcl_weight=args.kcl_weight,
            node_align_cache=node_align_cache,
            physics=args.physics_features,
            physics_cache_dir=cache_dir,
            physics_sweep_spec=sweep_spec,
            target_scale=args.target_scale,
            scale_cache=scale_cache,
            bilinear_hinge_weight=args.bilinear_hinge_weight,
            contour=contour,
            contour_loss_weight=args.contour_loss_weight,
        )
        if scheduler is not None:
            scheduler.step()
        print(f"epoch={epoch} train_loss={train_loss:.8f}")
        if (not args.skip_val) and args.eval_every > 0 and epoch % args.eval_every == 0:
            metrics = evaluate(
                model,
                val_paths,
                args.raw_data_dir,
                device,
                physics=args.physics_features,
                physics_cache_dir=cache_dir,
                physics_sweep_spec=sweep_spec,
                infer_dtype=args.infer_dtype,
                contour=contour,
            )
            print(f"epoch={epoch} val {format_metrics(metrics)}")
            if metrics["mae_x1e4"] < best_mae:
                best_mae = metrics["mae_x1e4"]
                torch.save(
                    {
                        "epoch": epoch,
                        "model": model.state_dict(),
                        "metrics": metrics,
                        "args": serializable_args(args),
                        "contour": None if contour is None else contour.state_dict(),
                    },
                    args.output_dir / "best.pt",
                )
            if metrics["f1"] > best_f1:
                best_f1 = metrics["f1"]
                torch.save(
                    {
                        "epoch": epoch,
                        "model": model.state_dict(),
                        "metrics": metrics,
                        "args": serializable_args(args),
                        "contour": None if contour is None else contour.state_dict(),
                    },
                    args.output_dir / "best_f1.pt",
                )

    torch.save(
        {
            "epoch": args.epochs,
            "model": model.state_dict(),
            "args": serializable_args(args),
            "contour": None if contour is None else contour.state_dict(),
        },
        args.output_dir / "last.pt",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
