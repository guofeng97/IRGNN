import argparse
import json
from argparse import Namespace
from pathlib import Path

import torch

from model.contour import ContourResidual
from model.train_iccad import build_model, evaluate, format_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("config.json"))
    parser.add_argument("--checkpoint", type=Path, default=Path("best_f1.pt"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    config.setdefault("predict_direct", False)
    config.setdefault("no_residual_zero_init", False)
    config.setdefault("landmark_key_bins", 0)
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    model_args = Namespace(**config)
    model = build_model(model_args).to(args.device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    contour = ContourResidual(config["contour_cap"], config["contour_band"]).to(args.device)
    contour.load_state_dict(checkpoint["contour"], strict=True)
    contour.eval()
    paths = sorted(Path(config["val_graph_dir"]).glob("*.npz"), key=lambda path: path.stem)
    sweeps = tuple(int(value) for value in config["physics_sweeps"].split(","))
    metrics = evaluate(
        model,
        paths,
        Path(config["raw_data_dir"]),
        torch.device(args.device),
        physics=config["physics_features"],
        physics_cache_dir=Path(config["physics_cache_dir"]),
        physics_sweep_spec=sweeps,
        infer_dtype="fp32",
        contour=contour,
    )
    print(format_metrics(metrics))


if __name__ == "__main__":
    main()
