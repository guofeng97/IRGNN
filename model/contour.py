from __future__ import annotations

import torch
from torch import Tensor, nn


class ContourResidual(nn.Module):

    def __init__(self, cap: float = 0.06, band: str = "shoulder") -> None:
        super().__init__()
        if band not in {"shoulder", "gauss"}:
            raise ValueError(f"unknown contour band: {band}")
        self.cap = float(cap)
        self.band = band
        self.net = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(8, 1, 3, padding=1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, pred: Tensor) -> Tensor:
        finite = torch.isfinite(pred)
        peak = pred[finite].detach().amax().clamp_min(1.0e-12)
        frac = pred.detach() / peak
        if self.band == "shoulder":

            band = ((frac >= 0.90) & (frac < 0.97)).to(dtype=pred.dtype)
        else:
            line = 0.9 * peak
            band = torch.exp(-((pred.detach() - line) / (0.05 * line)).pow(2))
        band = band * finite
        raw = self.net((pred.detach() / peak)[None, None])[0, 0]
        delta = self.cap * peak * torch.tanh(raw)
        return pred + band * delta


def boundary_loss(pred: Tensor, target: Tensor) -> Tensor:


    finite = torch.isfinite(pred) & torch.isfinite(target)
    if not bool(finite.any()):
        return pred.new_zeros(())
    threshold = target[finite].max() * 0.9
    peak = pred[finite].detach().amax().clamp_min(1.0e-12)
    missed = finite & (target >= threshold) & (pred < threshold)
    extra = finite & (target < threshold) & (pred >= threshold)
    terms = []
    if bool(missed.any()):
        terms.append(((threshold - pred[missed]) / peak).mean())
    if bool(extra.any()):
        terms.append(((pred[extra] - threshold) / peak).mean())
    if not terms:
        return pred.new_zeros(())
    return sum(terms) / len(terms)
