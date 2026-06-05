from __future__ import annotations

import torch


def per_sample_norms(grad: torch.Tensor) -> torch.Tensor:
    flat = grad.reshape(grad.shape[0], -1)
    norms: torch.Tensor = flat.norm(dim=1)
    return norms


def gradient_sensitivity(per_sample_grads: dict[str, torch.Tensor]) -> dict[str, float]:
    out: dict[str, float] = {}
    for name, grad in per_sample_grads.items():
        norms = per_sample_norms(grad)
        out[name] = float(torch.sqrt((norms**2).mean()).item())
    return out


def clip_thresholds(
    per_sample_grads: dict[str, torch.Tensor], percentile: float = 90.0
) -> dict[str, float]:
    out: dict[str, float] = {}
    for name, grad in per_sample_grads.items():
        norms = per_sample_norms(grad)
        out[name] = float(torch.quantile(norms, percentile / 100.0).item())
    return out
