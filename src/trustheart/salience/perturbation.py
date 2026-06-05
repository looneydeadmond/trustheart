from __future__ import annotations

import torch


def attention_sensitivity(scores_a: torch.Tensor, scores_b: torch.Tensor) -> float:
    diff = (scores_a - scores_b).abs().sum(dim=-1)
    return float(diff.max().item())


def laplace_scale(sensitivity: float, epsilon: float) -> float:
    return sensitivity / max(epsilon, 1e-6)


def laplacian_like(reference: torch.Tensor, scale: float) -> torch.Tensor:
    uniform = torch.rand_like(reference) - 0.5
    return -scale * torch.sign(uniform) * torch.log1p(-2.0 * uniform.abs())
