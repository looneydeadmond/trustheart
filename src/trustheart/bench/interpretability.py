from __future__ import annotations

import torch

from ..assembly.model import TrustHeart
from ..salience.saliency import saliency_map, upsample_to


def position_relevance(model: TrustHeart, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if model.attention is None:
        raise ValueError("relevance requires the attention module")
    feature_map = model.fabric.feature_map(x)
    tokens = feature_map.transpose(1, 2).clone().detach().requires_grad_(True)
    pooled, _ = model.attention(tokens.transpose(1, 2))
    logits = model.head(pooled)
    selected = logits.gather(1, target.view(-1, 1)).sum()
    grad = torch.autograd.grad(selected, tokens)[0]
    return grad.abs().mean(dim=-1)


def saliency_over_signal(model: TrustHeart, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    outputs = model.forward(x)
    if outputs.attention is None:
        raise ValueError("saliency requires the attention module")
    relevance = position_relevance(model, x, target)
    pooled = saliency_map(outputs.attention, relevance)
    return upsample_to(pooled, x.shape[-1])


def iou(saliency: torch.Tensor, annotation: torch.Tensor, quantile: float = 0.7) -> float:
    threshold = torch.quantile(saliency, quantile, dim=-1, keepdim=True)
    predicted = saliency >= threshold
    target = annotation > 0.5
    intersection = (predicted & target).float().sum(dim=-1)
    union = (predicted | target).float().sum(dim=-1).clamp_min(1.0)
    return float((intersection / union).mean().item())


def pointing_game(saliency: torch.Tensor, annotation: torch.Tensor) -> float:
    peak = saliency.argmax(dim=-1)
    hit = annotation.gather(1, peak.view(-1, 1)).squeeze(-1) > 0.5
    return float(hit.float().mean().item())
