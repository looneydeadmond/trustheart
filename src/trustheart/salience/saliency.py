from __future__ import annotations

import torch
import torch.nn.functional as F


def saliency_map(attention: torch.Tensor, relevance: torch.Tensor) -> torch.Tensor:
    weighted = attention * relevance.unsqueeze(-1)
    pooled = weighted.sum(dim=1)
    return pooled


def upsample_to(saliency: torch.Tensor, length: int) -> torch.Tensor:
    grid = saliency.unsqueeze(1)
    resized = F.interpolate(grid, size=length, mode="linear", align_corners=False)
    flat = resized.squeeze(1)
    lo = flat.amin(dim=-1, keepdim=True)
    hi = flat.amax(dim=-1, keepdim=True)
    return (flat - lo) / (hi - lo + 1e-8)
