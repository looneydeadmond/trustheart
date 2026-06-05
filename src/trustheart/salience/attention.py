from __future__ import annotations

import math

import torch
from torch import nn

from .perturbation import laplacian_like


def scaled_dot_product(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, scores_bias: torch.Tensor | None
) -> tuple[torch.Tensor, torch.Tensor]:
    d_k = q.shape[-1]
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
    if scores_bias is not None:
        scores = scores + scores_bias
    weights = torch.softmax(scores, dim=-1)
    context = torch.matmul(weights, v)
    return context, weights


class PrivacyAwareAttention(nn.Module):
    def __init__(self, dim: int = 128, heads: int = 4, lambda_att: float = 0.0) -> None:
        super().__init__()
        if dim % heads != 0:
            raise ValueError("dim must be divisible by heads")
        self.dim = dim
        self.heads = heads
        self.d_k = dim // heads
        self.lambda_att = lambda_att
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

    def _split(self, x: torch.Tensor) -> torch.Tensor:
        b, n, _ = x.shape
        return x.view(b, n, self.heads, self.d_k).transpose(1, 2)

    def forward(
        self, feature_map: torch.Tensor, perturb: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        tokens = feature_map.transpose(1, 2)
        q = self._split(self.q_proj(tokens))
        k = self._split(self.k_proj(tokens))
        v = self._split(self.v_proj(tokens))

        bias: torch.Tensor | None = None
        if perturb and self.lambda_att > 0.0:
            raw = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
            bias = laplacian_like(raw, self.lambda_att)

        context, weights = scaled_dot_product(q, k, v, bias)
        b, _, n, _ = context.shape
        merged = context.transpose(1, 2).contiguous().view(b, n, self.dim)
        attended = self.out_proj(merged)
        pooled = attended.mean(dim=1)
        attention = weights.mean(dim=1)
        return pooled, attention
