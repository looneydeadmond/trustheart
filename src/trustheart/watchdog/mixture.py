from __future__ import annotations

import math

import torch


class DiagonalGMM:
    def __init__(self, n_components: int = 2, dim: int = 128, eps: float = 1e-4) -> None:
        self.n_components = n_components
        self.dim = dim
        self.eps = eps
        self.weights = torch.full((n_components,), 1.0 / n_components)
        self.means = torch.zeros(n_components, dim)
        self.variances = torch.ones(n_components, dim)

    def _component_log_prob(self, x: torch.Tensor) -> torch.Tensor:
        diff = x.unsqueeze(1) - self.means.unsqueeze(0)
        var = self.variances.unsqueeze(0)
        quad = -0.5 * (diff**2 / var).sum(dim=-1)
        logdet = -0.5 * torch.log(var).sum(dim=-1)
        const = -0.5 * self.dim * math.log(2.0 * math.pi)
        return quad + logdet + const

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        weighted = self._component_log_prob(x) + torch.log(self.weights + 1e-12).unsqueeze(0)
        return torch.logsumexp(weighted, dim=-1)

    def fit(self, x: torch.Tensor, iters: int = 50, seed: int = 0) -> DiagonalGMM:
        n = x.shape[0]
        g = torch.Generator().manual_seed(seed)
        pick = torch.randperm(n, generator=g)[: self.n_components]
        self.means = x[pick].clone()
        self.variances = (
            x.var(dim=0, unbiased=False).clamp_min(self.eps).repeat(self.n_components, 1)
        )
        self.weights = torch.full((self.n_components,), 1.0 / self.n_components)
        for _ in range(iters):
            log_resp = self._component_log_prob(x) + torch.log(self.weights + 1e-12).unsqueeze(0)
            log_norm = torch.logsumexp(log_resp, dim=-1, keepdim=True)
            resp = torch.exp(log_resp - log_norm)
            nk = resp.sum(dim=0).clamp_min(1e-8)
            self.weights = nk / n
            self.means = (resp.t() @ x) / nk.unsqueeze(-1)
            centred = x.unsqueeze(1) - self.means.unsqueeze(0)
            self.variances = (
                (resp.unsqueeze(-1) * centred**2).sum(dim=0) / nk.unsqueeze(-1)
            ).clamp_min(self.eps)
        return self
