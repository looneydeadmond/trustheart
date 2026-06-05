from __future__ import annotations

import torch

from .mixture import DiagonalGMM


class AnomalyWatchdog:
    def __init__(
        self, n_classes: int, dim: int = 128, components: int = 2, beta: float = 0.5
    ) -> None:
        self.n_classes = n_classes
        self.dim = dim
        self.components = components
        self.beta = beta
        self.gmms: dict[int, DiagonalGMM] = {}
        self.class_mean = torch.zeros(n_classes, dim)
        self.class_var = torch.ones(n_classes, dim)
        self.threshold = float("inf")

    def fit(self, embeddings: torch.Tensor, labels: torch.Tensor, seed: int = 0) -> AnomalyWatchdog:
        for c in range(self.n_classes):
            mask = labels == c
            if int(mask.sum().item()) < self.components:
                continue
            members = embeddings[mask]
            self.gmms[c] = DiagonalGMM(self.components, self.dim).fit(members, seed=seed)
            self.class_mean[c] = members.mean(dim=0)
            self.class_var[c] = members.var(dim=0, unbiased=False).clamp_min(1e-4)
        return self

    def _mahalanobis(self, x: torch.Tensor, predicted: torch.Tensor) -> torch.Tensor:
        mean = self.class_mean[predicted]
        var = self.class_var[predicted]
        return torch.sqrt(((x - mean) ** 2 / var).sum(dim=-1) + 1e-8)

    def score(self, embeddings: torch.Tensor, predicted: torch.Tensor) -> torch.Tensor:
        log_likelihood = torch.zeros(embeddings.shape[0])
        for c, gmm in self.gmms.items():
            mask = predicted == c
            if bool(mask.any()):
                log_likelihood[mask] = gmm.log_prob(embeddings[mask])
        distance = self._mahalanobis(embeddings, predicted)
        return -log_likelihood + self.beta * distance

    def calibrate(self, clean_scores: torch.Tensor, percentile: float = 95.0) -> float:
        self.threshold = float(torch.quantile(clean_scores, percentile / 100.0).item())
        return self.threshold

    def detect(self, scores: torch.Tensor) -> torch.Tensor:
        return scores > self.threshold
