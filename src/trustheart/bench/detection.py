from __future__ import annotations

import torch


def detection_rates(
    clean_scores: torch.Tensor, adversarial_scores: torch.Tensor, threshold: float
) -> dict[str, float]:
    adr = float((adversarial_scores > threshold).float().mean().item())
    fpr = float((clean_scores > threshold).float().mean().item())
    return {"adr": adr, "fpr": fpr}
