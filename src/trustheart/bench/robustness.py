from __future__ import annotations

import torch

from ..assembly.model import TrustHeart
from .attacks import fgsm, pgd, sap
from .classification import classification_metrics

ATTACKS = {"fgsm": fgsm, "pgd": pgd, "sap": sap}


def evaluate_clean(model: TrustHeart, x: torch.Tensor, y: torch.Tensor) -> dict[str, float]:
    model.eval()
    with torch.no_grad():
        logits = model.logits(x)
    return classification_metrics(logits, y)


def evaluate_attack(
    model: TrustHeart,
    x: torch.Tensor,
    y: torch.Tensor,
    attack: str,
    epsilon: float = 0.1,
) -> dict[str, float]:
    model.eval()
    if attack == "fgsm":
        adv = fgsm(model, x, y, epsilon)
    elif attack == "pgd":
        adv = pgd(model, x, y, epsilon)
    else:
        adv = sap(model, x, y, epsilon)
    with torch.no_grad():
        logits = model.logits(adv)
    return classification_metrics(logits, y)
