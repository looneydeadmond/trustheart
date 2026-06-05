from __future__ import annotations

import torch
from torch import nn

from ..assembly.model import TrustHeart


def _grad_sign(
    model: TrustHeart, x: torch.Tensor, y: torch.Tensor, smooth: float = 0.0
) -> torch.Tensor:
    loss_fn = nn.CrossEntropyLoss()
    x = x.clone().detach().requires_grad_(True)
    loss = loss_fn(model.logits(x), y)
    grad = torch.autograd.grad(loss, x)[0]
    if smooth > 0.0:
        kernel = torch.ones(1, 1, 5, device=x.device) / 5.0
        grad = torch.nn.functional.conv1d(grad, kernel, padding=2)
    return grad.sign()


def fgsm(model: TrustHeart, x: torch.Tensor, y: torch.Tensor, epsilon: float = 0.1) -> torch.Tensor:
    perturbed = x + epsilon * _grad_sign(model, x, y)
    return perturbed.detach()


def pgd(
    model: TrustHeart,
    x: torch.Tensor,
    y: torch.Tensor,
    epsilon: float = 0.1,
    iters: int = 40,
    step: float = 0.01,
) -> torch.Tensor:
    adv = x.clone().detach()
    for _ in range(iters):
        adv = adv + step * _grad_sign(model, adv, y)
        delta = torch.clamp(adv - x, min=-epsilon, max=epsilon)
        adv = (x + delta).detach()
    return adv


def sap(
    model: TrustHeart,
    x: torch.Tensor,
    y: torch.Tensor,
    epsilon: float = 0.1,
    iters: int = 40,
    step: float = 0.01,
) -> torch.Tensor:
    adv = x.clone().detach()
    for _ in range(iters):
        adv = adv + step * _grad_sign(model, adv, y, smooth=3.0)
        delta = torch.clamp(adv - x, min=-epsilon, max=epsilon)
        adv = (x + delta).detach()
    return adv
