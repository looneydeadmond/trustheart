from __future__ import annotations

from collections.abc import Callable

import torch
from torch import nn

from .sensitivity import per_sample_norms

LossFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


def per_sample_gradients(
    model: nn.Module,
    loss_fn: LossFn,
    logits_fn: Callable[[torch.Tensor], torch.Tensor],
    x: torch.Tensor,
    y: torch.Tensor,
) -> dict[str, torch.Tensor]:
    names = [n for n, p in model.named_parameters() if p.requires_grad]
    buffers: dict[str, list[torch.Tensor]] = {n: [] for n in names}
    for i in range(x.shape[0]):
        model.zero_grad(set_to_none=True)
        logits = logits_fn(x[i : i + 1])
        loss = loss_fn(logits, y[i : i + 1])
        torch.autograd.backward(loss)
        for n, p in model.named_parameters():
            if p.requires_grad:
                grad = p.grad
                buffers[n].append(torch.zeros_like(p) if grad is None else grad.detach().clone())
    model.zero_grad(set_to_none=True)
    return {n: torch.stack(buffers[n]) for n in names}


class PrivateStep:
    def __init__(self, clip: dict[str, float], noise: dict[str, float]) -> None:
        self.clip = clip
        self.noise = noise

    def privatize(self, per_sample_grads: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        aggregated: dict[str, torch.Tensor] = {}
        for name, grads in per_sample_grads.items():
            batch = grads.shape[0]
            clip_value = self.clip.get(name, 1.0)
            norms = per_sample_norms(grads).clamp_min(1e-12)
            factor = torch.clamp(clip_value / norms, max=1.0)
            shape = [batch] + [1] * (grads.dim() - 1)
            clipped = grads * factor.view(shape)
            summed = clipped.sum(dim=0)
            sigma = self.noise.get(name, 0.0)
            noise = torch.randn_like(summed) * sigma
            aggregated[name] = (summed + noise) / batch
        return aggregated
