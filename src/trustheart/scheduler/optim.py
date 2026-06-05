from __future__ import annotations

import torch
from torch import nn

from ..assembly.config import TrainConfig


def build_optimizer(model: nn.Module, config: TrainConfig) -> torch.optim.Optimizer:
    return torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )


def build_schedule(
    optimizer: torch.optim.Optimizer, epochs: int
) -> torch.optim.lr_scheduler.CosineAnnealingLR:
    return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))
