from __future__ import annotations

import torch
from torch import nn


class ClassifierHead(nn.Module):
    def __init__(self, dim: int = 128, hidden: int = 64, n_classes: int = 5) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(hidden, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out: torch.Tensor = self.fc2(self.act(self.fc1(x)))
        return out
