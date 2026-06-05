from __future__ import annotations

import torch
from torch import nn

from .blocks import ConvBlock


class ConvFabric(nn.Module):
    def __init__(
        self,
        in_ch: int = 1,
        channels: tuple[int, ...] = (32, 64, 128, 128),
        kernel: int = 7,
    ) -> None:
        super().__init__()
        blocks: list[nn.Module] = []
        prev = in_ch
        for ch in channels:
            blocks.append(ConvBlock(prev, ch, kernel=kernel, stride=2))
            prev = ch
        self.blocks = nn.ModuleList(blocks)
        self.embed_dim = prev

    def feature_map(self, x: torch.Tensor) -> torch.Tensor:
        out = x
        for block in self.blocks:
            out = block(out)
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mapped = self.feature_map(x)
        pooled = mapped.mean(dim=-1)
        return pooled
