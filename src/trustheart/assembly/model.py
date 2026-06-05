from __future__ import annotations

import torch
from torch import nn

from ..decoder.head import ClassifierHead
from ..fabric.backbone import ConvFabric
from ..instrument.types import ForwardOutputs
from ..salience.attention import PrivacyAwareAttention
from .config import ModelConfig


class TrustHeart(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.fabric = ConvFabric(config.in_channels, config.channels, config.kernel)
        dim = self.fabric.embed_dim
        self.use_paam = config.use_paam
        if config.use_paam:
            self.attention: PrivacyAwareAttention | None = PrivacyAwareAttention(
                dim, config.attention_heads, config.lambda_att
            )
        else:
            self.attention = None
        self.head = ClassifierHead(dim, 64, config.n_classes)

    def forward(self, x: torch.Tensor, perturb: bool = False) -> ForwardOutputs:
        feature_map = self.fabric.feature_map(x)
        attention: torch.Tensor | None = None
        if self.attention is not None:
            pooled, attention = self.attention(feature_map, perturb=perturb)
        else:
            pooled = feature_map.mean(dim=-1)
        logits = self.head(pooled)
        return ForwardOutputs(logits=logits, embedding=pooled, attention=attention)

    def logits(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x).logits

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x).embedding
