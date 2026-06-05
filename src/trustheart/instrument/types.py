from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

AAMI_CLASSES: tuple[str, ...] = ("N", "S", "V", "F", "Q")
CPSC_CLASSES: tuple[str, ...] = (
    "N",
    "AF",
    "I-AVB",
    "LBBB",
    "RBBB",
    "PAC",
    "PVC",
    "STD",
    "STE",
)


@dataclass(frozen=True)
class Beat:
    signal: np.ndarray
    label: int
    subject: str


@dataclass
class ForwardOutputs:
    logits: torch.Tensor
    embedding: torch.Tensor
    attention: torch.Tensor | None = None
    saliency: torch.Tensor | None = None


@dataclass
class PrivacySpend:
    epsilon: float
    delta: float
    order: float


@dataclass
class TrainState:
    step: int = 0
    epoch: int = 0
    history: list[float] = field(default_factory=list)
