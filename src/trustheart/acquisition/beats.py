from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from ..instrument.types import Beat


class BeatDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, beats: list[Beat]) -> None:
        signals = np.stack([b.signal for b in beats]).astype(np.float32)
        labels = np.asarray([b.label for b in beats], dtype=np.int64)
        self.signals = torch.from_numpy(signals).unsqueeze(1)
        self.labels = torch.from_numpy(labels)
        self.subjects = [b.subject for b in beats]

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.signals[idx], self.labels[idx]
