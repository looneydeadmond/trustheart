from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch


def atomic_save(payload: dict[str, Any], path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, target)


def load_checkpoint(path: str | Path, map_location: str = "cpu") -> dict[str, Any]:
    payload: dict[str, Any] = torch.load(Path(path), map_location=map_location, weights_only=False)
    return payload
