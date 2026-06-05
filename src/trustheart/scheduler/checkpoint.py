from __future__ import annotations

from pathlib import Path
from typing import Any

from ..assembly.model import TrustHeart
from ..instrument.io import atomic_save, load_checkpoint


def save_model(
    model: TrustHeart, seed: int, path: str | Path, extra: dict[str, Any] | None = None
) -> None:
    payload: dict[str, Any] = {
        "state_dict": model.state_dict(),
        "config": model.config,
        "seed": seed,
    }
    if extra is not None:
        payload["extra"] = extra
    atomic_save(payload, path)


def restore_model(path: str | Path, map_location: str = "cpu") -> tuple[TrustHeart, int]:
    payload = load_checkpoint(path, map_location)
    model = TrustHeart(payload["config"])
    model.load_state_dict(payload["state_dict"])
    seed = int(payload.get("seed", 0))
    return model, seed
