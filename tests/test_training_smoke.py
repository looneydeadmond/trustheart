from __future__ import annotations

import math
from pathlib import Path

import torch

from trustheart.acquisition.beats import BeatDataset
from trustheart.acquisition.synthetic import SyntheticEcg
from trustheart.assembly.config import (
    DataConfig,
    ExperimentConfig,
    ModelConfig,
    PrivacyConfig,
    TrainConfig,
)
from trustheart.assembly.model import TrustHeart
from trustheart.scheduler.loop import train

ROOT = Path(__file__).resolve().parents[1]


def _experiment(private: bool, steps: int) -> ExperimentConfig:
    return ExperimentConfig(
        name="smoke",
        model=ModelConfig(n_classes=4),
        data=DataConfig(source="synthetic", n_classes=4, per_class=8, seed=1),
        privacy=PrivacyConfig(enabled=private),
        train=TrainConfig(epochs=1, batch_size=16, max_steps=steps, seed=1, device="cpu"),
    )


def test_non_private_loss_decreases() -> None:
    exp = _experiment(private=False, steps=40)
    beats = SyntheticEcg(4, 16, seed=1).beats()
    dataset = BeatDataset(beats)
    model = TrustHeart(exp.model)
    state = train(model, dataset, exp, torch.device("cpu"))
    assert state.history[-1] < state.history[0]


def test_private_smoke_runs_and_is_finite() -> None:
    exp = _experiment(private=True, steps=2)
    beats = SyntheticEcg(4, 8, seed=1).beats()
    dataset = BeatDataset(beats)
    model = TrustHeart(exp.model)
    state = train(model, dataset, exp, torch.device("cpu"))
    assert state.step == 2
    assert all(math.isfinite(v) for v in state.history)


def test_smoke_config_file_exists() -> None:
    assert (ROOT / "configs" / "experiment" / "_smoke.yaml").exists()
