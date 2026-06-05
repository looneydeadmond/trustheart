from __future__ import annotations

import numpy as np
import torch

from trustheart.assembly.config import ModelConfig
from trustheart.assembly.model import TrustHeart
from trustheart.bench.classification import classification_metrics
from trustheart.bench.detection import detection_rates
from trustheart.bench.footprint import footprint_report
from trustheart.bench.intervals import bootstrap_ci
from trustheart.bench.significance import bonferroni, mcnemar


def test_classification_accuracy_matches_manual() -> None:
    logits = torch.tensor([[2.0, 0.0], [0.0, 3.0], [1.0, 0.5], [0.0, 2.0]])
    targets = torch.tensor([0, 1, 0, 0])
    metrics = classification_metrics(logits, targets)
    assert abs(metrics["accuracy"] - 0.75) < 1e-9


def test_detection_rates_bounds() -> None:
    clean = torch.zeros(100)
    adversarial = torch.ones(100) * 5.0
    rates = detection_rates(clean, adversarial, threshold=1.0)
    assert rates["adr"] == 1.0
    assert rates["fpr"] == 0.0


def test_bootstrap_brackets_mean() -> None:
    values = np.random.default_rng(0).normal(0.9, 0.01, size=200)
    point, lo, hi = bootstrap_ci(values, resamples=200)
    assert lo <= point <= hi


def test_mcnemar_and_bonferroni() -> None:
    a = np.array([True, True, False, True])
    b = np.array([False, True, False, False])
    result = mcnemar(a, b)
    assert "p_value" in result
    assert abs(bonferroni(0.01, 6) - 0.01 / 6) < 1e-12


def test_footprint_close_to_paper_scale() -> None:
    model = TrustHeart(ModelConfig(n_classes=5))
    report = footprint_report(model)
    assert 150.0 < report["int8_size_kb"] < 400.0
