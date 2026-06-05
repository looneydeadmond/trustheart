from __future__ import annotations

import torch

from trustheart.watchdog.detector import AnomalyWatchdog
from trustheart.watchdog.mixture import DiagonalGMM


def test_gmm_assigns_higher_density_in_distribution() -> None:
    torch.manual_seed(0)
    data = torch.randn(200, 8)
    gmm = DiagonalGMM(n_components=2, dim=8).fit(data)
    inside = gmm.log_prob(torch.zeros(1, 8))
    outside = gmm.log_prob(20.0 * torch.ones(1, 8))
    assert float(inside.item()) > float(outside.item())


def test_watchdog_flags_shifted_inputs() -> None:
    torch.manual_seed(0)
    dim = 16
    n_classes = 3
    emb = torch.randn(300, dim)
    labels = torch.randint(0, n_classes, (300,))
    watch = AnomalyWatchdog(n_classes, dim=dim).fit(emb, labels)
    preds = labels
    clean_scores = watch.score(emb, preds)
    watch.calibrate(clean_scores, percentile=95.0)
    shifted = emb + 12.0
    shifted_scores = watch.score(shifted, preds)
    adr = float((shifted_scores > watch.threshold).float().mean())
    fpr = float((clean_scores > watch.threshold).float().mean())
    assert adr > 0.8
    assert fpr < 0.1


def test_threshold_is_finite_after_calibration() -> None:
    watch = AnomalyWatchdog(2, dim=4)
    watch.fit(torch.randn(40, 4), torch.randint(0, 2, (40,)))
    tau = watch.calibrate(watch.score(torch.randn(40, 4), torch.randint(0, 2, (40,))))
    assert tau == tau
