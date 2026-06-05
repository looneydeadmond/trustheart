from __future__ import annotations

import numpy as np


def bootstrap_ci(
    values: np.ndarray,
    statistic: str = "mean",
    resamples: int = 1000,
    level: float = 0.95,
    seed: int = 0,
) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    n = values.shape[0]
    estimates = np.empty(resamples, dtype=np.float64)
    for i in range(resamples):
        sample = values[rng.integers(0, n, size=n)]
        estimates[i] = float(np.mean(sample)) if statistic == "mean" else float(np.median(sample))
    point = float(np.mean(values)) if statistic == "mean" else float(np.median(values))
    lower = float(np.quantile(estimates, (1.0 - level) / 2.0))
    upper = float(np.quantile(estimates, 1.0 - (1.0 - level) / 2.0))
    return point, lower, upper
