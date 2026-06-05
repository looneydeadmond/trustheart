from __future__ import annotations

import numpy as np
from scipy.stats import wilcoxon


def mcnemar(correct_a: np.ndarray, correct_b: np.ndarray) -> dict[str, float]:
    b = int(np.sum(correct_a & ~correct_b))
    c = int(np.sum(~correct_a & correct_b))
    denom = b + c
    statistic = ((abs(b - c) - 1) ** 2) / denom if denom > 0 else 0.0
    from scipy.stats import chi2

    p_value = float(chi2.sf(statistic, df=1)) if denom > 0 else 1.0
    return {"statistic": float(statistic), "p_value": p_value, "b": float(b), "c": float(c)}


def wilcoxon_test(values_a: np.ndarray, values_b: np.ndarray) -> dict[str, float]:
    statistic, p_value = wilcoxon(values_a, values_b)
    return {"statistic": float(statistic), "p_value": float(p_value)}


def bonferroni(alpha: float, comparisons: int) -> float:
    return alpha / max(1, comparisons)
