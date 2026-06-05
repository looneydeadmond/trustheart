from __future__ import annotations

import math


def allocate_budget(
    sensitivities: dict[str, float], total_epsilon: float, alpha: float
) -> dict[str, float]:
    inverse = {k: (s ** (-alpha)) if s > 1e-12 else 0.0 for k, s in sensitivities.items()}
    denom = sum(inverse.values())
    if denom <= 0.0:
        uniform = total_epsilon / max(1, len(sensitivities))
        return {k: uniform for k in sensitivities}
    return {k: total_epsilon * v / denom for k, v in inverse.items()}


def noise_scales(
    budgets: dict[str, float],
    clip: dict[str, float],
    steps: int,
    delta: float,
    coefficient: float,
) -> dict[str, float]:
    factor = math.sqrt(2.0 * steps * math.log(1.0 / delta))
    return {k: coefficient * clip[k] * factor / max(budgets[k], 1e-8) for k in budgets}
