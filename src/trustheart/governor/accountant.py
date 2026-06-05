from __future__ import annotations

import math

from ..instrument.types import PrivacySpend

DEFAULT_ORDERS: tuple[float, ...] = (2.0, 5.0, 10.0, 20.0, 50.0, 100.0)


def rdp_gaussian(
    noise_multiplier: float, steps: int, orders: tuple[float, ...]
) -> dict[float, float]:
    return {a: steps * a / (2.0 * noise_multiplier**2) for a in orders}


def rdp_to_epsilon(rdp_by_order: dict[float, float], delta: float) -> PrivacySpend:
    best = PrivacySpend(epsilon=float("inf"), delta=delta, order=0.0)
    for order, rdp in rdp_by_order.items():
        if order <= 1.0:
            continue
        candidate = rdp + math.log(1.0 / delta) / (order - 1.0)
        if candidate < best.epsilon:
            best = PrivacySpend(epsilon=candidate, delta=delta, order=order)
    return best


def epsilon_from_noise(
    noise_multiplier: float,
    steps: int,
    delta: float,
    orders: tuple[float, ...] = DEFAULT_ORDERS,
) -> PrivacySpend:
    return rdp_to_epsilon(rdp_gaussian(noise_multiplier, steps, orders), delta)


def calibrate_noise_multiplier(
    target_epsilon: float,
    steps: int,
    delta: float,
    orders: tuple[float, ...] = DEFAULT_ORDERS,
    tolerance: float = 1e-3,
) -> float:
    low, high = 0.1, 100.0
    for _ in range(100):
        mid = 0.5 * (low + high)
        spent = epsilon_from_noise(mid, steps, delta, orders).epsilon
        if abs(spent - target_epsilon) < tolerance:
            return mid
        if spent > target_epsilon:
            low = mid
        else:
            high = mid
    return 0.5 * (low + high)
