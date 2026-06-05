from __future__ import annotations

import torch

from trustheart.governor.accountant import calibrate_noise_multiplier, epsilon_from_noise
from trustheart.governor.allocation import allocate_budget, noise_scales
from trustheart.governor.sensitivity import gradient_sensitivity


def test_budget_sums_to_total() -> None:
    sens = {"a": 1.0, "b": 2.0, "c": 4.0}
    budgets = allocate_budget(sens, total_epsilon=4.2, alpha=0.5)
    assert abs(sum(budgets.values()) - 4.2) < 1e-6


def test_higher_sensitivity_gets_less_budget() -> None:
    sens = {"low": 1.0, "high": 9.0}
    budgets = allocate_budget(sens, total_epsilon=4.2, alpha=0.5)
    assert budgets["low"] > budgets["high"]


def test_noise_increases_with_sensitivity() -> None:
    sens = {"low": 1.0, "high": 9.0}
    budgets = allocate_budget(sens, total_epsilon=4.2, alpha=0.5)
    clip = {"low": 1.0, "high": 1.0}
    noise = noise_scales(budgets, clip, steps=100, delta=1e-5, coefficient=1.0)
    assert noise["high"] > noise["low"]


def test_gradient_sensitivity_orders_layers() -> None:
    grads = {
        "small": 0.1 * torch.ones(8, 4),
        "large": 3.0 * torch.ones(8, 4),
    }
    sens = gradient_sensitivity(grads)
    assert sens["large"] > sens["small"]


def test_accountant_roundtrip_epsilon() -> None:
    target = 4.2
    z = calibrate_noise_multiplier(target, steps=500, delta=1e-5)
    spent = epsilon_from_noise(z, steps=500, delta=1e-5)
    assert abs(spent.epsilon - target) < 1e-2


def test_more_noise_lowers_epsilon() -> None:
    high_noise = epsilon_from_noise(8.0, steps=500, delta=1e-5).epsilon
    low_noise = epsilon_from_noise(2.0, steps=500, delta=1e-5).epsilon
    assert high_noise < low_noise
