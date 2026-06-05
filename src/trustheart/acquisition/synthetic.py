from __future__ import annotations

import random

import numpy as np

from ..instrument.types import Beat


def _gaussian(t: np.ndarray, center: float, amp: float, width: float) -> np.ndarray:
    return amp * np.exp(-((t - center) ** 2) / (2.0 * width**2))


def _morphology(class_index: int, n_classes: int) -> dict[str, float]:
    phase = class_index / max(1, n_classes)
    p_amp = 0.12 * (1.0 - 0.8 * (class_index % 3 == 1))
    r_amp = 1.0 + 0.4 * np.sin(2.0 * np.pi * phase)
    qrs_width = 0.012 + 0.02 * (class_index % 4 == 2)
    t_amp = 0.25 + 0.2 * np.cos(2.0 * np.pi * phase)
    st_shift = 0.15 * (class_index % 5 == 4) - 0.1 * (class_index % 5 == 3)
    return {
        "p_amp": float(p_amp),
        "r_amp": float(r_amp),
        "qrs_width": float(qrs_width),
        "t_amp": float(t_amp),
        "st_shift": float(st_shift),
    }


def synth_beat(
    class_index: int, n_classes: int, rng: np.random.Generator, length: int = 256
) -> np.ndarray:
    t = np.linspace(0.0, 1.0, length)
    m = _morphology(class_index, n_classes)
    wave = np.zeros(length, dtype=np.float64)
    wave += _gaussian(t, 0.20, m["p_amp"], 0.025)
    wave += _gaussian(t, 0.42, -0.18, m["qrs_width"])
    wave += _gaussian(t, 0.50, m["r_amp"], m["qrs_width"])
    wave += _gaussian(t, 0.58, -0.22, m["qrs_width"])
    wave += _gaussian(t, 0.75 + m["st_shift"] * 0.0, m["t_amp"], 0.05)
    wave += m["st_shift"] * (t > 0.6) * (t < 0.72)
    baseline = 0.05 * np.sin(2.0 * np.pi * (1.5 + 0.5 * rng.random()) * t + rng.random())
    noise = 0.02 * rng.standard_normal(length)
    return (wave + baseline + noise).astype(np.float32)


class SyntheticEcg:
    def __init__(
        self,
        n_classes: int,
        per_class: int = 64,
        seed: int = 0,
        length: int = 256,
    ) -> None:
        self.n_classes = n_classes
        self.per_class = per_class
        self.seed = seed
        self.length = length

    def beats(self) -> list[Beat]:
        rng = np.random.default_rng(self.seed)
        out: list[Beat] = []
        for c in range(self.n_classes):
            for i in range(self.per_class):
                signal = synth_beat(c, self.n_classes, rng, self.length)
                out.append(Beat(signal=signal, label=c, subject=f"synthetic-{c}-{i % 4}"))
        random.Random(self.seed).shuffle(out)
        return out
