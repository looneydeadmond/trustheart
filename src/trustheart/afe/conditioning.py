from __future__ import annotations

import numpy as np
from scipy.signal import butter, filtfilt, medfilt


def bandpass(
    x: np.ndarray, fs: float, low: float = 0.5, high: float = 45.0, order: int = 4
) -> np.ndarray:
    nyq = 0.5 * fs
    high = min(high, nyq * 0.99)
    b, a = butter(order, [low / nyq, high / nyq], btype="band")
    filtered: np.ndarray = filtfilt(b, a, x).astype(np.float64)
    return filtered


def _odd(n: int) -> int:
    return n if n % 2 == 1 else n + 1


def baseline_wander(x: np.ndarray, fs: float) -> np.ndarray:
    k1 = _odd(max(3, int(round(0.2 * fs))))
    k2 = _odd(max(3, int(round(0.6 * fs))))
    stage1 = medfilt(x, kernel_size=k1)
    estimate: np.ndarray = medfilt(stage1, kernel_size=k2).astype(np.float64)
    return estimate


def normalize(x: np.ndarray) -> np.ndarray:
    std = float(np.std(x))
    if std < 1e-8:
        zeros: np.ndarray = np.zeros_like(x)
        return zeros
    normalised: np.ndarray = ((x - float(np.mean(x))) / std).astype(np.float64)
    return normalised


class SignalFrontEnd:
    def __init__(self, fs: float, low: float = 0.5, high: float = 45.0, order: int = 4) -> None:
        self.fs = fs
        self.low = low
        self.high = high
        self.order = order

    def forward(self, x: np.ndarray) -> np.ndarray:
        detrended = x - baseline_wander(x, self.fs)
        filtered = bandpass(detrended, self.fs, self.low, self.high, self.order)
        return normalize(filtered)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)
