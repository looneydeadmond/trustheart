from __future__ import annotations

import numpy as np
from scipy.signal import butter, filtfilt


def _bandpass_qrs(x: np.ndarray, fs: float) -> np.ndarray:
    nyq = 0.5 * fs
    b, a = butter(2, [5.0 / nyq, min(15.0, nyq * 0.99) / nyq], btype="band")
    filtered: np.ndarray = filtfilt(b, a, x).astype(np.float64)
    return filtered


def pan_tompkins(x: np.ndarray, fs: float) -> np.ndarray:
    filtered = _bandpass_qrs(x, fs)
    derivative = np.gradient(filtered)
    squared = derivative**2
    window = max(1, int(round(0.150 * fs)))
    integrated = np.convolve(squared, np.ones(window) / window, mode="same")

    threshold = 0.5 * float(np.mean(integrated)) + 0.5 * float(np.std(integrated))
    refractory = int(round(0.2 * fs))
    peaks: list[int] = []
    last = -refractory
    for i in range(1, len(integrated) - 1):
        if (
            integrated[i] > threshold
            and integrated[i] >= integrated[i - 1]
            and integrated[i] > integrated[i + 1]
            and i - last > refractory
        ):
            lo = max(0, i - window)
            hi = min(len(x), i + window)
            local = lo + int(np.argmax(np.abs(x[lo:hi])))
            peaks.append(local)
            last = i
    return np.asarray(peaks, dtype=np.int64)
