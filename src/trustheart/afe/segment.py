from __future__ import annotations

import numpy as np
from scipy.signal import resample

BEAT_LENGTH = 256


def frame_beats(
    signal: np.ndarray,
    peaks: np.ndarray,
    fs: float,
    length: int = BEAT_LENGTH,
    half_window_s: float = 0.36,
) -> np.ndarray:
    half = int(round(half_window_s * fs))
    beats: list[np.ndarray] = []
    for r in peaks:
        lo = int(r) - half
        hi = int(r) + half
        if lo < 0 or hi > len(signal):
            continue
        segment = signal[lo:hi]
        resampled = resample(segment, length)
        beats.append(np.asarray(resampled, dtype=np.float32))
    if not beats:
        return np.empty((0, length), dtype=np.float32)
    return np.stack(beats)


def center_resample(segment: np.ndarray, length: int = BEAT_LENGTH) -> np.ndarray:
    return np.asarray(resample(segment, length), dtype=np.float32)
