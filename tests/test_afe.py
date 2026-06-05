from __future__ import annotations

import numpy as np

from trustheart.afe.conditioning import SignalFrontEnd, bandpass, normalize
from trustheart.afe.rpeak import pan_tompkins
from trustheart.afe.segment import center_resample, frame_beats


def _sine(freq: float, fs: float, seconds: float) -> np.ndarray:
    t = np.arange(0, seconds, 1.0 / fs)
    return np.sin(2.0 * np.pi * freq * t)


def test_bandpass_attenuates_out_of_band() -> None:
    fs = 360.0
    passband = bandpass(_sine(10.0, fs, 4.0), fs)
    stopband = bandpass(_sine(80.0, fs, 4.0), fs)
    assert np.var(passband) > 5.0 * np.var(stopband)


def test_normalize_zero_mean_unit_std() -> None:
    x = np.random.default_rng(0).normal(3.0, 2.0, size=512)
    z = normalize(x)
    assert abs(float(np.mean(z))) < 1e-6
    assert abs(float(np.std(z)) - 1.0) < 1e-6


def test_front_end_output_finite() -> None:
    fs = 360.0
    raw = _sine(8.0, fs, 3.0) + 0.5 * _sine(0.2, fs, 3.0)
    out = SignalFrontEnd(fs)(raw)
    assert np.isfinite(out).all()


def test_rpeak_detects_periodic_beats() -> None:
    fs = 360.0
    t = np.arange(0, 10.0, 1.0 / fs)
    signal = np.zeros_like(t)
    width = 0.012
    for centre in np.arange(0.5, 10.0, 1.0):
        signal += 5.0 * np.exp(-((t - centre) ** 2) / (2.0 * width**2))
    peaks = pan_tompkins(signal, fs)
    assert 8 <= len(peaks) <= 12


def test_frame_beats_shape() -> None:
    fs = 360.0
    signal = np.random.default_rng(1).normal(size=int(fs * 10))
    peaks = np.arange(200, len(signal) - 200, 360)
    beats = frame_beats(signal, peaks, fs)
    assert beats.shape[1] == 256


def test_center_resample_length() -> None:
    out = center_resample(np.ones(173), 256)
    assert out.shape == (256,)
