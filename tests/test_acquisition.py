from __future__ import annotations

import numpy as np

from trustheart.acquisition.beats import BeatDataset
from trustheart.acquisition.splits import fraction_split, subject_split
from trustheart.acquisition.synthetic import SyntheticEcg, synth_beat


def test_synth_beat_is_deterministic() -> None:
    a = synth_beat(2, 5, np.random.default_rng(123))
    b = synth_beat(2, 5, np.random.default_rng(123))
    assert np.array_equal(a, b)
    assert a.shape == (256,)


def test_synthetic_source_balanced() -> None:
    beats = SyntheticEcg(n_classes=5, per_class=10, seed=0).beats()
    counts = np.bincount([b.label for b in beats], minlength=5)
    assert counts.tolist() == [10, 10, 10, 10, 10]


def test_beat_dataset_tensor_shapes() -> None:
    beats = SyntheticEcg(n_classes=3, per_class=4, seed=1).beats()
    dataset = BeatDataset(beats)
    signal, label = dataset[0]
    assert signal.shape == (1, 256)
    assert label.ndim == 0


def test_fraction_split_sizes() -> None:
    beats = SyntheticEcg(n_classes=2, per_class=10, seed=2).beats()
    train, test = fraction_split(beats, 0.8)
    assert len(train) == 16
    assert len(test) == 4


def test_subject_split_partitions_by_subject() -> None:
    beats = SyntheticEcg(n_classes=2, per_class=8, seed=3).beats()
    train, test = subject_split(beats, ("synthetic-0-0",))
    train_subjects = {b.subject for b in train}
    assert train_subjects == {"synthetic-0-0"}
