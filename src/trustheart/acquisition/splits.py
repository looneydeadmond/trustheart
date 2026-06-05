from __future__ import annotations

from ..instrument.types import Beat

DE_CHAZAL_TRAIN: tuple[str, ...] = (
    "101",
    "106",
    "108",
    "109",
    "112",
    "114",
    "115",
    "116",
    "118",
    "119",
    "122",
    "124",
    "201",
    "203",
    "205",
    "207",
    "208",
    "209",
    "215",
    "220",
    "223",
    "230",
)
DE_CHAZAL_TEST: tuple[str, ...] = (
    "100",
    "103",
    "105",
    "111",
    "113",
    "117",
    "121",
    "123",
    "200",
    "202",
    "210",
    "212",
    "213",
    "214",
    "219",
    "221",
    "222",
    "228",
    "231",
    "232",
    "233",
    "234",
)


def subject_split(
    beats: list[Beat], train_subjects: tuple[str, ...]
) -> tuple[list[Beat], list[Beat]]:
    train = [b for b in beats if b.subject in train_subjects]
    test = [b for b in beats if b.subject not in train_subjects]
    return train, test


def kfold_subjects(subjects: list[str], folds: int, fold: int) -> tuple[list[str], list[str]]:
    unique = sorted(set(subjects))
    held = [s for i, s in enumerate(unique) if i % folds == fold]
    kept = [s for s in unique if s not in held]
    return kept, held


def fraction_split(beats: list[Beat], train_fraction: float) -> tuple[list[Beat], list[Beat]]:
    cut = int(round(len(beats) * train_fraction))
    return beats[:cut], beats[cut:]
