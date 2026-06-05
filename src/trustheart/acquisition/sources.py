from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np

from ..afe.conditioning import SignalFrontEnd
from ..afe.segment import center_resample
from ..instrument.types import Beat

AAMI_MAP: dict[str, str] = {
    "N": "N",
    "L": "N",
    "R": "N",
    "e": "N",
    "j": "N",
    "A": "S",
    "a": "S",
    "J": "S",
    "S": "S",
    "V": "V",
    "E": "V",
    "F": "F",
    "/": "Q",
    "f": "Q",
    "Q": "Q",
}


@runtime_checkable
class EcgSource(Protocol):
    n_classes: int

    def beats(self) -> list[Beat]: ...


class WfdbSource:
    def __init__(
        self, root: str | Path, fs: float = 360.0, records: list[str] | None = None
    ) -> None:
        self.root = Path(root)
        self.fs = fs
        self.records = records
        self.n_classes = 5
        self._classes = ("N", "S", "V", "F", "Q")
        self._front = SignalFrontEnd(fs)

    def _list_records(self) -> list[str]:
        if self.records is not None:
            return self.records
        return sorted(p.stem for p in self.root.glob("*.hea"))

    def beats(self) -> list[Beat]:
        import wfdb

        index = {name: i for i, name in enumerate(self._classes)}
        half = int(round(0.36 * self.fs))
        out: list[Beat] = []
        for name in self._list_records():
            record = wfdb.rdrecord(str(self.root / name))
            channel = np.asarray(record.p_signal[:, 0], dtype=np.float64)
            conditioned = self._front(channel)
            ann = wfdb.rdann(str(self.root / name), "atr")
            peaks = np.asarray(ann.sample, dtype=np.int64)
            symbols = list(ann.symbol)
            for r_index, symbol in zip(peaks, symbols, strict=False):
                group = AAMI_MAP.get(symbol)
                lo = int(r_index) - half
                hi = int(r_index) + half
                if group is None or lo < 0 or hi > conditioned.shape[0]:
                    continue
                beat = center_resample(conditioned[lo:hi])
                out.append(Beat(signal=beat, label=index[group], subject=name))
        return out
