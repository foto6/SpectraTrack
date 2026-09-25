from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import statistics
import time


@dataclass
class StageTimer:
    max_samples: int = 240
    samples_ms: dict[str, deque[float]] = field(default_factory=dict)

    def add(self, name: str, elapsed_ms: float) -> None:
        self.samples_ms.setdefault(name, deque(maxlen=self.max_samples)).append(float(elapsed_ms))

    def measure(self, name: str):
        return _TimerContext(self, name)

    def latest(self, name: str) -> float:
        q = self.samples_ms.get(name)
        return q[-1] if q else 0.0

    def average(self, name: str) -> float:
        q = self.samples_ms.get(name)
        return statistics.fmean(q) if q else 0.0

    def p95(self, name: str) -> float:
        q = self.samples_ms.get(name)
        if not q:
            return 0.0
        vals = sorted(q)
        idx = min(len(vals) - 1, max(0, int(round((len(vals) - 1) * 0.95))))
        return vals[idx]


class _TimerContext:
    def __init__(self, owner: StageTimer, name: str) -> None:
        self.owner = owner
        self.name = name
        self.start = 0.0

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.owner.add(self.name, (time.perf_counter() - self.start) * 1000.0)
