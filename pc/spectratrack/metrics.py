from __future__ import annotations

from collections import defaultdict, deque
from contextlib import contextmanager
from time import perf_counter
from typing import Iterator


class RollingProfiler:
    def __init__(self, window: int = 120) -> None:
        self.window = int(window)
        self._samples: dict[str, deque[float]] = defaultdict(lambda: deque(maxlen=self.window))

    @contextmanager
    def measure(self, name: str) -> Iterator[None]:
        start = perf_counter()
        try:
            yield
        finally:
            self.add(name, (perf_counter() - start) * 1000.0)

    def add(self, name: str, value_ms: float) -> None:
        self._samples[name].append(float(value_ms))

    def average_ms(self, name: str) -> float:
        values = self._samples.get(name)
        return sum(values) / len(values) if values else 0.0

    def p95_ms(self, name: str) -> float:
        values = list(self._samples.get(name, ()))
        if not values:
            return 0.0
        values.sort()
        return values[int(round((len(values) - 1) * 0.95))]

    def summary(self) -> dict[str, float]:
        out: dict[str, float] = {}
        for name in sorted(self._samples):
            out[f"{name}_avg_ms"] = self.average_ms(name)
            out[f"{name}_p95_ms"] = self.p95_ms(name)
        return out
