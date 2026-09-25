from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import os
import statistics
import time


def _p95(values: list[float] | deque[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * 0.95))))
    return float(ordered[idx])


@dataclass
class StageTimer:
    max_samples: int = 240
    samples_ms: dict[str, deque[float]] = field(default_factory=dict)
    _total_ms: dict[str, float] = field(default_factory=dict, init=False, repr=False)
    _count: dict[str, int] = field(default_factory=dict, init=False, repr=False)
    _maximum_ms: dict[str, float] = field(default_factory=dict, init=False, repr=False)

    def add(self, name: str, elapsed_ms: float) -> None:
        value = float(elapsed_ms)
        self.samples_ms.setdefault(name, deque(maxlen=self.max_samples)).append(value)
        self._total_ms[name] = self._total_ms.get(name, 0.0) + value
        self._count[name] = self._count.get(name, 0) + 1
        self._maximum_ms[name] = max(value, self._maximum_ms.get(name, value))

    def measure(self, name: str):
        return _TimerContext(self, name)

    def latest(self, name: str) -> float:
        q = self.samples_ms.get(name)
        return q[-1] if q else 0.0

    def average(self, name: str) -> float:
        """Average over the bounded recent sample window."""
        q = self.samples_ms.get(name)
        return statistics.fmean(q) if q else 0.0

    def run_average(self, name: str) -> float:
        """Average over every sample observed during this process run."""
        count = self._count.get(name, 0)
        return self._total_ms.get(name, 0.0) / count if count else 0.0

    def count(self, name: str) -> int:
        return self._count.get(name, 0)

    def maximum(self, name: str) -> float:
        return self._maximum_ms.get(name, 0.0)

    def p95(self, name: str) -> float:
        """P95 over the bounded recent sample window."""
        q = self.samples_ms.get(name)
        return _p95(q) if q else 0.0

    def summary(self, name: str) -> dict[str, float | int]:
        return {
            "samples": self.count(name),
            "avg": round(self.run_average(name), 3),
            "p95_recent": round(self.p95(name), 3),
            "max": round(self.maximum(name), 3),
        }

    def summaries(self, names: list[str] | tuple[str, ...] | None = None) -> dict[str, dict[str, float | int]]:
        selected = names if names is not None else sorted(self._count)
        return {name: self.summary(name) for name in selected if self.count(name) > 0}


@dataclass
class ProcessCpuSampler:
    """Dependency-free process CPU sampler.

    Percent is normalized to total logical CPU capacity, so 100% means the
    process consumed all logical CPUs during the sample interval.
    """

    max_samples: int = 240
    min_interval_s: float = 0.5
    samples_pct: deque[float] = field(default_factory=deque, init=False)
    logical_cpus: int = field(default_factory=lambda: max(1, os.cpu_count() or 1), init=False)
    _last_wall: float = field(default=0.0, init=False, repr=False)
    _last_cpu: float = field(default=0.0, init=False, repr=False)
    _total_pct: float = field(default=0.0, init=False, repr=False)
    _count: int = field(default=0, init=False, repr=False)
    _maximum_pct: float = field(default=0.0, init=False, repr=False)

    def __post_init__(self) -> None:
        self.samples_pct = deque(maxlen=self.max_samples)
        self._last_wall = time.perf_counter()
        self._last_cpu = time.process_time()

    def sample(self, force: bool = False) -> float | None:
        now_wall = time.perf_counter()
        wall_delta = now_wall - self._last_wall
        if wall_delta <= 0.0 or (not force and wall_delta < self.min_interval_s):
            return None

        now_cpu = time.process_time()
        cpu_delta = max(0.0, now_cpu - self._last_cpu)
        value = 100.0 * cpu_delta / (wall_delta * self.logical_cpus)
        value = max(0.0, min(100.0, value))

        self.samples_pct.append(value)
        self._total_pct += value
        self._count += 1
        self._maximum_pct = max(self._maximum_pct, value)
        self._last_wall = now_wall
        self._last_cpu = now_cpu
        return value

    def summary(self) -> dict[str, float | int | str]:
        return {
            "samples": self._count,
            "avg_pct": round(self._total_pct / self._count, 3) if self._count else 0.0,
            "p95_recent_pct": round(_p95(self.samples_pct), 3),
            "max_pct": round(self._maximum_pct, 3),
            "logical_cpus": self.logical_cpus,
            "scale": "percent_of_total_logical_cpu_capacity",
        }


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
