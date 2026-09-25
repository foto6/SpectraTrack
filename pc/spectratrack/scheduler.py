from __future__ import annotations

from dataclasses import dataclass
from math import ceil


@dataclass
class DetectionScheduler:
    """Choose detector cadence from measured inference cost.

    Tracking and CMC still run on skipped frames.  Adaptive mode targets a
    detector-time budget per displayed frame and uses hysteresis to avoid
    bouncing between intervals.
    """

    fixed_interval: int = 1
    adaptive: bool = False
    target_fps: float = 30.0
    max_interval: int = 4
    ema_alpha: float = 0.15

    def __post_init__(self) -> None:
        self.fixed_interval = max(1, int(self.fixed_interval))
        self.max_interval = max(1, int(self.max_interval))
        self.target_fps = max(1.0, float(self.target_fps))
        self._interval = min(self.fixed_interval, self.max_interval)
        self._detector_ms_ema = 0.0
        self._samples = 0
        self._down_votes = 0

    @property
    def interval(self) -> int:
        return self._interval

    @property
    def detector_ms_ema(self) -> float:
        return self._detector_ms_ema

    def should_detect(self, frame_index: int, force: bool = False) -> bool:
        if force:
            return True
        return frame_index <= 1 or ((frame_index - 1) % self._interval == 0)

    def observe_detector_ms(self, value_ms: float) -> int:
        value_ms = max(0.0, float(value_ms))
        if self._samples == 0:
            self._detector_ms_ema = value_ms
        else:
            a = self.ema_alpha
            self._detector_ms_ema = self._detector_ms_ema * (1.0 - a) + value_ms * a
        self._samples += 1

        if not self.adaptive:
            self._interval = min(self.fixed_interval, self.max_interval)
            return self._interval

        frame_budget_ms = 1000.0 / self.target_fps
        desired = max(1, min(self.max_interval, int(ceil(self._detector_ms_ema / max(frame_budget_ms * 0.78, 1.0)))))

        if desired > self._interval:
            self._interval = desired
            self._down_votes = 0
        elif desired < self._interval:
            self._down_votes += 1
            if self._down_votes >= 18:
                self._interval -= 1
                self._down_votes = 0
        else:
            self._down_votes = 0
        return self._interval
