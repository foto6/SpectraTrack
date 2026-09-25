from __future__ import annotations

from dataclasses import dataclass, field
from collections import deque
from typing import Deque, Tuple

BBox = Tuple[float, float, float, float]


@dataclass(slots=True)
class Detection:
    bbox: BBox
    score: float
    class_id: int
    label: str

    @property
    def center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) * 0.5, (y1 + y2) * 0.5)


@dataclass
class Track:
    track_id: int
    bbox: BBox
    score: float
    class_id: int
    label: str
    age: int = 1
    hits: int = 1
    missed: int = 0
    vx: float = 0.0
    vy: float = 0.0
    confirmed: bool = False
    last_detection_score: float = 0.0
    recoveries: int = 0
    history: Deque[tuple[int, int]] = field(default_factory=lambda: deque(maxlen=64))

    @property
    def center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) * 0.5, (y1 + y2) * 0.5)

    @property
    def width(self) -> float:
        return max(0.0, self.bbox[2] - self.bbox[0])

    @property
    def height(self) -> float:
        return max(0.0, self.bbox[3] - self.bbox[1])

    @property
    def quality(self) -> float:
        maturity = min(1.0, self.hits / 8.0)
        score_term = max(0.0, min(1.0, self.last_detection_score or self.score))
        loss_penalty = 1.0 / (1.0 + self.missed * 0.35)
        confirmed_factor = 1.0 if self.confirmed else 0.72
        return max(0.0, min(1.0, maturity * 0.35 + score_term * 0.65)) * loss_penalty * confirmed_factor

    @property
    def lifecycle(self) -> str:
        if not self.confirmed:
            return "TENTATIVE"
        if self.missed > 0:
            return "PREDICTED"
        return "TRACKED"
