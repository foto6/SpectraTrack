from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
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
    association_score: float = 0.0
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
    def speed_px_per_frame(self) -> float:
        return (self.vx * self.vx + self.vy * self.vy) ** 0.5

    @property
    def state(self) -> str:
        if self.missed > 0:
            return "COAST"
        return "LOCK" if self.confirmed else "TENTATIVE"
