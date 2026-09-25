from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path


@dataclass(slots=True)
class CameraCalibration:
    width: int
    height: int
    hfov_deg: float
    vfov_deg: float | None = None
    name: str = "camera"

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError("width and height must be positive")
        if not 1.0 <= self.hfov_deg < 179.0:
            raise ValueError("hfov_deg must be in [1, 179)")
        if self.vfov_deg is None:
            aspect = self.height / self.width
            hfov = math.radians(self.hfov_deg)
            self.vfov_deg = math.degrees(2.0 * math.atan(math.tan(hfov / 2.0) * aspect))
        if not 1.0 <= float(self.vfov_deg) < 179.0:
            raise ValueError("vfov_deg must be in [1, 179)")

    @classmethod
    def from_json(cls, path: str | Path) -> "CameraCalibration":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(**data)

    def to_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps({
            "width": self.width,
            "height": self.height,
            "hfov_deg": self.hfov_deg,
            "vfov_deg": self.vfov_deg,
            "name": self.name,
        }, indent=2), encoding="utf-8")

    def angular_offset_deg(self, x_px: float, y_px: float) -> tuple[float, float]:
        nx = (x_px - self.width / 2.0) / (self.width / 2.0)
        ny = (y_px - self.height / 2.0) / (self.height / 2.0)
        yaw = nx * (self.hfov_deg / 2.0)
        pitch = -ny * (float(self.vfov_deg) / 2.0)
        return yaw, pitch

    def angular_size_deg(self, width_px: float, height_px: float) -> tuple[float, float]:
        return (
            max(0.0, width_px) / self.width * self.hfov_deg,
            max(0.0, height_px) / self.height * float(self.vfov_deg),
        )
