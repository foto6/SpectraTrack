from __future__ import annotations

from dataclasses import dataclass
from math import atan, degrees, radians, tan


@dataclass(slots=True)
class CameraGeometry:
    """Pinhole helpers using only a user-supplied horizontal FOV.

    Results are angular image geometry, not physical range.  Metric distance is
    intentionally not inferred from one uncalibrated RGB camera.
    """

    horizontal_fov_deg: float | None = None

    def valid(self) -> bool:
        return self.horizontal_fov_deg is not None and 1.0 < self.horizontal_fov_deg < 179.0

    def focal_px(self, frame_width: int) -> float | None:
        if not self.valid() or frame_width <= 1:
            return None
        return (frame_width * 0.5) / tan(radians(float(self.horizontal_fov_deg)) * 0.5)

    def bearing_offset_deg(self, x_px: float, frame_width: int) -> float | None:
        fx = self.focal_px(frame_width)
        if fx is None:
            return None
        return degrees(atan((float(x_px) - frame_width * 0.5) / fx))

    def angular_rate_deg_s(self, x_px: float, vx_px_per_frame: float, fps: float, frame_width: int) -> float | None:
        if fps <= 0:
            return None
        now = self.bearing_offset_deg(x_px, frame_width)
        prev = self.bearing_offset_deg(x_px - vx_px_per_frame, frame_width)
        if now is None or prev is None:
            return None
        return (now - prev) * fps
