from __future__ import annotations

from dataclasses import dataclass
from math import hypot
from typing import Iterable

from .types import Track


@dataclass(slots=True)
class SchedulerDecision:
    run_detector: bool
    reason: str
    interval: int


class AdaptiveDetectorScheduler:
    """Choose detector cadence from track health and global image motion.

    The scheduler never changes model thresholds. It only decides whether the
    current frame gets neural inference or tracker prediction-only.
    """

    PROFILE_BASE = {"quality": 1, "balanced": 2, "speed": 3}

    def __init__(
        self,
        profile: str = "balanced",
        max_interval: int = 4,
        low_quality: float = 0.55,
        high_quality: float = 0.82,
        high_camera_motion_px: float = 18.0,
    ) -> None:
        if profile not in self.PROFILE_BASE:
            raise ValueError(f"Unknown profile: {profile}")
        self.profile = profile
        self.max_interval = max(1, int(max_interval))
        self.low_quality = float(low_quality)
        self.high_quality = float(high_quality)
        self.high_camera_motion_px = float(high_camera_motion_px)
        self.frames_since_detection = 10_000

    def mark_detector_ran(self) -> None:
        self.frames_since_detection = 0

    def mark_skipped(self) -> None:
        self.frames_since_detection += 1

    def decide(
        self,
        tracks: Iterable[Track],
        camera_motion: tuple[float, float] | None = None,
        frame_low_light: bool = False,
        frame_blurred: bool = False,
    ) -> SchedulerDecision:
        tracks = list(tracks)
        base = self.PROFILE_BASE[self.profile]

        if self.frames_since_detection >= self.max_interval:
            return SchedulerDecision(True, "max_interval", 1)

        if not tracks:
            return SchedulerDecision(True, "no_tracks", 1)

        if any(t.missed > 0 for t in tracks):
            return SchedulerDecision(True, "track_missing", 1)

        qualities = [t.quality for t in tracks]
        mean_quality = sum(qualities) / len(qualities)
        if mean_quality < self.low_quality:
            return SchedulerDecision(True, "low_track_quality", 1)

        if camera_motion is not None:
            if hypot(camera_motion[0], camera_motion[1]) >= self.high_camera_motion_px:
                return SchedulerDecision(True, "high_camera_motion", 1)

        # Bad source frames are often poor inference candidates. Do not infer more
        # often merely because the image is dark/blurred; keep the normal cadence.
        interval = base
        if mean_quality >= self.high_quality and not frame_low_light and not frame_blurred:
            interval = min(self.max_interval, base + 1)

        due = self.frames_since_detection >= interval - 1
        return SchedulerDecision(due, "cadence" if due else "stable_prediction", interval)
