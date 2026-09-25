from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(slots=True)
class FrameQuality:
    brightness: float
    contrast: float
    sharpness: float
    dark_fraction: float

    @property
    def low_light(self) -> bool:
        return self.brightness < 55.0 or self.dark_fraction > 0.55

    @property
    def blurred(self) -> bool:
        return self.sharpness < 55.0

    @property
    def low_contrast(self) -> bool:
        return self.contrast < 28.0


def measure_frame_quality(frame_bgr: np.ndarray) -> FrameQuality:
    if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
        raise ValueError("Expected BGR image with shape HxWx3")
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    # Downsample only for inexpensive statistics; blur metric remains representative.
    h, w = gray.shape
    if max(h, w) > 960:
        scale = 960.0 / max(h, w)
        gray = cv2.resize(gray, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA)
    brightness = float(np.mean(gray))
    contrast = float(np.std(gray))
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    dark_fraction = float(np.mean(gray < 45))
    return FrameQuality(brightness, contrast, sharpness, dark_fraction)
