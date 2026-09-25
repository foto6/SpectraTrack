from __future__ import annotations

from dataclasses import dataclass, asdict
import json
from pathlib import Path

import cv2
import numpy as np


@dataclass(slots=True)
class StereoRigCalibration:
    width: int
    height: int
    fx_px: float
    baseline_m: float
    min_disparity: int = 0
    num_disparities: int = 128
    block_size: int = 5
    name: str = "stereo-rig"

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError("Stereo resolution must be positive")
        if self.fx_px <= 0 or self.baseline_m <= 0:
            raise ValueError("fx_px and baseline_m must be positive")
        if self.num_disparities <= 0 or self.num_disparities % 16 != 0:
            raise ValueError("num_disparities must be positive and divisible by 16")
        if self.block_size < 3 or self.block_size % 2 == 0:
            raise ValueError("block_size must be odd and >= 3")

    @classmethod
    def from_json(cls, path: str | Path) -> "StereoRigCalibration":
        return cls(**json.loads(Path(path).read_text(encoding="utf-8")))

    def to_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")

    def distance_from_disparity(self, disparity_px: float) -> float:
        if disparity_px <= 0:
            raise ValueError("disparity must be positive")
        return self.fx_px * self.baseline_m / disparity_px


@dataclass(slots=True)
class StereoRange:
    distance_m: float
    disparity_px: float
    valid_fraction: float


class StereoDepthEstimator:
    """Depth from a rectified stereo pair using OpenCV StereoSGBM.

    Inputs must already be rectified so corresponding points lie on the same
    horizontal scanlines. This module does not fabricate depth for monocular video.
    """

    def __init__(self, calibration: StereoRigCalibration) -> None:
        self.cal = calibration
        self.matcher = cv2.StereoSGBM_create(
            minDisparity=calibration.min_disparity,
            numDisparities=calibration.num_disparities,
            blockSize=calibration.block_size,
            P1=8 * 3 * calibration.block_size**2,
            P2=32 * 3 * calibration.block_size**2,
            disp12MaxDiff=1,
            uniquenessRatio=8,
            speckleWindowSize=80,
            speckleRange=2,
            preFilterCap=31,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
        )

    def compute_disparity(self, left_bgr: np.ndarray, right_bgr: np.ndarray) -> np.ndarray:
        if left_bgr.shape != right_bgr.shape:
            raise ValueError("left/right frames must have identical shape")
        if left_bgr.ndim != 3 or left_bgr.shape[2] != 3:
            raise ValueError("expected BGR HxWx3 images")
        h, w = left_bgr.shape[:2]
        if (w, h) != (self.cal.width, self.cal.height):
            raise ValueError(
                f"frame resolution {w}x{h} does not match stereo calibration "
                f"{self.cal.width}x{self.cal.height}"
            )
        left = cv2.cvtColor(left_bgr, cv2.COLOR_BGR2GRAY)
        right = cv2.cvtColor(right_bgr, cv2.COLOR_BGR2GRAY)
        disparity = self.matcher.compute(left, right).astype(np.float32) / 16.0
        disparity[disparity <= max(0, self.cal.min_disparity)] = np.nan
        return disparity

    def range_for_bbox(
        self,
        disparity: np.ndarray,
        bbox: tuple[float, float, float, float],
        center_fraction: float = 0.60,
        min_valid_fraction: float = 0.15,
    ) -> StereoRange | None:
        h, w = disparity.shape[:2]
        x1, y1, x2, y2 = bbox
        cx, cy = (x1 + x2) * 0.5, (y1 + y2) * 0.5
        bw, bh = max(1.0, x2 - x1), max(1.0, y2 - y1)
        half_w = bw * center_fraction * 0.5
        half_h = bh * center_fraction * 0.5
        ix1 = max(0, min(w - 1, int(cx - half_w)))
        iy1 = max(0, min(h - 1, int(cy - half_h)))
        ix2 = max(ix1 + 1, min(w, int(cx + half_w)))
        iy2 = max(iy1 + 1, min(h, int(cy + half_h)))
        roi = disparity[iy1:iy2, ix1:ix2]
        valid = roi[np.isfinite(roi) & (roi > 0)]
        fraction = float(valid.size / max(roi.size, 1))
        if valid.size == 0 or fraction < min_valid_fraction:
            return None
        # Median is robust to background leakage and SGBM outliers.
        d = float(np.median(valid))
        return StereoRange(
            distance_m=self.cal.distance_from_disparity(d),
            disparity_px=d,
            valid_fraction=fraction,
        )
