from __future__ import annotations

from dataclasses import dataclass
from math import atan2, degrees
from typing import Iterable

import cv2
import numpy as np

from .types import BBox


@dataclass
class MotionEstimate:
    matrix: np.ndarray
    valid: bool = False
    inlier_ratio: float = 0.0
    features: int = 0

    @classmethod
    def identity(cls) -> "MotionEstimate":
        return cls(np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64), False, 0.0, 0)

    @property
    def dx(self) -> float:
        return float(self.matrix[0, 2])

    @property
    def dy(self) -> float:
        return float(self.matrix[1, 2])

    @property
    def rotation_deg(self) -> float:
        return degrees(atan2(float(self.matrix[1, 0]), float(self.matrix[0, 0])))

    @property
    def scale(self) -> float:
        a, b = float(self.matrix[0, 0]), float(self.matrix[1, 0])
        return (a * a + b * b) ** 0.5

    def to_dict(self) -> dict[str, float | int | bool]:
        return {
            "valid": self.valid,
            "dx": self.dx,
            "dy": self.dy,
            "rotation_deg": self.rotation_deg,
            "scale": self.scale,
            "inlier_ratio": self.inlier_ratio,
            "features": self.features,
        }


class CameraMotionEstimator:
    """Robust global camera-motion estimate from sparse optical flow.

    The estimate maps points from the previous frame into the current frame.
    Track boxes can therefore be predicted in the current coordinate system
    before detector association.
    """

    def __init__(
        self,
        max_corners: int = 320,
        min_features: int = 16,
        min_inlier_ratio: float = 0.35,
    ) -> None:
        self.max_corners = int(max_corners)
        self.min_features = int(min_features)
        self.min_inlier_ratio = float(min_inlier_ratio)
        self.prev_gray: np.ndarray | None = None

    def reset(self) -> None:
        self.prev_gray = None

    def update(self, frame: np.ndarray, exclusion_boxes: Iterable[BBox] = ()) -> MotionEstimate:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self.prev_gray is None or self.prev_gray.shape != gray.shape:
            self.prev_gray = gray
            return MotionEstimate.identity()

        h, w = gray.shape[:2]
        mask = np.full_like(self.prev_gray, 255, dtype=np.uint8)
        for box in exclusion_boxes:
            x1, y1, x2, y2 = box
            mx = max(4, int((x2 - x1) * 0.12))
            my = max(4, int((y2 - y1) * 0.12))
            xa = max(0, int(x1) - mx)
            ya = max(0, int(y1) - my)
            xb = min(w - 1, int(x2) + mx)
            yb = min(h - 1, int(y2) + my)
            if xb > xa and yb > ya:
                cv2.rectangle(mask, (xa, ya), (xb, yb), 0, -1)

        prev_pts = cv2.goodFeaturesToTrack(
            self.prev_gray,
            mask=mask,
            maxCorners=self.max_corners,
            qualityLevel=0.01,
            minDistance=18,
            blockSize=5,
        )
        if prev_pts is None or len(prev_pts) < self.min_features:
            self.prev_gray = gray
            return MotionEstimate.identity()

        curr_pts, status, error = cv2.calcOpticalFlowPyrLK(
            self.prev_gray,
            gray,
            prev_pts,
            None,
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        )
        self.prev_gray = gray
        if curr_pts is None or status is None:
            return MotionEstimate.identity()

        good = status.ravel() == 1
        if error is not None:
            good &= error.ravel() < 35.0
        p0 = prev_pts.reshape(-1, 2)[good]
        p1 = curr_pts.reshape(-1, 2)[good]
        if len(p0) < self.min_features:
            return MotionEstimate.identity()

        matrix, inliers = cv2.estimateAffinePartial2D(
            p0,
            p1,
            method=cv2.RANSAC,
            ransacReprojThreshold=2.5,
            maxIters=1500,
            confidence=0.995,
            refineIters=10,
        )
        if matrix is None:
            return MotionEstimate.identity()

        inlier_ratio = float(inliers.mean()) if inliers is not None and len(inliers) else 0.0
        estimate = MotionEstimate(np.asarray(matrix, dtype=np.float64), True, inlier_ratio, len(p0))

        sane_translation = abs(estimate.dx) <= w * 0.28 and abs(estimate.dy) <= h * 0.28
        sane_rotation = abs(estimate.rotation_deg) <= 14.0
        sane_scale = 0.82 <= estimate.scale <= 1.18
        if inlier_ratio < self.min_inlier_ratio or not sane_translation or not sane_rotation or not sane_scale:
            return MotionEstimate.identity()
        return estimate
