from __future__ import annotations

from dataclasses import dataclass
import cv2
import numpy as np


@dataclass(slots=True)
class CameraMotion:
    dx: float = 0.0
    dy: float = 0.0
    rotation_rad: float = 0.0
    inliers: int = 0
    valid: bool = False
    affine: tuple[float, float, float, float, float, float] | None = None


class GlobalMotionEstimator:
    """Estimate global image motion between consecutive frames.

    The result is image-space motion, not physical camera pose.
    """

    def __init__(self, max_corners: int = 220) -> None:
        self.max_corners = int(max_corners)
        self.prev_gray: np.ndarray | None = None

    def reset(self) -> None:
        self.prev_gray = None

    def update(self, frame: np.ndarray) -> CameraMotion:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self.prev_gray is None:
            self.prev_gray = gray
            return CameraMotion()

        pts = cv2.goodFeaturesToTrack(
            self.prev_gray, maxCorners=self.max_corners, qualityLevel=0.01,
            minDistance=20, blockSize=3
        )
        if pts is None or len(pts) < 8:
            self.prev_gray = gray
            return CameraMotion()

        nxt, status, _ = cv2.calcOpticalFlowPyrLK(self.prev_gray, gray, pts, None)
        old = self.prev_gray
        self.prev_gray = gray
        if nxt is None or status is None:
            return CameraMotion()

        p0 = pts[status.ravel() == 1]
        p1 = nxt[status.ravel() == 1]
        if len(p0) < 8:
            return CameraMotion()

        mat, inliers = cv2.estimateAffinePartial2D(
            p0, p1, method=cv2.RANSAC, ransacReprojThreshold=3.0,
            maxIters=1200, confidence=0.99
        )
        if mat is None:
            return CameraMotion()
        count = int(inliers.sum()) if inliers is not None else len(p0)
        return CameraMotion(
            dx=float(mat[0, 2]),
            dy=float(mat[1, 2]),
            rotation_rad=float(np.arctan2(mat[1, 0], mat[0, 0])),
            inliers=count,
            valid=count >= 6,
            affine=(
                float(mat[0, 0]), float(mat[0, 1]), float(mat[0, 2]),
                float(mat[1, 0]), float(mat[1, 1]), float(mat[1, 2]),
            ),
        )
