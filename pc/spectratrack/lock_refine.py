from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .types import BBox


@dataclass(slots=True)
class RefineResult:
    bbox: BBox
    valid: bool
    inliers: int = 0
    inlier_ratio: float = 0.0


class LockRefiner:
    """Sparse optical-flow refinement for one explicitly selected track."""

    def __init__(self, max_points: int = 80) -> None:
        self.max_points = int(max_points)
        self.track_id: int | None = None
        self.prev_gray: np.ndarray | None = None
        self.points: np.ndarray | None = None

    def reset(self) -> None:
        self.track_id = None
        self.prev_gray = None
        self.points = None

    def initialize(self, frame_bgr: np.ndarray, track_id: int, bbox: BBox) -> bool:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        x1, y1, x2, y2 = bbox
        x1 = max(0, min(w - 1, int(round(x1))))
        y1 = max(0, min(h - 1, int(round(y1))))
        x2 = max(x1 + 1, min(w, int(round(x2))))
        y2 = max(y1 + 1, min(h, int(round(y2))))

        mask = np.zeros_like(gray)
        margin_x = max(1, int((x2 - x1) * 0.10))
        margin_y = max(1, int((y2 - y1) * 0.10))
        ix1 = min(x2 - 1, x1 + margin_x)
        iy1 = min(y2 - 1, y1 + margin_y)
        ix2 = max(ix1 + 1, x2 - margin_x)
        iy2 = max(iy1 + 1, y2 - margin_y)
        mask[iy1:iy2, ix1:ix2] = 255

        points = cv2.goodFeaturesToTrack(
            gray,
            maxCorners=self.max_points,
            qualityLevel=0.01,
            minDistance=6,
            blockSize=3,
            mask=mask,
        )
        self.track_id = track_id
        self.prev_gray = gray
        self.points = points
        return points is not None and len(points) >= 6

    def update(self, frame_bgr: np.ndarray, track_id: int, bbox: BBox) -> RefineResult:
        if self.track_id != track_id or self.prev_gray is None or self.points is None or len(self.points) < 6:
            self.initialize(frame_bgr, track_id, bbox)
            return RefineResult(bbox, False)

        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        nxt, status, _ = cv2.calcOpticalFlowPyrLK(self.prev_gray, gray, self.points, None)
        if nxt is None or status is None:
            self.initialize(frame_bgr, track_id, bbox)
            return RefineResult(bbox, False)

        p0 = self.points[status.ravel() == 1]
        p1 = nxt[status.ravel() == 1]
        if len(p0) < 6:
            self.initialize(frame_bgr, track_id, bbox)
            return RefineResult(bbox, False)

        mat, inliers = cv2.estimateAffinePartial2D(
            p0,
            p1,
            method=cv2.RANSAC,
            ransacReprojThreshold=2.5,
            maxIters=800,
            confidence=0.99,
        )
        if mat is None:
            self.initialize(frame_bgr, track_id, bbox)
            return RefineResult(bbox, False)

        count = int(inliers.sum()) if inliers is not None else len(p0)
        ratio = count / max(len(p0), 1)
        scale = float(np.hypot(mat[0, 0], mat[1, 0]))
        rotation = float(np.arctan2(mat[1, 0], mat[0, 0]))
        valid = count >= 6 and ratio >= 0.45 and 0.85 <= scale <= 1.18 and abs(rotation) <= np.deg2rad(20.0)

        self.prev_gray = gray
        self.points = p1.reshape(-1, 1, 2)

        if not valid:
            self.initialize(frame_bgr, track_id, bbox)
            return RefineResult(bbox, False, count, ratio)

        x1, y1, x2, y2 = bbox
        corners = np.array([
            [x1, y1, 1.0],
            [x2, y1, 1.0],
            [x2, y2, 1.0],
            [x1, y2, 1.0],
        ], dtype=np.float32)
        transformed = corners @ mat.T
        xs = transformed[:, 0]
        ys = transformed[:, 1]
        refined = (float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max()))
        return RefineResult(refined, True, count, ratio)
