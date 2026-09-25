from __future__ import annotations

import cv2
import numpy as np


class VideoStabilizer:
    """Lightweight online 2D camera-motion compensator using LK optical flow."""

    def __init__(self, smoothing: float = 0.82) -> None:
        self.prev_gray: np.ndarray | None = None
        self.accum_dx = 0.0
        self.accum_dy = 0.0
        self.accum_da = 0.0
        self.smooth_dx = 0.0
        self.smooth_dy = 0.0
        self.smooth_da = 0.0
        self.smoothing = float(smoothing)

    def reset(self) -> None:
        self.prev_gray = None
        self.accum_dx = self.accum_dy = self.accum_da = 0.0
        self.smooth_dx = self.smooth_dy = self.smooth_da = 0.0

    def apply(self, frame: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self.prev_gray is None:
            self.prev_gray = gray
            return frame

        pts = cv2.goodFeaturesToTrack(self.prev_gray, maxCorners=180, qualityLevel=0.01, minDistance=24, blockSize=3)
        if pts is None or len(pts) < 8:
            self.prev_gray = gray
            return frame

        nxt, status, _ = cv2.calcOpticalFlowPyrLK(self.prev_gray, gray, pts, None)
        self.prev_gray = gray
        if nxt is None or status is None:
            return frame
        good_prev = pts[status.ravel() == 1]
        good_next = nxt[status.ravel() == 1]
        if len(good_prev) < 8:
            return frame

        mat, _ = cv2.estimateAffinePartial2D(good_prev, good_next, method=cv2.RANSAC, ransacReprojThreshold=3.0)
        if mat is None:
            return frame
        dx = float(mat[0, 2])
        dy = float(mat[1, 2])
        da = float(np.arctan2(mat[1, 0], mat[0, 0]))
        self.accum_dx += dx
        self.accum_dy += dy
        self.accum_da += da

        a = self.smoothing
        self.smooth_dx = a * self.smooth_dx + (1 - a) * self.accum_dx
        self.smooth_dy = a * self.smooth_dy + (1 - a) * self.accum_dy
        self.smooth_da = a * self.smooth_da + (1 - a) * self.accum_da
        corr_dx = self.smooth_dx - self.accum_dx
        corr_dy = self.smooth_dy - self.accum_dy
        corr_da = self.smooth_da - self.accum_da

        c, s = np.cos(corr_da), np.sin(corr_da)
        transform = np.array([[c, -s, corr_dx], [s, c, corr_dy]], dtype=np.float32)
        h, w = frame.shape[:2]
        stabilized = cv2.warpAffine(frame, transform, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        # Slight crop/zoom hides most moving borders.
        margin = int(min(w, h) * 0.018)
        if margin > 1 and w > 2 * margin and h > 2 * margin:
            crop = stabilized[margin:h-margin, margin:w-margin]
            stabilized = cv2.resize(crop, (w, h), interpolation=cv2.INTER_LINEAR)
        return stabilized
