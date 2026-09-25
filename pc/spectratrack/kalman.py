from __future__ import annotations

import numpy as np

from .types import BBox


def bbox_to_measurement(box: BBox) -> np.ndarray:
    x1, y1, x2, y2 = box
    w = max(2.0, x2 - x1)
    h = max(2.0, y2 - y1)
    return np.array([(x1 + x2) * 0.5, (y1 + y2) * 0.5, w, h], dtype=np.float64)


def measurement_to_bbox(z: np.ndarray) -> BBox:
    cx, cy, w, h = [float(v) for v in z[:4]]
    w = max(2.0, w)
    h = max(2.0, h)
    return (cx - w * 0.5, cy - h * 0.5, cx + w * 0.5, cy + h * 0.5)


class KalmanBox:
    """Small constant-velocity Kalman filter for bounding boxes.

    State is [cx, cy, w, h, vx, vy, vw, vh].  A camera-motion affine can be
    applied after the physical-motion prediction so matching stays stable while
    the camera itself pans, tilts or slightly zooms.
    """

    def __init__(self, box: BBox) -> None:
        z = bbox_to_measurement(box)
        self.x = np.zeros((8, 1), dtype=np.float64)
        self.x[:4, 0] = z
        self.P = np.diag([30.0, 30.0, 50.0, 50.0, 100.0, 100.0, 120.0, 120.0])
        self.H = np.zeros((4, 8), dtype=np.float64)
        self.H[0, 0] = self.H[1, 1] = self.H[2, 2] = self.H[3, 3] = 1.0
        self.R = np.diag([9.0, 9.0, 20.0, 20.0])

    @property
    def bbox(self) -> BBox:
        return measurement_to_bbox(self.x[:, 0])

    @property
    def velocity(self) -> tuple[float, float]:
        return float(self.x[4, 0]), float(self.x[5, 0])

    def predict(self, dt: float = 1.0, camera_affine: np.ndarray | None = None) -> BBox:
        dt = float(max(0.05, min(dt, 4.0)))
        F = np.eye(8, dtype=np.float64)
        for i in range(4):
            F[i, i + 4] = dt
        q_pos = 2.5 * dt * dt
        q_vel = 7.0 * dt
        Q = np.diag([q_pos, q_pos, q_pos, q_pos, q_vel, q_vel, q_vel, q_vel])

        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q

        if camera_affine is not None:
            m = np.asarray(camera_affine, dtype=np.float64)
            if m.shape == (2, 3):
                cx, cy = self.x[0, 0], self.x[1, 0]
                self.x[0, 0] = m[0, 0] * cx + m[0, 1] * cy + m[0, 2]
                self.x[1, 0] = m[1, 0] * cx + m[1, 1] * cy + m[1, 2]

                linear = m[:, :2]
                vx, vy = self.x[4, 0], self.x[5, 0]
                self.x[4, 0] = linear[0, 0] * vx + linear[0, 1] * vy
                self.x[5, 0] = linear[1, 0] * vx + linear[1, 1] * vy

                area_scale = abs(float(np.linalg.det(linear)))
                scale = max(0.75, min(1.25, area_scale ** 0.5))
                self.x[2, 0] = max(2.0, self.x[2, 0] * scale)
                self.x[3, 0] = max(2.0, self.x[3, 0] * scale)
        return self.bbox

    def update(self, box: BBox) -> BBox:
        z = bbox_to_measurement(box).reshape(4, 1)
        innovation = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ innovation
        self.P = (np.eye(8) - K @ self.H) @ self.P
        self.x[2, 0] = max(2.0, self.x[2, 0])
        self.x[3, 0] = max(2.0, self.x[3, 0])
        return self.bbox
