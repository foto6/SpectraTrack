from __future__ import annotations

import json
from dataclasses import dataclass
from math import atan, degrees
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


@dataclass
class CameraCalibration:
    image_width: int
    image_height: int
    camera_matrix: np.ndarray
    dist_coeffs: np.ndarray
    rms: float = 0.0
    used_images: int = 0
    mean_reprojection_error_px: float = 0.0

    def __post_init__(self) -> None:
        self.camera_matrix = np.asarray(self.camera_matrix, dtype=np.float64).reshape(3, 3)
        self.dist_coeffs = np.asarray(self.dist_coeffs, dtype=np.float64).reshape(-1)
        if self.image_width <= 0 or self.image_height <= 0:
            raise ValueError("Calibration image dimensions must be positive")
        if self.camera_matrix[0, 0] <= 0 or self.camera_matrix[1, 1] <= 0:
            raise ValueError("Calibration focal lengths must be positive")

    @property
    def hfov_deg(self) -> float:
        fx = float(self.camera_matrix[0, 0])
        return degrees(2.0 * atan(self.image_width / (2.0 * fx)))

    @property
    def vfov_deg(self) -> float:
        fy = float(self.camera_matrix[1, 1])
        return degrees(2.0 * atan(self.image_height / (2.0 * fy)))

    def scaled_camera_matrix(self, width: int, height: int) -> np.ndarray:
        if width <= 0 or height <= 0:
            raise ValueError("Frame dimensions must be positive")
        sx = width / self.image_width
        sy = height / self.image_height
        k = self.camera_matrix.copy()
        k[0, 0] *= sx
        k[0, 2] *= sx
        k[1, 1] *= sy
        k[1, 2] *= sy
        return k

    def to_dict(self) -> dict:
        return {
            "image_width": int(self.image_width),
            "image_height": int(self.image_height),
            "camera_matrix": self.camera_matrix.tolist(),
            "dist_coeffs": self.dist_coeffs.tolist(),
            "rms": float(self.rms),
            "used_images": int(self.used_images),
            "mean_reprojection_error_px": float(self.mean_reprojection_error_px),
            "hfov_deg": float(self.hfov_deg),
            "vfov_deg": float(self.vfov_deg),
        }

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")

    @classmethod
    def from_dict(cls, data: dict) -> "CameraCalibration":
        return cls(
            int(data["image_width"]),
            int(data["image_height"]),
            np.asarray(data["camera_matrix"], dtype=np.float64),
            np.asarray(data["dist_coeffs"], dtype=np.float64),
            float(data.get("rms", 0.0)),
            int(data.get("used_images", 0)),
            float(data.get("mean_reprojection_error_px", 0.0)),
        )

    @classmethod
    def load(cls, path: str | Path) -> "CameraCalibration":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


class Undistorter:
    def __init__(self, calibration: CameraCalibration) -> None:
        self.calibration = calibration
        self._cache: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}

    def apply(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        key = (w, h)
        maps = self._cache.get(key)
        if maps is None:
            k = self.calibration.scaled_camera_matrix(w, h)
            map1, map2 = cv2.initUndistortRectifyMap(
                k,
                self.calibration.dist_coeffs,
                None,
                k,
                (w, h),
                cv2.CV_32FC1,
            )
            maps = (map1, map2)
            self._cache[key] = maps
        return cv2.remap(frame, maps[0], maps[1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)


def calibrate_checkerboard(
    image_paths: Iterable[str | Path],
    cols: int,
    rows: int,
    square_size: float = 1.0,
    min_images: int = 5,
) -> CameraCalibration:
    cols, rows = int(cols), int(rows)
    if cols < 3 or rows < 3:
        raise ValueError("Checkerboard must have at least 3x3 inner corners")
    if square_size <= 0:
        raise ValueError("square_size must be positive")

    obj_template = np.zeros((rows * cols, 3), np.float32)
    obj_template[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    obj_template[:, :2] *= float(square_size)

    obj_points: list[np.ndarray] = []
    img_points: list[np.ndarray] = []
    image_size: tuple[int, int] | None = None

    for path in image_paths:
        image = cv2.imread(str(path))
        if image is None:
            continue
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        size = (gray.shape[1], gray.shape[0])
        if image_size is None:
            image_size = size
        elif size != image_size:
            raise ValueError(f"Mixed calibration image sizes: expected {image_size}, got {size} for {path}")

        if hasattr(cv2, "findChessboardCornersSB"):
            flags = cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_EXHAUSTIVE
            found, corners = cv2.findChessboardCornersSB(gray, (cols, rows), flags=flags)
        else:
            found, corners = cv2.findChessboardCorners(gray, (cols, rows))
            if found:
                corners = cv2.cornerSubPix(
                    gray, corners, (11, 11), (-1, -1),
                    (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.001),
                )
        if found:
            obj_points.append(obj_template.copy())
            img_points.append(np.asarray(corners, dtype=np.float32))

    if image_size is None or len(obj_points) < min_images:
        raise ValueError(f"Need at least {min_images} usable checkerboard images; got {len(obj_points)}")

    rms, k, dist, rvecs, tvecs = cv2.calibrateCamera(
        obj_points,
        img_points,
        image_size,
        None,
        None,
    )

    errors = []
    for obj, img, rvec, tvec in zip(obj_points, img_points, rvecs, tvecs):
        projected, _ = cv2.projectPoints(obj, rvec, tvec, k, dist)
        err = cv2.norm(img, projected, cv2.NORM_L2) / max(len(projected), 1)
        errors.append(float(err))

    return CameraCalibration(
        image_width=image_size[0],
        image_height=image_size[1],
        camera_matrix=k,
        dist_coeffs=dist.reshape(-1),
        rms=float(rms),
        used_images=len(obj_points),
        mean_reprojection_error_px=float(np.mean(errors)) if errors else 0.0,
    )
