from __future__ import annotations

from dataclasses import dataclass
import os
import time

import cv2
import numpy as np


@dataclass(slots=True)
class CaptureConfig:
    width: int = 0
    height: int = 0
    fps: float = 0.0
    backend: str = "auto"
    buffer_size: int = 1
    reconnect_attempts: int = 3
    reconnect_delay_ms: int = 250


@dataclass(slots=True)
class CaptureRead:
    ok: bool
    frame: np.ndarray | None
    reconnected: bool = False


class RobustCapture:
    """Small synchronous capture wrapper with bounded webcam reconnects."""

    def __init__(self, source: int | str, config: CaptureConfig | None = None) -> None:
        self.source = source
        self.config = config or CaptureConfig()
        self.cap: cv2.VideoCapture | None = None
        self.open_count = 0
        self.read_failures = 0
        self.reconnects = 0
        self._reconnect_budget = max(0, int(self.config.reconnect_attempts))
        self._open()

    @property
    def is_camera(self) -> bool:
        return isinstance(self.source, int)

    def _backend_flag(self) -> int:
        name = self.config.backend.lower()
        if name == "auto":
            if os.name == "nt" and self.is_camera:
                return cv2.CAP_DSHOW
            return cv2.CAP_ANY
        if name == "dshow":
            if os.name != "nt":
                raise ValueError("DirectShow backend is only available on Windows")
            return cv2.CAP_DSHOW
        if name == "msmf":
            if os.name != "nt":
                raise ValueError("MSMF backend is only available on Windows")
            return cv2.CAP_MSMF
        raise ValueError(f"Unsupported capture backend: {self.config.backend}")

    def _open(self) -> bool:
        if self.cap is not None:
            self.cap.release()
        backend = self._backend_flag()
        if self.is_camera:
            self.cap = cv2.VideoCapture(self.source, backend)
        elif backend == cv2.CAP_ANY:
            self.cap = cv2.VideoCapture(self.source)
        else:
            self.cap = cv2.VideoCapture(self.source, backend)
        self.open_count += 1
        if not self.cap.isOpened():
            return False

        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, max(1, int(self.config.buffer_size)))
        if self.is_camera:
            if self.config.width > 0:
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(self.config.width))
            if self.config.height > 0:
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(self.config.height))
            if self.config.fps > 0:
                self.cap.set(cv2.CAP_PROP_FPS, float(self.config.fps))
        return True

    def is_opened(self) -> bool:
        return bool(self.cap is not None and self.cap.isOpened())

    def actual_properties(self) -> dict[str, float | int | str]:
        if self.cap is None:
            return {"width": 0, "height": 0, "fps": 0.0, "backend": self.config.backend}
        return {
            "width": int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0),
            "height": int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0),
            "fps": round(float(self.cap.get(cv2.CAP_PROP_FPS) or 0.0), 3),
            "backend": self.config.backend,
        }

    def read(self) -> CaptureRead:
        if self.cap is None:
            return CaptureRead(False, None)
        ok, frame = self.cap.read()
        if ok and frame is not None:
            return CaptureRead(True, frame)

        self.read_failures += 1
        if not self.is_camera:
            return CaptureRead(False, None)

        while self.reconnects < self._reconnect_budget:
            self.reconnects += 1
            self.release()
            time.sleep(max(0, self.config.reconnect_delay_ms) / 1000.0)
            if not self._open():
                continue
            ok, frame = self.cap.read()
            if ok and frame is not None:
                return CaptureRead(True, frame, reconnected=True)
            self.read_failures += 1
        return CaptureRead(False, None)

    def release(self) -> None:
        if self.cap is not None:
            self.cap.release()

    def stats(self) -> dict[str, int]:
        return {
            "open_count": self.open_count,
            "read_failures": self.read_failures,
            "reconnects": self.reconnects,
        }
