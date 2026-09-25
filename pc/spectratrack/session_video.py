from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


class SessionVideoWriter:
    """Lazy MP4 writer for the exact tracking-coordinate frame stream."""

    def __init__(self, path: str | Path, fps: float = 30.0) -> None:
        self.path = Path(path)
        self.fps = float(fps) if fps and fps > 1.0 else 30.0
        self._writer: cv2.VideoWriter | None = None
        self._size: tuple[int, int] | None = None

    def write(self, frame: np.ndarray) -> None:
        h, w = frame.shape[:2]
        size = (w, h)
        if self._writer is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(self.path), fourcc, self.fps, size)
            if not writer.isOpened():
                raise RuntimeError(f"Cannot open session video writer: {self.path}")
            self._writer = writer
            self._size = size
        elif size != self._size:
            raise RuntimeError(f"Session video frame size changed from {self._size} to {size}")
        self._writer.write(frame)

    def close(self) -> None:
        if self._writer is not None:
            self._writer.release()
            self._writer = None

    def __enter__(self) -> "SessionVideoWriter":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
