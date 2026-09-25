from __future__ import annotations

import cv2
import numpy as np

from .types import BBox, Track


def grabcut_mask(frame: np.ndarray, bbox: BBox, iterations: int = 2) -> np.ndarray | None:
    """Classical foreground mask initialized by a tracked box.

    This is intentionally not presented as neural segmentation.  It is a local
    visual aid and can fail when foreground/background appearance is similar.
    """
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = bbox
    xa = max(1, min(w - 2, int(round(x1))))
    ya = max(1, min(h - 2, int(round(y1))))
    xb = max(xa + 1, min(w - 1, int(round(x2))))
    yb = max(ya + 1, min(h - 1, int(round(y2))))
    rw, rh = xb - xa, yb - ya
    if rw < 12 or rh < 12:
        return None

    mask = np.zeros((h, w), dtype=np.uint8)
    bgd = np.zeros((1, 65), np.float64)
    fgd = np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(frame, mask, (xa, ya, rw, rh), bgd, fgd, max(1, int(iterations)), cv2.GC_INIT_WITH_RECT)
    except cv2.error:
        return None
    return np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)


class TargetMaskCache:
    def __init__(self, update_every: int = 8) -> None:
        self.update_every = max(1, int(update_every))
        self.track_id: int | None = None
        self.frame_index = -1
        self.mask: np.ndarray | None = None

    def clear(self) -> None:
        self.track_id = None
        self.frame_index = -1
        self.mask = None

    def update(self, frame: np.ndarray, track: Track | None, frame_index: int) -> np.ndarray | None:
        if track is None:
            self.clear()
            return None
        needs_update = (
            self.track_id != track.track_id
            or self.mask is None
            or frame_index - self.frame_index >= self.update_every
        )
        if needs_update:
            self.track_id = track.track_id
            self.frame_index = frame_index
            self.mask = grabcut_mask(frame, track.bbox)
        return self.mask
