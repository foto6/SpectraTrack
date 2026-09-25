from __future__ import annotations

import cv2
import numpy as np

from .types import Detection


def appearance_descriptor(frame_bgr: np.ndarray, bbox: tuple[float, float, float, float]) -> tuple[float, ...] | None:
    h, w = frame_bgr.shape[:2]
    x1, y1, x2, y2 = bbox
    x1 = max(0, min(w - 1, int(round(x1))))
    y1 = max(0, min(h - 1, int(round(y1))))
    x2 = max(x1 + 1, min(w, int(round(x2))))
    y2 = max(y1 + 1, min(h, int(round(y2))))
    crop = frame_bgr[y1:y2, x1:x2]
    if crop.size == 0 or crop.shape[0] < 4 or crop.shape[1] < 4:
        return None

    # Use the central region to reduce background contamination around a detector box.
    ch, cw = crop.shape[:2]
    mx, my = int(cw * 0.12), int(ch * 0.12)
    if cw - 2 * mx >= 4 and ch - 2 * my >= 4:
        crop = crop[my:ch-my, mx:cw-mx]

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [12, 4], [0, 180, 0, 256]).astype(np.float32).ravel()
    norm = float(np.linalg.norm(hist))
    if norm <= 1e-12:
        return None
    hist /= norm
    return tuple(float(x) for x in hist)


def attach_appearance(frame_bgr: np.ndarray, detections: list[Detection]) -> None:
    for det in detections:
        det.appearance = appearance_descriptor(frame_bgr, det.bbox)
