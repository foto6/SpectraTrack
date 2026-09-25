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



def crossvideo_descriptor(
    frame_bgr: np.ndarray,
    bbox: tuple[float, float, float, float],
) -> tuple[float, ...] | None:
    """Richer non-biometric appearance descriptor for cross-video candidates.

    It combines a 2x2 spatial HSV histogram with grayscale and edge-orientation
    histograms. It is intentionally generic: no face detector, face embedding,
    biometric template, or identity model is used.
    """
    h, w = frame_bgr.shape[:2]
    x1, y1, x2, y2 = bbox
    x1 = max(0, min(w - 1, int(round(x1))))
    y1 = max(0, min(h - 1, int(round(y1))))
    x2 = max(x1 + 1, min(w, int(round(x2))))
    y2 = max(y1 + 1, min(h, int(round(y2))))
    crop = frame_bgr[y1:y2, x1:x2]
    if crop.size == 0 or crop.shape[0] < 8 or crop.shape[1] < 8:
        return None

    ch, cw = crop.shape[:2]
    mx, my = max(1, int(cw * 0.08)), max(1, int(ch * 0.08))
    if cw - 2 * mx >= 8 and ch - 2 * my >= 8:
        crop = crop[my:ch - my, mx:cw - mx]

    crop = cv2.resize(crop, (96, 96), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    features: list[np.ndarray] = []
    for gy in range(2):
        for gx in range(2):
            y0, y1s = gy * 48, (gy + 1) * 48
            x0, x1s = gx * 48, (gx + 1) * 48
            cell = hsv[y0:y1s, x0:x1s]
            hist = cv2.calcHist([cell], [0, 1], None, [8, 4], [0, 180, 0, 256]).astype(np.float32).ravel()
            hist /= max(float(np.linalg.norm(hist)), 1e-12)
            features.append(hist)

    gray_hist = cv2.calcHist([gray], [0], None, [16], [0, 256]).astype(np.float32).ravel()
    gray_hist /= max(float(np.linalg.norm(gray_hist)), 1e-12)
    features.append(gray_hist)

    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    magnitude, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)
    bins = np.floor(angle / 45.0).astype(np.int32) % 8
    orient = np.zeros(8, dtype=np.float32)
    for index in range(8):
        orient[index] = float(magnitude[bins == index].sum())
    orient /= max(float(np.linalg.norm(orient)), 1e-12)
    features.append(orient)

    vector = np.concatenate(features).astype(np.float32)
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-12:
        return None
    vector /= norm
    return tuple(float(v) for v in vector)
