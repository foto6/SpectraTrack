from __future__ import annotations

import math
import time

import cv2
import numpy as np

from .enhance import crop_with_margin, upscale_preview
from .types import Track


FONT = cv2.FONT_HERSHEY_SIMPLEX


def _clip_box(box, w, h):
    x1, y1, x2, y2 = map(int, box)
    return max(0, x1), max(0, y1), min(w - 1, x2), min(h - 1, y2)


def draw_corner_box(frame: np.ndarray, box, selected: bool = False) -> None:
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = _clip_box(box, w, h)
    thickness = 2 if not selected else 3
    length = max(10, int(min(x2 - x1, y2 - y1) * 0.22))
    # Default OpenCV green/cyan-ish without hardcoding a full style framework.
    color = (255, 255, 255) if selected else (220, 220, 220)
    for a, b, c, d in [
        (x1, y1, x1 + length, y1), (x1, y1, x1, y1 + length),
        (x2, y1, x2 - length, y1), (x2, y1, x2, y1 + length),
        (x1, y2, x1 + length, y2), (x1, y2, x1, y2 - length),
        (x2, y2, x2 - length, y2), (x2, y2, x2, y2 - length),
    ]:
        cv2.line(frame, (a, b), (c, d), color, thickness, cv2.LINE_AA)


def draw_tracks(frame: np.ndarray, tracks: list[Track], selected_id: int | None) -> None:
    for tr in tracks:
        selected = tr.track_id == selected_id
        draw_corner_box(frame, tr.bbox, selected)
        x1, y1, _, _ = map(int, tr.bbox)
        text = f"T{tr.track_id:03d} {tr.label.upper()} {tr.score:.2f}"
        cv2.putText(frame, text, (max(4, x1), max(18, y1 - 7)), FONT, 0.48, (245, 245, 245), 1, cv2.LINE_AA)
        pts = list(tr.history)
        if len(pts) >= 2:
            for i in range(1, len(pts)):
                alpha = i / len(pts)
                shade = int(70 + 160 * alpha)
                cv2.line(frame, pts[i - 1], pts[i], (shade, shade, shade), 1, cv2.LINE_AA)


def draw_center_reticle(frame: np.ndarray) -> None:
    h, w = frame.shape[:2]
    cx, cy = w // 2, h // 2
    r = max(18, min(w, h) // 32)
    cv2.circle(frame, (cx, cy), r, (190, 190, 190), 1, cv2.LINE_AA)
    cv2.line(frame, (cx - r - 14, cy), (cx - r + 2, cy), (190, 190, 190), 1, cv2.LINE_AA)
    cv2.line(frame, (cx + r - 2, cy), (cx + r + 14, cy), (190, 190, 190), 1, cv2.LINE_AA)
    cv2.line(frame, (cx, cy - r - 14), (cx, cy - r + 2), (190, 190, 190), 1, cv2.LINE_AA)
    cv2.line(frame, (cx, cy + r - 2), (cx, cy + r + 14), (190, 190, 190), 1, cv2.LINE_AA)


def compose_hud(
    frame: np.ndarray,
    tracks: list[Track],
    selected_id: int | None,
    fps: float,
    provider_text: str,
    enhanced: bool,
) -> np.ndarray:
    base = frame.copy()
    h, w = base.shape[:2]
    panel_w = min(420, max(300, w // 3))
    canvas = np.zeros((h, w + panel_w, 3), dtype=np.uint8)
    canvas[:, :w] = base

    draw_tracks(canvas[:, :w], tracks, selected_id)
    draw_center_reticle(canvas[:, :w])

    cv2.putText(canvas, "SPECTRATRACK // LOCAL VISION", (16, 28), FONT, 0.62, (245, 245, 245), 1, cv2.LINE_AA)
    cv2.putText(canvas, f"FPS {fps:5.1f} | {provider_text}", (16, 52), FONT, 0.45, (210, 210, 210), 1, cv2.LINE_AA)
    cv2.putText(canvas, f"ENHANCE {'ON' if enhanced else 'OFF'} | TRACKS {len(tracks)}", (16, 73), FONT, 0.45, (210, 210, 210), 1, cv2.LINE_AA)

    px = w
    cv2.rectangle(canvas, (px, 0), (w + panel_w - 1, h - 1), (32, 32, 32), -1)
    cv2.line(canvas, (px, 0), (px, h), (110, 110, 110), 1)
    cv2.putText(canvas, "TARGET VIEW", (px + 18, 30), FONT, 0.6, (240, 240, 240), 1, cv2.LINE_AA)

    selected = next((t for t in tracks if t.track_id == selected_id), None)
    if selected is None:
        cv2.putText(canvas, "click target to lock", (px + 18, 62), FONT, 0.46, (180, 180, 180), 1, cv2.LINE_AA)
    else:
        crop = crop_with_margin(frame, selected.bbox)
        preview_h = min(300, h // 2)
        preview = upscale_preview(crop, panel_w - 24, preview_h)
        canvas[52:52 + preview_h, px + 12:px + 12 + preview.shape[1]] = preview
        y = 52 + preview_h + 28
        cx, cy = selected.center
        speed = math.hypot(selected.vx, selected.vy)
        lines = [
            f"ID       T{selected.track_id:03d}",
            f"CLASS    {selected.label.upper()}",
            f"CONF     {selected.score:.3f}",
            f"CENTER   {int(cx):04d},{int(cy):04d}",
            f"MOTION   {speed:5.1f} px/frame",
            f"AGE      {selected.age}",
            f"MISSED   {selected.missed}",
        ]
        for line in lines:
            cv2.putText(canvas, line, (px + 18, y), FONT, 0.47, (220, 220, 220), 1, cv2.LINE_AA)
            y += 24

    footer = "Q quit | E enhance | Z stabilize | H hud | S snapshot | U AI upscale"
    cv2.putText(canvas, footer, (16, h - 16), FONT, 0.42, (190, 190, 190), 1, cv2.LINE_AA)
    return canvas
