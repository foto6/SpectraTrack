from __future__ import annotations

import cv2
import numpy as np

from .calibration import CameraGeometry
from .cmc import MotionEstimate
from .enhance import crop_with_margin, upscale_preview
from .types import Track

FONT = cv2.FONT_HERSHEY_SIMPLEX


def _clip_box(box, w, h):
    x1, y1, x2, y2 = map(int, box)
    return max(0, x1), max(0, y1), min(w - 1, x2), min(h - 1, y2)


def draw_corner_box(frame: np.ndarray, box, selected: bool = False, confirmed: bool = True) -> None:
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = _clip_box(box, w, h)
    thickness = 3 if selected else 2
    length = max(10, int(min(max(1, x2 - x1), max(1, y2 - y1)) * 0.22))
    shade = 255 if selected else (220 if confirmed else 145)
    color = (shade, shade, shade)
    for a, b, c, d in [
        (x1, y1, x1 + length, y1), (x1, y1, x1, y1 + length),
        (x2, y1, x2 - length, y1), (x2, y1, x2, y1 + length),
        (x1, y2, x1 + length, y2), (x1, y2, x1, y2 - length),
        (x2, y2, x2 - length, y2), (x2, y2, x2, y2 - length),
    ]:
        cv2.line(frame, (a, b), (c, d), color, thickness, cv2.LINE_AA)


def draw_mask_outline(frame: np.ndarray, mask: np.ndarray | None) -> None:
    if mask is None or mask.shape[:2] != frame.shape[:2]:
        return
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        cv2.drawContours(frame, contours, -1, (255, 255, 255), 1, cv2.LINE_AA)


def draw_tracks(frame: np.ndarray, tracks: list[Track], selected_id: int | None) -> None:
    for tr in tracks:
        selected = tr.track_id == selected_id
        draw_corner_box(frame, tr.bbox, selected, tr.confirmed)
        x1, y1, _, _ = map(int, tr.bbox)
        prefix = "" if tr.confirmed else "?"
        text = f"{prefix}T{tr.track_id:03d} {tr.label.upper()} {tr.score:.2f} {tr.state}"
        cv2.putText(frame, text, (max(4, x1), max(18, y1 - 7)), FONT, 0.46, (242, 242, 242), 1, cv2.LINE_AA)
        pts = list(tr.history)
        if len(pts) >= 2:
            for i in range(1, len(pts)):
                shade = int(60 + 170 * (i / len(pts)))
                cv2.line(frame, pts[i - 1], pts[i], (shade, shade, shade), 1, cv2.LINE_AA)


def draw_center_reticle(frame: np.ndarray) -> None:
    h, w = frame.shape[:2]
    cx, cy = w // 2, h // 2
    r = max(18, min(w, h) // 32)
    color = (190, 190, 190)
    cv2.circle(frame, (cx, cy), r, color, 1, cv2.LINE_AA)
    cv2.line(frame, (cx - r - 14, cy), (cx - r + 2, cy), color, 1, cv2.LINE_AA)
    cv2.line(frame, (cx + r - 2, cy), (cx + r + 14, cy), color, 1, cv2.LINE_AA)
    cv2.line(frame, (cx, cy - r - 14), (cx, cy - r + 2), color, 1, cv2.LINE_AA)
    cv2.line(frame, (cx, cy + r - 2), (cx, cy + r + 14), color, 1, cv2.LINE_AA)


def _metric(metrics: dict[str, float], name: str) -> float:
    return float(metrics.get(f"{name}_avg_ms", metrics.get(name, 0.0)))


def compose_hud(
    frame: np.ndarray,
    tracks: list[Track],
    selected_id: int | None,
    fps: float,
    provider_text: str,
    enhancement_mode: str,
    metrics: dict[str, float] | None = None,
    motion: MotionEstimate | None = None,
    geometry: CameraGeometry | None = None,
    cmc_enabled: bool = True,
    stabilization_enabled: bool = False,
    detector_interval: int = 1,
    detector_ran: bool = True,
    target_mask: np.ndarray | None = None,
    mask_enabled: bool = False,
) -> np.ndarray:
    metrics = metrics or {}
    geometry = geometry or CameraGeometry()
    base = frame.copy()
    h, w = base.shape[:2]
    panel_w = min(440, max(320, w // 3))
    canvas = np.zeros((h, w + panel_w, 3), dtype=np.uint8)
    canvas[:, :w] = base

    if mask_enabled:
        draw_mask_outline(canvas[:, :w], target_mask)
    draw_tracks(canvas[:, :w], tracks, selected_id)
    draw_center_reticle(canvas[:, :w])

    cv2.putText(canvas, "SPECTRATRACK // PC V0.2", (16, 27), FONT, 0.62, (245, 245, 245), 1, cv2.LINE_AA)
    cv2.putText(
        canvas,
        f"FPS {fps:5.1f} | DET {_metric(metrics,'detect'):5.1f} ms x{detector_interval} {'RUN' if detector_ran else 'SKIP'} | TRK {_metric(metrics,'track'):4.1f} ms | {provider_text}",
        (16, 51), FONT, 0.41, (210, 210, 210), 1, cv2.LINE_AA,
    )
    cmc_text = "CMC OFF"
    if cmc_enabled and motion is not None:
        cmc_text = f"CMC {motion.dx:+.1f},{motion.dy:+.1f}px {motion.rotation_deg:+.2f}deg {'OK' if motion.valid else 'HOLD'}"
    cv2.putText(
        canvas,
        f"ENH {enhancement_mode.upper()} | STAB {'ON' if stabilization_enabled else 'OFF'} | MASK {'ON' if mask_enabled else 'OFF'} | {cmc_text}",
        (16, 73), FONT, 0.41, (205, 205, 205), 1, cv2.LINE_AA,
    )

    px = w
    cv2.rectangle(canvas, (px, 0), (w + panel_w - 1, h - 1), (26, 26, 26), -1)
    cv2.line(canvas, (px, 0), (px, h), (110, 110, 110), 1)
    cv2.putText(canvas, "TARGET VIEW", (px + 18, 30), FONT, 0.6, (240, 240, 240), 1, cv2.LINE_AA)

    selected = next((t for t in tracks if t.track_id == selected_id), None)
    if selected is None:
        cv2.putText(canvas, "click target to lock", (px + 18, 62), FONT, 0.46, (180, 180, 180), 1, cv2.LINE_AA)
        cv2.putText(canvas, "values are image-derived", (px + 18, 88), FONT, 0.42, (150, 150, 150), 1, cv2.LINE_AA)
    else:
        crop = crop_with_margin(frame, selected.bbox)
        preview_h = min(300, h // 2)
        preview = upscale_preview(crop, panel_w - 24, preview_h)
        canvas[52:52 + preview_h, px + 12:px + 12 + preview.shape[1]] = preview
        y = 52 + preview_h + 27
        cx, cy = selected.center
        bearing = geometry.bearing_offset_deg(cx, w)
        angular = geometry.angular_rate_deg_s(cx, selected.vx, fps, w)
        lines = [
            f"ID          T{selected.track_id:03d}",
            f"STATE       {selected.state}",
            f"CLASS       {selected.label.upper()}",
            f"CONF        {selected.score:.3f}",
            f"ASSOC       {selected.association_score:.2f}",
            f"CENTER      {int(cx):04d},{int(cy):04d}",
            f"IMG SPEED   {selected.speed_px_per_frame * fps:6.1f} px/s",
        ]
        if bearing is not None:
            lines.append(f"REL ANGLE   {bearing:+6.2f} deg [CAL]")
        if angular is not None:
            lines.append(f"ANG RATE    {angular:+6.2f} deg/s [EST]")
        if mask_enabled:
            lines.append(f"MASK        {'CLASSICAL' if target_mask is not None else 'NO LOCK'}")
        lines.extend([f"AGE         {selected.age}", f"MISSED      {selected.missed}"])
        for line in lines:
            if y > h - 42:
                break
            cv2.putText(canvas, line, (px + 18, y), FONT, 0.45, (220, 220, 220), 1, cv2.LINE_AA)
            y += 23

    footer = "Q quit | E enhance | C CMC | Z stabilize | M mask | H HUD | S snapshot | U AI SR"
    cv2.putText(canvas, footer, (16, h - 16), FONT, 0.39, (190, 190, 190), 1, cv2.LINE_AA)
    return canvas
