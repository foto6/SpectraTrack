from __future__ import annotations

import math

import cv2
import numpy as np

from .calibration import CameraCalibration
from .enhance import crop_with_margin, upscale_preview
from .frame_quality import FrameQuality
from .types import Track


FONT = cv2.FONT_HERSHEY_SIMPLEX


def _clip_box(box, w, h):
    x1, y1, x2, y2 = map(int, box)
    return max(0, x1), max(0, y1), min(w - 1, x2), min(h - 1, y2)


def draw_corner_box(frame: np.ndarray, box, selected: bool = False, confirmed: bool = True) -> None:
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = _clip_box(box, w, h)
    thickness = 3 if selected else (2 if confirmed else 1)
    length = max(10, int(min(x2 - x1, y2 - y1) * 0.22))
    shade = 255 if selected else (220 if confirmed else 135)
    color = (shade, shade, shade)
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
        draw_corner_box(frame, tr.bbox, selected, tr.confirmed)
        x1, y1, _, _ = map(int, tr.bbox)
        state = "" if tr.confirmed else "?"
        label = f"T{tr.track_id:03d}{state} {tr.label.upper()} {tr.score:.2f} Q{tr.quality:.2f}"
        cv2.putText(frame, label, (max(4, x1), max(18, y1 - 7)), FONT, 0.48, (225, 225, 225), 1, cv2.LINE_AA)
        pts = list(tr.history)
        for i in range(1, len(pts)):
            shade = int(70 + 160 * i / max(1, len(pts)))
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


def _calibrated_lines(cal: CameraCalibration, tr: Track, frame_w: int, frame_h: int) -> list[str]:
    cx, cy = tr.center
    sx = cal.width / max(frame_w, 1)
    sy = cal.height / max(frame_h, 1)
    yaw, pitch = cal.angular_offset_deg(cx * sx, cy * sy)
    aw, ah = cal.angular_size_deg(tr.width * sx, tr.height * sy)
    return [
        f"YAW      {yaw:+6.2f} deg EST",
        f"PITCH    {pitch:+6.2f} deg EST",
        f"ANG SIZE {aw:5.2f}x{ah:5.2f} deg",
    ]


def compose_hud(
    frame: np.ndarray,
    tracks: list[Track],
    selected_id: int | None,
    fps: float,
    provider_text: str,
    enhanced: bool,
    timings_ms: dict[str, float] | None = None,
    camera_motion: tuple[float, float, float] | None = None,
    calibration: CameraCalibration | None = None,
    view_mode: str = "normal",
    frame_quality: FrameQuality | None = None,
    detector_reason: str = "",
) -> np.ndarray:
    base = frame.copy()
    h, w = base.shape[:2]
    panel_w = min(460, max(320, w // 3))
    canvas = np.zeros((h, w + panel_w, 3), dtype=np.uint8)
    canvas[:, :w] = base

    draw_tracks(canvas[:, :w], tracks, selected_id)
    draw_center_reticle(canvas[:, :w])

    confirmed = sum(1 for t in tracks if t.confirmed)
    cv2.putText(canvas, "SPECTRATRACK // PC V0.2", (16, 28), FONT, 0.62, (245, 245, 245), 1, cv2.LINE_AA)
    cv2.putText(canvas, f"FPS {fps:5.1f} | {provider_text}", (16, 52), FONT, 0.45, (210, 210, 210), 1, cv2.LINE_AA)
    view_label = view_mode.upper()
    if view_mode == "pseudo-thermal":
        view_label += " (FALSE COLOR)"
    cv2.putText(canvas, f"ANALYSIS ENH {'ON' if enhanced else 'OFF'} | VIEW {view_label} | TRACKS {confirmed}/{len(tracks)}", (16, 73), FONT, 0.40, (210, 210, 210), 1, cv2.LINE_AA)
    if timings_ms:
        timing = f"DET {timings_ms.get('detect',0):.1f}ms | TRK {timings_ms.get('track',0):.1f} | CMC {timings_ms.get('cmc',0):.1f}"
        cv2.putText(canvas, timing, (16, 94), FONT, 0.40, (185, 185, 185), 1, cv2.LINE_AA)
    if camera_motion:
        dx, dy, rot = camera_motion
        cv2.putText(canvas, f"CAM IMG MOTION {dx:+.1f},{dy:+.1f}px {math.degrees(rot):+.2f}deg", (16, 115), FONT, 0.40, (185, 185, 185), 1, cv2.LINE_AA)
    if frame_quality is not None:
        flags = []
        if frame_quality.low_light:
            flags.append("LOW-LIGHT")
        if frame_quality.blurred:
            flags.append("BLUR")
        if frame_quality.low_contrast:
            flags.append("LOW-CONTRAST")
        quality_text = (
            f"IMG B {frame_quality.brightness:.0f} C {frame_quality.contrast:.0f} "
            f"S {frame_quality.sharpness:.0f} {'/'.join(flags) if flags else 'OK'}"
        )
        cv2.putText(canvas, quality_text, (16, 136), FONT, 0.40, (185, 185, 185), 1, cv2.LINE_AA)
    if detector_reason:
        cv2.putText(canvas, f"DETECTOR {detector_reason.upper()}", (16, 157), FONT, 0.40, (185, 185, 185), 1, cv2.LINE_AA)

    px = w
    cv2.rectangle(canvas, (px, 0), (w + panel_w - 1, h - 1), (32, 32, 32), -1)
    cv2.line(canvas, (px, 0), (px, h), (110, 110, 110), 1)
    cv2.putText(canvas, "TARGET VIEW", (px + 18, 30), FONT, 0.6, (240, 240, 240), 1, cv2.LINE_AA)

    selected = next((t for t in tracks if t.track_id == selected_id), None)
    if selected is None:
        cv2.putText(canvas, "click target to lock", (px + 18, 62), FONT, 0.46, (180, 180, 180), 1, cv2.LINE_AA)
    else:
        crop = crop_with_margin(frame, selected.bbox)
        preview_h = min(280, h // 2)
        preview = upscale_preview(crop, panel_w - 24, preview_h)
        canvas[52:52 + preview_h, px + 12:px + 12 + preview.shape[1]] = preview
        y = 52 + preview_h + 28
        cx, cy = selected.center
        speed = math.hypot(selected.vx, selected.vy)
        lines = [
            f"ID       T{selected.track_id:03d}",
            f"STATE    {selected.lifecycle}",
            f"QUALITY  {selected.quality:.3f}",
            f"RECOVER  {selected.recoveries}",
            f"CLASS    {selected.label.upper()}",
            f"CONF     {selected.score:.3f}",
            f"CENTER   {int(cx):04d},{int(cy):04d}",
            f"MOTION   {speed:5.1f} px/frame EST",
            f"AGE      {selected.age}",
            f"MISSED   {selected.missed}",
        ]
        if calibration is not None:
            lines += _calibrated_lines(calibration, selected, w, h)
        for line in lines:
            if y > h - 24:
                break
            cv2.putText(canvas, line, (px + 18, y), FONT, 0.45, (220, 220, 220), 1, cv2.LINE_AA)
            y += 23

    footer = "Q quit | E analysis | M view | Z stabilize | R reset | H hud | S snapshot | U AI upscale"
    cv2.putText(canvas, footer, (16, h - 16), FONT, 0.42, (190, 190, 190), 1, cv2.LINE_AA)
    return canvas
