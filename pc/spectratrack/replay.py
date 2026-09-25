from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

FONT = cv2.FONT_HERSHEY_SIMPLEX


def load_session_rows(path: str | Path) -> dict[int, dict]:
    p = Path(path)
    if p.is_dir():
        p = p / "frames.jsonl"
    rows: dict[int, dict] = {}
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            rows[int(row["frame"])] = row
    return rows


def render_telemetry(frame: np.ndarray, row: dict | None) -> np.ndarray:
    out = frame.copy()
    if row is None:
        cv2.putText(out, "NO TELEMETRY FOR FRAME", (18, 30), FONT, 0.6, (230, 230, 230), 1, cv2.LINE_AA)
        return out

    selected = row.get("selected_id")
    for tr in row.get("tracks", []):
        x1, y1, x2, y2 = [int(round(v)) for v in tr["bbox"]]
        is_selected = tr.get("id") == selected
        shade = 255 if is_selected else (220 if tr.get("confirmed") else 140)
        color = (shade, shade, shade)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2 if is_selected else 1, cv2.LINE_AA)
        state = "PREDICT" if tr.get("predicted_only") else ("COAST" if tr.get("missed", 0) else "LOCK")
        label = f"T{int(tr['id']):03d} {tr.get('label','?').upper()} {float(tr.get('score',0)):.2f} {state}"
        cv2.putText(out, label, (max(2, x1), max(18, y1 - 5)), FONT, 0.45, color, 1, cv2.LINE_AA)

    extra = row.get("extra", {})
    motion = row.get("motion") or {}
    line = (
        f"FRAME {row.get('frame')} | DET {'RUN' if extra.get('detector_ran', True) else 'SKIP'} "
        f"x{extra.get('detector_interval',1)} | CMC {float(motion.get('dx',0)):+.1f},{float(motion.get('dy',0)):+.1f}"
    )
    cv2.putText(out, line, (18, 28), FONT, 0.55, (245, 245, 245), 1, cv2.LINE_AA)
    return out


def main() -> int:
    p = argparse.ArgumentParser(description="Replay SpectraTrack telemetry over the original video")
    p.add_argument("--video", default="", help="Tracking-input video; defaults to session/tracking-input.mp4")
    p.add_argument("--session", required=True, help="Session directory or frames.jsonl")
    p.add_argument("--start-frame", type=int, default=1)
    args = p.parse_args()

    rows = load_session_rows(args.session)
    video_path = Path(args.video) if args.video else Path(args.session) / "tracking-input.mp4"
    if not video_path.is_file():
        raise SystemExit(f"Replay video not found: {video_path}")
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise SystemExit(f"Cannot open video: {video_path}")
    if args.start_frame > 1:
        cap.set(cv2.CAP_PROP_POS_FRAMES, args.start_frame - 1)

    paused = False
    frame_no = max(1, args.start_frame)
    current = None
    try:
        while True:
            if not paused or current is None:
                ok, frame = cap.read()
                if not ok:
                    break
                current = frame
            display = render_telemetry(current, rows.get(frame_no))
            cv2.putText(display, "SPACE pause | N step | Q quit", (18, display.shape[0] - 16), FONT, 0.45, (220, 220, 220), 1, cv2.LINE_AA)
            cv2.imshow("SpectraTrack replay", display)
            key = cv2.waitKey(0 if paused else 1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord(" "):
                paused = not paused
            if paused and key == ord("n"):
                ok, frame = cap.read()
                if not ok:
                    break
                current = frame
                frame_no += 1
                continue
            if not paused:
                frame_no += 1
    finally:
        cap.release()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
