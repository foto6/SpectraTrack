from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2


FONT = cv2.FONT_HERSHEY_SIMPLEX


def load_frames(session_path: str | Path) -> dict[int, dict]:
    frames: dict[int, dict] = {}
    for line in Path(session_path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("type") == "frame":
            frames[int(row["frame"])] = row
    return frames


def main() -> int:
    p = argparse.ArgumentParser(description="Render recorded SpectraTrack metadata over the original video")
    p.add_argument("--video", required=True)
    p.add_argument("--session", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()

    metadata = load_frames(args.session)
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open video: {args.video}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0
    writer = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise SystemExit(f"Cannot create: {args.output}")

    frame_index = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_index += 1
            row = metadata.get(frame_index)
            if row:
                selected = row.get("selected_id")
                for tr in row.get("tracks") or []:
                    x1, y1, x2, y2 = map(int, tr["bbox"])
                    tid = int(tr["id"])
                    shade = 255 if tid == selected else (210 if tr.get("confirmed", False) else 130)
                    color = (shade, shade, shade)
                    thickness = 3 if tid == selected else 2
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
                    label = f"T{tid:03d} {tr.get('label','').upper()} {tr.get('lifecycle','')} Q{tr.get('quality',0):.2f}"
                    cv2.putText(frame, label, (max(3, x1), max(18, y1 - 5)), FONT, 0.45, color, 1, cv2.LINE_AA)
                cv2.putText(frame, f"RECORDED METADATA frame={frame_index}", (12, 26), FONT, 0.55, (230,230,230), 1, cv2.LINE_AA)
            writer.write(frame)
    finally:
        cap.release()
        writer.release()

    print(f"frames={frame_index} output={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
