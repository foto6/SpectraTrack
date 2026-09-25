from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


FIELDS = [
    "frame", "t", "fps", "selected_id", "track_id", "label", "class_id", "score",
    "x1", "y1", "x2", "y2", "cx", "cy", "vx", "vy",
    "age", "hits", "missed", "confirmed", "lifecycle", "quality", "recoveries",
]


def export_csv(session_path: str | Path, output_path: str | Path) -> int:
    rows_written = 0
    with Path(session_path).open("r", encoding="utf-8") as src, Path(output_path).open("w", newline="", encoding="utf-8") as dst:
        writer = csv.DictWriter(dst, fieldnames=FIELDS)
        writer.writeheader()
        for line in src:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("type") != "frame":
                continue
            for tr in row.get("tracks") or []:
                x1, y1, x2, y2 = tr["bbox"]
                cx, cy = tr["center"]
                vx, vy = tr.get("velocity_px_frame", [0.0, 0.0])
                writer.writerow({
                    "frame": row["frame"],
                    "t": row.get("t", 0.0),
                    "fps": row.get("fps", 0.0),
                    "selected_id": row.get("selected_id"),
                    "track_id": tr["id"],
                    "label": tr.get("label", ""),
                    "class_id": tr.get("class_id", -1),
                    "score": tr.get("score", 0.0),
                    "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                    "cx": cx, "cy": cy, "vx": vx, "vy": vy,
                    "age": tr.get("age", 0),
                    "hits": tr.get("hits", 0),
                    "missed": tr.get("missed", 0),
                    "confirmed": tr.get("confirmed", False),
                    "lifecycle": tr.get("lifecycle", ""),
                    "quality": tr.get("quality", 0.0),
                    "recoveries": tr.get("recoveries", 0),
                })
                rows_written += 1
    return rows_written


def main() -> int:
    p = argparse.ArgumentParser(description="Export SpectraTrack JSONL tracks to CSV")
    p.add_argument("session")
    p.add_argument("output")
    args = p.parse_args()
    count = export_csv(args.session, args.output)
    print(f"rows={count} output={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
