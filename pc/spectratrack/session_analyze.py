from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean


def analyze_session(path: str | Path) -> dict:
    p = Path(path)
    if p.is_dir():
        p = p / "frames.jsonl"
    rows = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    if not rows:
        return {"frames": 0, "tracks": 0, "duration_s": 0.0, "per_track": {}}

    by_track: dict[int, list[dict]] = defaultdict(list)
    max_concurrent = 0
    metric_values: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        max_concurrent = max(max_concurrent, len(row.get("tracks", [])))
        for tr in row.get("tracks", []):
            by_track[int(tr["id"])].append(tr)
        for key, value in row.get("metrics", {}).items():
            if isinstance(value, (int, float)):
                metric_values[key].append(float(value))

    per_track = {}
    for tid, samples in sorted(by_track.items()):
        per_track[str(tid)] = {
            "label": samples[-1].get("label"),
            "frames_present": len(samples),
            "confirmed_seen": any(bool(x.get("confirmed")) for x in samples),
            "mean_confidence": mean(float(x.get("score", 0.0)) for x in samples),
            "max_missed": max(int(x.get("missed", 0)) for x in samples),
            "mean_speed_px_per_frame": mean(
                (float(x.get("vx", 0.0)) ** 2 + float(x.get("vy", 0.0)) ** 2) ** 0.5
                for x in samples
            ),
        }

    start_t = float(rows[0].get("t", 0.0))
    end_t = float(rows[-1].get("t", start_t))
    return {
        "frames": len(rows),
        "tracks": len(by_track),
        "duration_s": max(0.0, end_t - start_t),
        "max_concurrent_tracks": max_concurrent,
        "metrics_mean": {k: mean(v) for k, v in sorted(metric_values.items()) if v},
        "per_track": per_track,
    }


def main() -> int:
    p = argparse.ArgumentParser(description="Summarize a SpectraTrack JSONL session")
    p.add_argument("session", help="Session directory or frames.jsonl")
    p.add_argument("--output", default="")
    args = p.parse_args()
    result = analyze_session(args.session)
    payload = json.dumps(result, indent=2, sort_keys=True)
    print(payload)
    if args.output:
        Path(args.output).write_text(payload + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
