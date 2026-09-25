from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
import statistics


def summarize(path: str | Path) -> dict:
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    frames = [r for r in rows if r.get("type") == "frame"]
    header = next((r for r in rows if r.get("type") == "header"), {})
    footer = next((r for r in reversed(rows) if r.get("type") == "footer"), {})
    events = [r for r in rows if r.get("type") == "event"]

    fps = [float(r.get("fps", 0.0)) for r in frames if float(r.get("fps", 0.0)) > 0]
    timing: dict[str, list[float]] = defaultdict(list)
    track_frames: dict[int, int] = defaultdict(int)
    track_missed: dict[int, int] = defaultdict(int)
    track_confirmed: dict[int, int] = defaultdict(int)
    track_quality: dict[int, list[float]] = defaultdict(list)
    track_recoveries: dict[int, int] = defaultdict(int)
    labels: dict[int, str] = {}

    for row in frames:
        for name, value in (row.get("timings_ms") or {}).items():
            timing[name].append(float(value))
        for tr in row.get("tracks") or []:
            tid = int(tr["id"])
            track_frames[tid] += 1
            track_missed[tid] += int(tr.get("missed", 0) > 0)
            track_confirmed[tid] += int(bool(tr.get("confirmed", False)))
            track_quality[tid].append(float(tr.get("quality", 0.0)))
            track_recoveries[tid] = max(track_recoveries[tid], int(tr.get("recoveries", 0)))
            labels[tid] = tr.get("label", "")

    def p95(values: list[float]) -> float:
        if not values:
            return 0.0
        values = sorted(values)
        return values[min(len(values)-1, round((len(values)-1)*0.95))]

    tracks = [
        {
            "id": tid,
            "label": labels.get(tid, ""),
            "frames_present": count,
            "frames_predicted": track_missed.get(tid, 0),
            "prediction_ratio": round(track_missed.get(tid, 0) / max(count, 1), 4),
            "confirmed_ratio": round(track_confirmed.get(tid, 0) / max(count, 1), 4),
            "avg_quality": round(statistics.fmean(track_quality[tid]), 4) if track_quality[tid] else 0.0,
            "recoveries": track_recoveries.get(tid, 0),
        }
        for tid, count in sorted(track_frames.items(), key=lambda kv: (-kv[1], kv[0]))
    ]

    return {
        "session": {
            "frames": len(frames),
            "duration_s": footer.get("duration_s", frames[-1].get("t", 0.0) if frames else 0.0),
            "metadata": header.get("metadata", {}),
        },
        "fps": {
            "avg": round(statistics.fmean(fps), 3) if fps else 0.0,
            "p95": round(p95(fps), 3),
            "min": round(min(fps), 3) if fps else 0.0,
        },
        "timings_ms": {
            name: {
                "avg": round(statistics.fmean(values), 3),
                "p95": round(p95(values), 3),
                "max": round(max(values), 3),
            }
            for name, values in sorted(timing.items())
            if values
        },
        "track_count": len(tracks),
        "longest_track_frames": max((t["frames_present"] for t in tracks), default=0),
        "capture": {
            "reconnect_events": sum(1 for e in events if e.get("name") == "capture_reconnected"),
            "summary": next(
                (e.get("data", {}) for e in reversed(events) if e.get("name") == "capture_summary"),
                {},
            ),
        },
        "events": {
            "created": sum(1 for e in events if e.get("name") == "track_created"),
            "confirmed": sum(1 for e in events if e.get("name") == "track_confirmed"),
            "recovered": sum(1 for e in events if e.get("name") == "track_recovered"),
            "ended": sum(1 for e in events if e.get("name") == "track_ended"),
            "target_lost": sum(1 for e in events if e.get("name") == "target_lost"),
        },
        "tracks": tracks,
    }


def main() -> int:
    p = argparse.ArgumentParser(description="Summarize a SpectraTrack JSONL session")
    p.add_argument("session")
    p.add_argument("--json", dest="json_out", default="")
    args = p.parse_args()
    report = summarize(args.session)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
