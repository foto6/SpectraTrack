from __future__ import annotations

import argparse
import math
import time

from .tracker import MultiObjectTracker
from .types import Detection


def make_frame(frame: int, targets: int, width: int = 1920, height: int = 1080) -> list[Detection]:
    out: list[Detection] = []
    for i in range(targets):
        phase = i * 0.73
        cx = width * (0.1 + 0.8 * ((i + 1) / (targets + 1))) + math.sin(frame * 0.031 + phase) * 45
        cy = height * 0.5 + math.cos(frame * 0.027 + phase) * 180
        bw = 45 + (i % 4) * 12
        bh = 80 + (i % 3) * 18
        score = 0.86 if frame % 47 else 0.28
        if frame % 89 == 0 and i % 5 == 0:
            continue
        out.append(Detection((cx-bw/2, cy-bh/2, cx+bw/2, cy+bh/2), score, i % 3, f"class_{i%3}"))
    return out


def quality_probe() -> tuple[int, int]:
    appearance_a = tuple([1.0] + [0.0] * 7)
    appearance_b = tuple([0.0, 1.0] + [0.0] * 6)

    crossing = MultiObjectTracker(min_hits=1, max_center_ratio=2.5)
    initial = crossing.update([
        Detection((0, 100, 100, 200), 0.9, 0, "obj", appearance_a),
        Detection((300, 100, 400, 200), 0.9, 0, "obj", appearance_b),
    ])
    a_id = min(initial, key=lambda t: t.center[0]).track_id
    b_id = max(initial, key=lambda t: t.center[0]).track_id
    tracks = initial
    for xa, xb in [(60, 240), (120, 180), (180, 120), (240, 60)]:
        tracks = crossing.update([
            Detection((xa, 100, xa + 100, 200), 0.9, 0, "obj", appearance_a),
            Detection((xb, 100, xb + 100, 200), 0.9, 0, "obj", appearance_b),
        ])
    by_id = {track.track_id: track for track in tracks}
    crossing_switches = int(
        a_id not in by_id
        or b_id not in by_id
        or by_id[a_id].center[0] <= by_id[b_id].center[0]
    )

    reappearance = MultiObjectTracker(max_missed=2, min_hits=1, reactivation_window=6)
    first_id = reappearance.update([
        Detection((50, 50, 150, 150), 0.9, 0, "obj", appearance_a)
    ])[0].track_id
    reappearance.update([])
    reappearance.update([])
    reappearance.update([])
    resumed = reappearance.update([
        Detection((80, 50, 180, 150), 0.9, 0, "obj", appearance_a)
    ])[0]
    reappearance_switches = int(resumed.track_id != first_id)
    return crossing_switches, reappearance_switches


def main() -> int:
    p = argparse.ArgumentParser(description="Synthetic SpectraTrack tracker benchmark")
    p.add_argument("--frames", type=int, default=3000)
    p.add_argument("--targets", type=int, default=32)
    args = p.parse_args()
    tracker = MultiObjectTracker(min_hits=1)
    start = time.perf_counter()
    observations = 0
    for frame in range(args.frames):
        camera_dx = 2.0 * math.sin(frame * 0.02)
        camera_dy = 1.0 * math.cos(frame * 0.018)
        detections = make_frame(frame, args.targets)
        observations += len(detections)
        tracker.update(detections, camera_motion=(camera_dx, camera_dy))
    elapsed = time.perf_counter() - start
    fps = args.frames / max(elapsed, 1e-9)
    print(f"frames={args.frames} targets={args.targets} observations={observations}")
    print(f"elapsed={elapsed:.3f}s tracker_fps={fps:.1f} active_tracks={len(tracker.tracks)}")
    crossing_switches, reappearance_switches = quality_probe()
    print(
        f"quality crossing_id_switches={crossing_switches} "
        f"reappearance_id_switches={reappearance_switches}"
    )
    if fps < 30:
        raise SystemExit("Synthetic tracker benchmark below 30 FPS")
    if crossing_switches or reappearance_switches:
        raise SystemExit("Synthetic tracker identity quality probe failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
