from __future__ import annotations

import argparse
from dataclasses import dataclass
from math import hypot, sin, cos
import random

from .tracker import MultiObjectTracker
from .types import Detection, Track


@dataclass(slots=True)
class EvalMetrics:
    frames: int = 0
    gt_observations: int = 0
    matched_observations: int = 0
    id_switches: int = 0
    fragmentations: int = 0

    @property
    def match_rate(self) -> float:
        return self.matched_observations / max(self.gt_observations, 1)


@dataclass(slots=True)
class GroundTruth:
    target_id: int
    class_id: int
    cx: float
    cy: float
    width: float
    height: float

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        return (
            self.cx - self.width / 2,
            self.cy - self.height / 2,
            self.cx + self.width / 2,
            self.cy + self.height / 2,
        )


def appearance_for_target(target_id: int, dims: int = 16) -> tuple[float, ...]:
    values = [0.0] * dims
    values[target_id % dims] = 1.0
    return tuple(values)


def ground_truth_frame(frame: int, targets: int) -> list[GroundTruth]:
    out = []
    for tid in range(targets):
        # Multiple same-class trajectories deliberately share the central region.
        phase = tid * 0.73
        cx = 220 + tid * 135 + 115 * sin(frame * 0.035 + phase)
        cy = 300 + 140 * cos(frame * 0.027 + phase)
        out.append(GroundTruth(
            target_id=tid,
            class_id=tid % 2,
            cx=cx,
            cy=cy,
            width=72 + (tid % 3) * 12,
            height=120 + (tid % 2) * 22,
        ))
    return out


def detections_for_frame(
    frame: int,
    truths: list[GroundTruth],
    rng: random.Random,
    dropout_period: int = 73,
    use_appearance: bool = True,
) -> list[Detection]:
    detections: list[Detection] = []
    for gt in truths:
        if (frame + gt.target_id * 11) % dropout_period in (0, 1):
            continue
        dx = rng.gauss(0.0, 2.2)
        dy = rng.gauss(0.0, 2.2)
        x1, y1, x2, y2 = gt.bbox
        score = 0.90 if frame % 41 else 0.28
        detections.append(Detection(
            (x1 + dx, y1 + dy, x2 + dx, y2 + dy),
            score,
            gt.class_id,
            f"class_{gt.class_id}",
            appearance_for_target(gt.target_id) if use_appearance else None,
        ))
    return detections


def match_tracks_to_truth(
    tracks: list[Track],
    truths: list[GroundTruth],
    max_distance: float = 100.0,
) -> dict[int, int]:
    pairs: list[tuple[float, int, int]] = []
    for gt in truths:
        for tr in tracks:
            if tr.class_id != gt.class_id or not tr.confirmed:
                continue
            cx, cy = tr.center
            d = hypot(cx - gt.cx, cy - gt.cy)
            if d <= max_distance:
                pairs.append((d, gt.target_id, tr.track_id))
    pairs.sort()
    result: dict[int, int] = {}
    used_tracks: set[int] = set()
    for _, gid, tid in pairs:
        if gid not in result and tid not in used_tracks:
            result[gid] = tid
            used_tracks.add(tid)
    return result


def evaluate(
    frames: int = 600,
    targets: int = 8,
    seed: int = 12345,
    use_appearance: bool = True,
) -> EvalMetrics:
    rng = random.Random(seed)
    tracker = MultiObjectTracker(
        min_hits=2,
        max_missed=10,
        max_center_ratio=2.4,
    )
    metrics = EvalMetrics()
    previous_assignment: dict[int, int] = {}
    was_matched: dict[int, bool] = {}

    for frame in range(frames):
        truths = ground_truth_frame(frame, targets)

        camera_dx = 3.0 * sin(frame * 0.023)
        camera_dy = 2.0 * cos(frame * 0.019)
        affine = (1.0, 0.0, camera_dx, 0.0, 1.0, camera_dy)

        # Move GT into image coordinates for this synthetic camera transform.
        transformed = [
            GroundTruth(
                gt.target_id, gt.class_id,
                gt.cx + camera_dx, gt.cy + camera_dy,
                gt.width, gt.height,
            )
            for gt in truths
        ]
        detections = detections_for_frame(frame, transformed, rng, use_appearance=use_appearance)
        tracks = tracker.update(detections, camera_transform=affine)
        assignment = match_tracks_to_truth(tracks, transformed)

        metrics.frames += 1
        metrics.gt_observations += len(transformed)
        metrics.matched_observations += len(assignment)

        for gid in range(targets):
            matched_now = gid in assignment
            if matched_now:
                current_tid = assignment[gid]
                prev_tid = previous_assignment.get(gid)
                if prev_tid is not None and current_tid != prev_tid:
                    metrics.id_switches += 1
                if was_matched.get(gid) is False and prev_tid is not None:
                    metrics.fragmentations += 1
                previous_assignment[gid] = current_tid
            was_matched[gid] = matched_now

    return metrics


def main() -> int:
    p = argparse.ArgumentParser(description="Deterministic synthetic SpectraTrack identity evaluation")
    p.add_argument("--frames", type=int, default=600)
    p.add_argument("--targets", type=int, default=8)
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--no-appearance", action="store_true")
    p.add_argument("--max-id-switches", type=int, default=-1)
    p.add_argument("--min-match-rate", type=float, default=0.75)
    args = p.parse_args()

    m = evaluate(args.frames, args.targets, args.seed, use_appearance=not args.no_appearance)
    print(f"frames={m.frames}")
    print(f"gt_observations={m.gt_observations}")
    print(f"matched_observations={m.matched_observations}")
    print(f"match_rate={m.match_rate:.4f}")
    print(f"id_switches={m.id_switches}")
    print(f"fragmentations={m.fragmentations}")

    if m.match_rate < args.min_match_rate:
        raise SystemExit(f"match rate {m.match_rate:.4f} below {args.min_match_rate:.4f}")
    if args.max_id_switches >= 0 and m.id_switches > args.max_id_switches:
        raise SystemExit(f"id switches {m.id_switches} above {args.max_id_switches}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
