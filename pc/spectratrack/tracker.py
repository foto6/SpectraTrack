from __future__ import annotations

from math import hypot
from typing import Iterable

from .types import Detection, Track, BBox


def bbox_iou(a: BBox, b: BBox) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    xx1, yy1 = max(ax1, bx1), max(ay1, by1)
    xx2, yy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, xx2 - xx1) * max(0.0, yy2 - yy1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    return inter / max(area_a + area_b - inter, 1e-6)


def shift_box(box: BBox, dx: float, dy: float) -> BBox:
    x1, y1, x2, y2 = box
    return x1 + dx, y1 + dy, x2 + dx, y2 + dy


class MultiObjectTracker:
    """Lightweight multi-object tracker for local camera use.

    It combines constant-velocity prediction, class-aware IoU matching and
    center-distance gating. It is deliberately dependency-light so the same
    logic can be mirrored on Android.
    """

    def __init__(self, max_missed: int = 12, min_iou: float = 0.12, max_center_ratio: float = 1.8) -> None:
        self.max_missed = int(max_missed)
        self.min_iou = float(min_iou)
        self.max_center_ratio = float(max_center_ratio)
        self._next_id = 1
        self.tracks: dict[int, Track] = {}

    def reset(self) -> None:
        self._next_id = 1
        self.tracks.clear()

    def _predicted_box(self, track: Track) -> BBox:
        # Damp prediction as a target remains unobserved.
        factor = max(0.25, 1.0 - track.missed * 0.08)
        return shift_box(track.bbox, track.vx * factor, track.vy * factor)

    def update(self, detections: Iterable[Detection]) -> list[Track]:
        detections = list(detections)
        unmatched_tracks = set(self.tracks.keys())
        unmatched_dets = set(range(len(detections)))
        candidates: list[tuple[float, int, int]] = []

        for tid, track in self.tracks.items():
            predicted = self._predicted_box(track)
            pcx = (predicted[0] + predicted[2]) * 0.5
            pcy = (predicted[1] + predicted[3]) * 0.5
            diag = max(hypot(track.width, track.height), 24.0)
            for didx, det in enumerate(detections):
                if det.class_id != track.class_id:
                    continue
                iou = bbox_iou(predicted, det.bbox)
                dcx, dcy = det.center
                dist_ratio = hypot(dcx - pcx, dcy - pcy) / diag
                if iou < self.min_iou and dist_ratio > self.max_center_ratio:
                    continue
                # Higher score is better: IoU dominates, distance helps fast motion.
                match_score = iou * 2.2 + max(0.0, 1.0 - dist_ratio / self.max_center_ratio)
                candidates.append((match_score, tid, didx))

        candidates.sort(reverse=True)
        matches: list[tuple[int, int]] = []
        for _, tid, didx in candidates:
            if tid in unmatched_tracks and didx in unmatched_dets:
                unmatched_tracks.remove(tid)
                unmatched_dets.remove(didx)
                matches.append((tid, didx))

        for tid, didx in matches:
            track = self.tracks[tid]
            det = detections[didx]
            old_cx, old_cy = track.center
            new_cx, new_cy = det.center
            measured_vx = new_cx - old_cx
            measured_vy = new_cy - old_cy
            track.vx = track.vx * 0.6 + measured_vx * 0.4
            track.vy = track.vy * 0.6 + measured_vy * 0.4
            track.bbox = det.bbox
            track.score = det.score
            track.label = det.label
            track.age += 1
            track.hits += 1
            track.missed = 0
            track.history.append((int(new_cx), int(new_cy)))

        for tid in unmatched_tracks:
            track = self.tracks[tid]
            track.bbox = self._predicted_box(track)
            track.age += 1
            track.missed += 1
            cx, cy = track.center
            track.history.append((int(cx), int(cy)))

        for didx in unmatched_dets:
            det = detections[didx]
            tid = self._next_id
            self._next_id += 1
            t = Track(tid, det.bbox, det.score, det.class_id, det.label)
            cx, cy = det.center
            t.history.append((int(cx), int(cy)))
            self.tracks[tid] = t

        expired = [tid for tid, tr in self.tracks.items() if tr.missed > self.max_missed]
        for tid in expired:
            del self.tracks[tid]

        return sorted(self.tracks.values(), key=lambda t: t.track_id)
