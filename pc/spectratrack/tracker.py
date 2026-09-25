from __future__ import annotations

from dataclasses import dataclass
from math import hypot
from typing import Iterable

import numpy as np

from .kalman import KalmanBox
from .types import BBox, Detection, Track


def bbox_iou(a: BBox, b: BBox) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    xx1, yy1 = max(ax1, bx1), max(ay1, by1)
    xx2, yy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, xx2 - xx1) * max(0.0, yy2 - yy1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    return inter / max(area_a + area_b - inter, 1e-6)


def _size_similarity(a: BBox, b: BBox) -> float:
    aw, ah = max(1.0, a[2] - a[0]), max(1.0, a[3] - a[1])
    bw, bh = max(1.0, b[2] - b[0]), max(1.0, b[3] - b[1])
    return min(aw, bw) / max(aw, bw) * min(ah, bh) / max(ah, bh)


@dataclass(slots=True)
class TrackerConfig:
    high_conf: float = 0.45
    low_conf: float = 0.12
    new_track_conf: float = 0.55
    max_missed: int = 24
    min_hits: int = 2
    min_iou: float = 0.04
    max_center_ratio: float = 2.2
    tentative_max_missed: int = 1


@dataclass
class _State:
    public: Track
    kf: KalmanBox


class MultiObjectTracker:
    """Two-stage tracker with Kalman prediction and global camera compensation."""

    def __init__(
        self,
        max_missed: int | None = None,
        min_iou: float | None = None,
        max_center_ratio: float | None = None,
        config: TrackerConfig | None = None,
    ) -> None:
        cfg = config or TrackerConfig()
        if max_missed is not None:
            cfg.max_missed = int(max_missed)
        if min_iou is not None:
            cfg.min_iou = float(min_iou)
        if max_center_ratio is not None:
            cfg.max_center_ratio = float(max_center_ratio)
        self.config = cfg
        self._next_id = 1
        self._states: dict[int, _State] = {}

    @property
    def tracks(self) -> dict[int, Track]:
        return {tid: state.public for tid, state in self._states.items()}

    def reset(self) -> None:
        self._next_id = 1
        self._states.clear()

    def _spawn(self, det: Detection) -> None:
        tid = self._next_id
        self._next_id += 1
        track = Track(tid, det.bbox, det.score, det.class_id, det.label)
        track.confirmed = self.config.min_hits <= 1
        cx, cy = det.center
        track.history.append((int(cx), int(cy)))
        self._states[tid] = _State(track, KalmanBox(det.bbox))

    def _candidate_score(self, track: Track, det: Detection, relaxed: bool) -> float | None:
        if track.class_id != det.class_id:
            return None
        iou = bbox_iou(track.bbox, det.bbox)
        tcx, tcy = track.center
        dcx, dcy = det.center
        diag = max(hypot(track.width, track.height), 24.0)
        dist_ratio = hypot(dcx - tcx, dcy - tcy) / diag
        max_dist = self.config.max_center_ratio * (1.35 if relaxed else 1.0)
        min_iou = self.config.min_iou * (0.5 if relaxed else 1.0)
        if iou < min_iou and dist_ratio > max_dist:
            return None
        distance_score = max(0.0, 1.0 - dist_ratio / max_dist)
        size_score = _size_similarity(track.bbox, det.bbox)
        return iou * 2.6 + distance_score * 0.9 + size_score * 0.35 + det.score * 0.2

    def _associate(
        self,
        track_ids: set[int],
        det_indices: set[int],
        detections: list[Detection],
        relaxed: bool = False,
    ) -> tuple[list[tuple[int, int, float]], set[int], set[int]]:
        candidates: list[tuple[float, int, int]] = []
        for tid in track_ids:
            track = self._states[tid].public
            for didx in det_indices:
                score = self._candidate_score(track, detections[didx], relaxed)
                if score is not None:
                    candidates.append((score, tid, didx))

        candidates.sort(reverse=True)
        remaining_tracks = set(track_ids)
        remaining_dets = set(det_indices)
        matches: list[tuple[int, int, float]] = []
        min_score = 0.24 if relaxed else 0.34
        for score, tid, didx in candidates:
            if score < min_score:
                continue
            if tid in remaining_tracks and didx in remaining_dets:
                remaining_tracks.remove(tid)
                remaining_dets.remove(didx)
                matches.append((tid, didx, score))
        return matches, remaining_tracks, remaining_dets

    def update(
        self,
        detections: Iterable[Detection],
        camera_affine: np.ndarray | None = None,
        dt: float = 1.0,
        detector_ran: bool = True,
    ) -> list[Track]:
        detections = [d for d in detections if d.score >= self.config.low_conf] if detector_ran else []

        for state in self._states.values():
            state.public.bbox = state.kf.predict(dt=dt, camera_affine=camera_affine)
            vx, vy = state.kf.velocity
            state.public.vx, state.public.vy = vx, vy

        all_tracks = set(self._states)
        high = {i for i, d in enumerate(detections) if d.score >= self.config.high_conf}
        low = set(range(len(detections))) - high

        stage1, unmatched_tracks, unmatched_high = self._associate(all_tracks, high, detections, relaxed=False)
        confirmed_unmatched = {tid for tid in unmatched_tracks if self._states[tid].public.confirmed}
        stage2, still_unmatched, _ = self._associate(confirmed_unmatched, low, detections, relaxed=True)
        unmatched_tracks = (unmatched_tracks - confirmed_unmatched) | still_unmatched

        for tid, didx, assoc_score in stage1 + stage2:
            state = self._states[tid]
            det = detections[didx]
            state.public.bbox = state.kf.update(det.bbox)
            vx, vy = state.kf.velocity
            state.public.vx, state.public.vy = vx, vy
            state.public.score = det.score
            state.public.label = det.label
            state.public.age += 1
            state.public.hits += 1
            state.public.missed = 0
            state.public.predicted_only = False
            state.public.association_score = assoc_score
            if state.public.hits >= self.config.min_hits:
                state.public.confirmed = True
            cx, cy = state.public.center
            state.public.history.append((int(cx), int(cy)))

        for tid in unmatched_tracks:
            state = self._states[tid]
            state.public.age += 1
            state.public.association_score = 0.0
            if detector_ran:
                state.public.missed += 1
                state.public.predicted_only = False
            else:
                state.public.predicted_only = True
            cx, cy = state.public.center
            state.public.history.append((int(cx), int(cy)))

        if detector_ran:
            for didx in unmatched_high:
                det = detections[didx]
                if det.score >= self.config.new_track_conf:
                    self._spawn(det)

        expired: list[int] = []
        for tid, state in self._states.items():
            limit = self.config.max_missed if state.public.confirmed else self.config.tentative_max_missed
            if state.public.missed > limit:
                expired.append(tid)
        for tid in expired:
            del self._states[tid]

        return sorted((state.public for state in self._states.values()), key=lambda t: t.track_id)
