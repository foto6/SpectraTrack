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


def transform_point(x: float, y: float, affine: tuple[float, float, float, float, float, float]) -> tuple[float, float]:
    a, b, tx, c, d, ty = affine
    return a * x + b * y + tx, c * x + d * y + ty


def appearance_similarity(a: tuple[float, ...] | None, b: tuple[float, ...] | None) -> float | None:
    if a is None or b is None or len(a) != len(b) or not a:
        return None
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    if na <= 1e-12 or nb <= 1e-12:
        return None
    return max(0.0, min(1.0, dot / (na * nb)))


def blend_appearance(old: tuple[float, ...] | None, new: tuple[float, ...] | None, alpha: float = 0.20) -> tuple[float, ...] | None:
    if new is None:
        return old
    if old is None or len(old) != len(new):
        return new
    mixed = tuple((1.0 - alpha) * x + alpha * y for x, y in zip(old, new))
    norm = sum(v * v for v in mixed) ** 0.5
    if norm <= 1e-12:
        return new
    return tuple(v / norm for v in mixed)


def transform_box(box: BBox, affine: tuple[float, float, float, float, float, float]) -> BBox:
    x1, y1, x2, y2 = box
    pts = [
        transform_point(x1, y1, affine),
        transform_point(x2, y1, affine),
        transform_point(x2, y2, affine),
        transform_point(x1, y2, affine),
    ]
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return min(xs), min(ys), max(xs), max(ys)


class MultiObjectTracker:
    """Dependency-light two-stage tracker with camera-motion compensation.

    High-confidence detections create tracks by default. Optional class-specific
    creation thresholds can admit lower-confidence tentative tracks; confirmation
    still requires repeated hits. Track velocity is stored as residual image
    motion after subtracting estimated global camera translation.
    """

    def __init__(
        self,
        max_missed: int = 14,
        min_iou: float = 0.10,
        max_center_ratio: float = 1.9,
        high_conf: float = 0.45,
        low_conf: float = 0.12,
        min_hits: int = 3,
        creation_thresholds: dict[int, float] | None = None,
    ) -> None:
        if not 0.0 <= low_conf <= high_conf <= 1.0:
            raise ValueError("Require 0 <= low_conf <= high_conf <= 1")
        creation_thresholds = dict(creation_thresholds or {})
        if any(not low_conf <= threshold <= 1.0 for threshold in creation_thresholds.values()):
            raise ValueError("creation thresholds must be between low_conf and 1")
        self.max_missed = int(max_missed)
        self.min_iou = float(min_iou)
        self.max_center_ratio = float(max_center_ratio)
        self.high_conf = float(high_conf)
        self.low_conf = float(low_conf)
        self.min_hits = int(min_hits)
        self.creation_thresholds = creation_thresholds
        self._next_id = 1
        self.tracks: dict[int, Track] = {}

    def reset(self) -> None:
        self._next_id = 1
        self.tracks.clear()

    def _predicted_box(
        self,
        track: Track,
        camera_motion: tuple[float, float],
        camera_transform: tuple[float, float, float, float, float, float] | None,
    ) -> BBox:
        factor = max(0.25, 1.0 - track.missed * 0.08)
        if camera_transform is not None:
            camera_box = transform_box(track.bbox, camera_transform)
            return shift_box(camera_box, track.vx * factor, track.vy * factor)
        camera_dx, camera_dy = camera_motion
        return shift_box(track.bbox, camera_dx + track.vx * factor, camera_dy + track.vy * factor)

    def _candidate_score(
        self,
        track: Track,
        det: Detection,
        camera_motion: tuple[float, float],
        camera_transform: tuple[float, float, float, float, float, float] | None,
        loose: bool,
    ) -> float | None:
        if det.class_id != track.class_id:
            return None
        predicted = self._predicted_box(track, camera_motion, camera_transform)
        pcx = (predicted[0] + predicted[2]) * 0.5
        pcy = (predicted[1] + predicted[3]) * 0.5
        dcx, dcy = det.center
        diag = max(hypot(track.width, track.height), 24.0)
        iou = bbox_iou(predicted, det.bbox)
        dist_ratio = hypot(dcx - pcx, dcy - pcy) / diag
        iou_gate = self.min_iou * (0.65 if loose else 1.0)
        center_gate = self.max_center_ratio * (1.25 if loose else 1.0)
        if iou < iou_gate and dist_ratio > center_gate:
            return None
        score = iou * 2.4 + max(0.0, 1.0 - dist_ratio / center_gate) + det.score * 0.15
        appearance = appearance_similarity(track.appearance, det.appearance)
        if appearance is not None:
            score += appearance * 0.55
        return score

    def _associate(
        self,
        track_ids: set[int],
        detections: list[Detection],
        det_ids: set[int],
        camera_motion: tuple[float, float],
        camera_transform: tuple[float, float, float, float, float, float] | None,
        loose: bool = False,
    ) -> list[tuple[int, int]]:
        candidates: list[tuple[float, int, int]] = []
        for tid in track_ids:
            track = self.tracks[tid]
            for didx in det_ids:
                score = self._candidate_score(track, detections[didx], camera_motion, camera_transform, loose)
                if score is not None:
                    candidates.append((score, tid, didx))
        candidates.sort(reverse=True)

        matches: list[tuple[int, int]] = []
        remaining_tracks = set(track_ids)
        remaining_dets = set(det_ids)
        for _, tid, didx in candidates:
            if tid in remaining_tracks and didx in remaining_dets:
                remaining_tracks.remove(tid)
                remaining_dets.remove(didx)
                matches.append((tid, didx))
        return matches

    def _apply_match(
        self,
        track: Track,
        det: Detection,
        camera_motion: tuple[float, float],
        camera_transform: tuple[float, float, float, float, float, float] | None,
    ) -> None:
        old_cx, old_cy = track.center
        new_cx, new_cy = det.center
        if camera_transform is not None:
            cam_cx, cam_cy = transform_point(old_cx, old_cy, camera_transform)
            measured_vx = new_cx - cam_cx
            measured_vy = new_cy - cam_cy
        else:
            camera_dx, camera_dy = camera_motion
            measured_vx = (new_cx - old_cx) - camera_dx
            measured_vy = (new_cy - old_cy) - camera_dy
        track.vx = track.vx * 0.62 + measured_vx * 0.38
        track.vy = track.vy * 0.62 + measured_vy * 0.38
        if track.missed > 0:
            track.recoveries += 1
        track.bbox = det.bbox
        track.score = det.score
        track.last_detection_score = det.score
        track.label = det.label
        track.appearance = blend_appearance(track.appearance, det.appearance)
        track.age += 1
        track.hits += 1
        track.missed = 0
        track.confirmed = track.confirmed or track.hits >= self.min_hits
        track.history.append((int(new_cx), int(new_cy)))

    def predict_only(
        self,
        camera_motion: tuple[float, float] = (0.0, 0.0),
        camera_transform: tuple[float, float, float, float, float, float] | None = None,
    ) -> list[Track]:
        """Advance tracks on an intentionally detector-skipped frame.

        This does not increment missed because no detector observation was
        expected on this frame.
        """
        for track in self.tracks.values():
            track.bbox = self._predicted_box(track, camera_motion, camera_transform)
            track.age += 1
            cx, cy = track.center
            track.history.append((int(cx), int(cy)))
        return sorted(self.tracks.values(), key=lambda t: t.track_id)

    def update(
        self,
        detections: Iterable[Detection],
        camera_motion: tuple[float, float] = (0.0, 0.0),
        camera_transform: tuple[float, float, float, float, float, float] | None = None,
    ) -> list[Track]:
        detections = [d for d in detections if d.score >= self.low_conf]
        all_tracks = set(self.tracks)
        high = {i for i, d in enumerate(detections) if d.score >= self.high_conf}
        low = set(range(len(detections))) - high
        creation_candidates = {
            i
            for i, d in enumerate(detections)
            if d.score >= self.creation_thresholds.get(d.class_id, self.high_conf)
        }

        matches_high = self._associate(all_tracks, detections, high, camera_motion, camera_transform, loose=False)
        used_tracks = {tid for tid, _ in matches_high}
        used_dets = {didx for _, didx in matches_high}

        unmatched_tracks = all_tracks - used_tracks
        low_candidates = low | (high - used_dets)
        matches_low = self._associate(unmatched_tracks, detections, low_candidates, camera_motion, camera_transform, loose=True)

        matches = matches_high + matches_low
        used_tracks |= {tid for tid, _ in matches_low}
        used_dets |= {didx for _, didx in matches_low}

        for tid, didx in matches:
            self._apply_match(self.tracks[tid], detections[didx], camera_motion, camera_transform)

        for tid in all_tracks - used_tracks:
            track = self.tracks[tid]
            track.bbox = self._predicted_box(track, camera_motion, camera_transform)
            track.age += 1
            track.missed += 1
            cx, cy = track.center
            track.history.append((int(cx), int(cy)))

        # New identities use the default high-confidence gate unless an
        # explicit class-specific creation threshold was configured.
        for didx in creation_candidates - used_dets:
            det = detections[didx]
            tid = self._next_id
            self._next_id += 1
            t = Track(
                tid,
                det.bbox,
                det.score,
                det.class_id,
                det.label,
                confirmed=self.min_hits <= 1,
                last_detection_score=det.score,
                appearance=det.appearance,
            )
            cx, cy = det.center
            t.history.append((int(cx), int(cy)))
            self.tracks[tid] = t

        expired = [
            tid for tid, tr in self.tracks.items()
            if tr.missed > self.max_missed or (not tr.confirmed and tr.missed > min(2, self.max_missed))
        ]
        for tid in expired:
            del self.tracks[tid]

        return sorted(self.tracks.values(), key=lambda t: t.track_id)
