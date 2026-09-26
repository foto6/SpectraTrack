from __future__ import annotations

from dataclasses import dataclass
import argparse
import json
import math
import statistics
import time
from typing import Callable, Iterable

from .detection_replay import (
    DETECTION_REPLAY_SCHEMA,
    DetectionReplay,
    ReplayDetection,
    ReplayFrame,
    ReplayMetadata,
    load_detection_replay,
)
from .tracker import MultiObjectTracker, appearance_similarity, bbox_iou, shift_box, transform_box
from .types import BBox, Detection, Track

RESEARCH_BASE_SHA = "d03af3ae6425d3ea2e4d52e25389fecc09957394"
DEFAULT_MATCH_IOU = 0.30


@dataclass(frozen=True, slots=True)
class TruthObject:
    object_id: str
    bbox: BBox
    class_id: int = 0
    label: str = "person"


@dataclass(frozen=True, slots=True)
class ResearchScenario:
    name: str
    replay: DetectionReplay
    truth_by_frame: dict[int, tuple[TruthObject, ...]]
    focus: str


@dataclass(frozen=True, slots=True)
class TrackView:
    track_id: int
    bbox: BBox
    class_id: int
    label: str
    missed: int
    confirmed: bool


@dataclass(slots=True)
class _ReferenceTrack:
    track_id: int
    bbox: BBox
    score: float
    class_id: int
    label: str
    appearance: tuple[float, ...] | None
    hits: int = 1
    missed: int = 0
    confirmed: bool = False
    vx: float = 0.0
    vy: float = 0.0
    previous_observation: tuple[float, float] | None = None
    last_observation: tuple[float, float] | None = None

    @property
    def center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return (x1 + x2) * 0.5, (y1 + y2) * 0.5

    @property
    def width(self) -> float:
        return max(0.0, self.bbox[2] - self.bbox[0])

    @property
    def height(self) -> float:
        return max(0.0, self.bbox[3] - self.bbox[1])


def _box(cx: float, cy: float, width: float = 36.0, height: float = 90.0) -> BBox:
    return (
        cx - width * 0.5,
        cy - height * 0.5,
        cx + width * 0.5,
        cy + height * 0.5,
    )


def _det(
    bbox: BBox,
    score: float = 0.90,
    appearance: tuple[float, ...] | None = None,
    class_id: int = 0,
    label: str = "person",
) -> ReplayDetection:
    return ReplayDetection(bbox, score, class_id, label, appearance)


def _truth(object_id: str, bbox: BBox, class_id: int = 0, label: str = "person") -> TruthObject:
    return TruthObject(object_id, bbox, class_id, label)


def _make_scenario(
    name: str,
    rows: Iterable[
        tuple[
            tuple[ReplayDetection, ...],
            tuple[TruthObject, ...],
            bool,
            tuple[float, float],
            tuple[float, float, float, float, float, float] | None,
        ]
    ],
    focus: str,
) -> ResearchScenario:
    width, height = 640, 360
    frames: list[ReplayFrame] = []
    truth_by_frame: dict[int, tuple[TruthObject, ...]] = {}
    for index, (detections, truth, detector_ran, camera_motion, camera_transform) in enumerate(rows):
        frames.append(
            ReplayFrame(
                video=f"synthetic/{name}.mp4",
                frame=index,
                timestamp_s=index / 30.0,
                width=width,
                height=height,
                detections=detections,
                detector_ran=detector_ran,
                camera_motion=camera_motion,
                camera_transform=camera_transform,
            )
        )
        truth_by_frame[index] = truth
    replay = DetectionReplay(
        metadata=ReplayMetadata(
            source_commit=RESEARCH_BASE_SHA,
            video=f"synthetic/{name}.mp4",
            video_sha256=None,
            detector="synthetic-deterministic-v1",
            model_sha256=None,
            provider="CPUExecutionProvider",
            config={"scenario": name, "detector_inference_calls": 0},
            width=width,
            height=height,
        ),
        frames=tuple(frames),
    )
    return ResearchScenario(name=name, replay=replay, truth_by_frame=truth_by_frame, focus=focus)


def synthetic_scenarios() -> tuple[ResearchScenario, ...]:
    appearance_a = (1.0, 0.0, 0.0, 0.0)
    appearance_b = (0.0, 1.0, 0.0, 0.0)
    scenarios: list[ResearchScenario] = []

    crossing_rows = []
    for ax, bx in [(120, 420), (160, 380), (200, 340), (240, 300), (270, 270), (300, 240), (340, 200), (380, 160)]:
        a = _box(ax, 180)
        b = _box(bx, 180)
        crossing_rows.append(((_det(a), _det(b)), (_truth("a", a), _truth("b", b)), True, (0.0, 0.0), None))
    scenarios.append(
        _make_scenario(
            "crossing",
            crossing_rows,
            "Greedy one-to-one assignment becomes ambiguous at equal-overlap crossing; tie/order can select the wrong identity.",
        )
    )

    partial_rows = []
    for frame in range(8):
        truth_box = _box(180 + frame * 8, 180)
        if frame == 4:
            det_box = _box(180 + frame * 8, 180, 18, 78)
            detection = _det(det_box, score=0.28, appearance=appearance_a)
        else:
            detection = _det(truth_box, appearance=appearance_a)
        partial_rows.append(((detection,), (_truth("p", truth_box),), True, (0.0, 0.0), None))
    scenarios.append(
        _make_scenario(
            "partial_occlusion",
            partial_rows,
            "Weak/narrow partial-occlusion box should enter the existing low-confidence rescue stage, not create a new ID.",
        )
    )

    full_rows = []
    for frame in range(12):
        if 4 <= frame <= 7:
            full_rows.append(((), (), True, (0.0, 0.0), None))
        else:
            box = _box(220 + max(0, frame - 7) * 4, 180)
            full_rows.append(((_det(box, appearance=appearance_a),), (_truth("p", box),), True, (0.0, 0.0), None))
    scenarios.append(
        _make_scenario(
            "full_occlusion",
            full_rows,
            "No visible GT exists during occlusion; active-track lifetime must bridge the gap without producing visible false identity changes.",
        )
    )

    short_rows = []
    for frame in range(10):
        box = _box(130 + frame * 7, 180)
        detections = () if frame in {4, 5} else (_det(box, appearance=appearance_a),)
        short_rows.append((detections, (_truth("p", box),), True, (0.0, 0.0), None))
    scenarios.append(
        _make_scenario(
            "short_dropout",
            short_rows,
            "Two detector misses exercise predicted geometry and max_missed without entering dormant state.",
        )
    )

    long_rows = []
    for frame in range(24):
        box = _box(220, 180)
        detections = () if 4 <= frame <= 19 else (_det(box),)
        long_rows.append((detections, (_truth("p", box),), True, (0.0, 0.0), None))
    scenarios.append(
        _make_scenario(
            "long_dropout",
            long_rows,
            "A long detector dropout without appearance causes deletion because dormant reactivation is intentionally unavailable without appearance.",
        )
    )

    dormant_rows = []
    for frame in range(24):
        box = _box(220, 180)
        detections = () if 4 <= frame <= 19 else (_det(box, appearance=appearance_a),)
        dormant_rows.append((detections, (_truth("p", box),), True, (0.0, 0.0), None))
    scenarios.append(
        _make_scenario(
            "dormant_reactivation",
            dormant_rows,
            "Long loss with stable appearance should move confirmed track to dormant pool and recover the original ID.",
        )
    )

    mismatch_rows = []
    for frame in range(24):
        box = _box(220, 180)
        if 4 <= frame <= 19:
            detections = ()
        else:
            appearance = appearance_b if frame >= 20 else appearance_a
            detections = (_det(box, appearance=appearance),)
        mismatch_rows.append((detections, (_truth("p", box),), True, (0.0, 0.0), None))
    scenarios.append(
        _make_scenario(
            "appearance_mismatch",
            mismatch_rows,
            "Dormant candidate should be rejected by reactivation_min_appearance and a fresh strong ID is created.",
        )
    )

    pan_rows = []
    for frame in range(9):
        box = _box(140 + frame * 30, 180)
        motion = (0.0, 0.0) if frame == 0 else (30.0, 0.0)
        pan_rows.append(((_det(box),), (_truth("p", box),), True, motion, None))
    scenarios.append(
        _make_scenario(
            "camera_pan",
            pan_rows,
            "Camera translation should be removed before residual target velocity and association gating are evaluated.",
        )
    )

    transform_rows = []
    box = _box(160, 150, 40, 80)
    transform_rows.append(((_det(box),), (_truth("p", box),), True, (0.0, 0.0), None))
    affine = (1.02, 0.0, 12.0, 0.0, 1.02, 4.0)
    for _ in range(1, 8):
        box = transform_box(box, affine)
        transform_rows.append(((_det(box),), (_truth("p", box),), True, (0.0, 0.0), affine))
    scenarios.append(
        _make_scenario(
            "camera_transform",
            transform_rows,
            "Affine CMC should transform the predicted box before geometry scoring; scale and translation must not look like target motion.",
        )
    )

    skipped_rows = []
    for frame in range(12):
        box = _box(120 + frame * 6, 180)
        detector_ran = frame % 2 == 0
        detections = (_det(box),) if detector_ran else ()
        skipped_rows.append((detections, (_truth("p", box),), detector_ran, (0.0, 0.0), None))
    scenarios.append(
        _make_scenario(
            "detector_skipped_frames",
            skipped_rows,
            "Intentional skipped frames must call predict_only and must not consume max_missed budget.",
        )
    )

    weak_rows = []
    for frame in range(12):
        box = _box(120 + frame * 5, 180)
        score = 0.9 if frame < 3 else 0.20
        weak_rows.append(((_det(box, score=score),), (_truth("p", box),), True, (0.0, 0.0), None))
    scenarios.append(
        _make_scenario(
            "weak_detection_sequence",
            weak_rows,
            "Existing low-confidence second stage should maintain an established track; this is the ByteTrack-like behavior already present.",
        )
    )

    false_weak_rows = []
    for frame in range(8):
        box = _box(500, 180)
        false_weak_rows.append(((_det(box, score=0.20),), (), True, (0.0, 0.0), None))
    scenarios.append(
        _make_scenario(
            "false_weak_detections",
            false_weak_rows,
            "Low-confidence detections below high_conf must not create new identities.",
        )
    )

    nearby_rows = []
    for _ in range(3):
        a = _box(200, 180, 30, 90)
        b = _box(300, 180, 90, 90)
        nearby_rows.append(((_det(a), _det(b)), (_truth("a", a), _truth("b", b)), True, (0.0, 0.0), None))
    for ax, bx in [(240, 250), (270, 230), (300, 200), (330, 170)]:
        a = _box(ax, 180, 80, 90)
        b = _box(bx, 180, 30, 90)
        nearby_rows.append(((_det(a), _det(b)), (_truth("a", a), _truth("b", b)), True, (0.0, 0.0), None))
    scenarios.append(
        _make_scenario(
            "nearby_same_class",
            nearby_rows,
            (
                "Nearby same-class people with asymmetric bbox geometry expose a greedy-assignment conflict: "
                "one locally best pair can block the better total one-to-one assignment."
            ),
        )
    )

    scale_rows = []
    for scale in (1.0, 1.08, 1.18, 1.32, 1.50, 1.70, 1.85, 2.0):
        box = _box(240, 180, 36 * scale, 90 * scale)
        scale_rows.append(((_det(box, appearance=appearance_a),), (_truth("p", box),), True, (0.0, 0.0), None))
    scenarios.append(
        _make_scenario(
            "changing_bbox_scale",
            scale_rows,
            "Center-distance gate can preserve identity through large scale change even when IoU falls; this tests whether scale alone fragments the track.",
        )
    )

    direction_rows = []
    xs = [100, 135, 170, 205, 240, 80, 55, 30, 20]
    for x in xs:
        box = _box(x, 180)
        direction_rows.append(((_det(box),), (_truth("p", box),), True, (0.0, 0.0), None))
    scenarios.append(
        _make_scenario(
            "sudden_direction_change",
            direction_rows,
            "Large reversal tests the hard geometry gate after constant-velocity prediction; a strong unmatched detection may spawn a new ID.",
        )
    )

    return tuple(scenarios)


def _view(track: Track | _ReferenceTrack) -> TrackView:
    return TrackView(
        track_id=int(track.track_id),
        bbox=tuple(track.bbox),
        class_id=int(track.class_id),
        label=str(track.label),
        missed=int(track.missed),
        confirmed=bool(track.confirmed),
    )


def _predicted_current_box(
    tracker: MultiObjectTracker,
    track: Track,
    frame: ReplayFrame,
) -> BBox:
    return tracker._predicted_box(track, frame.camera_motion, frame.camera_transform)


def _current_candidate_diagnostic(
    tracker: MultiObjectTracker,
    track: Track,
    detection: Detection,
    frame: ReplayFrame,
    loose: bool,
) -> dict:
    if detection.class_id != track.class_id:
        return {"accepted": False, "reason": "class_mismatch"}
    predicted = _predicted_current_box(tracker, track, frame)
    pcx = (predicted[0] + predicted[2]) * 0.5
    pcy = (predicted[1] + predicted[3]) * 0.5
    dcx, dcy = detection.center
    diag = max(math.hypot(track.width, track.height), 24.0)
    iou = bbox_iou(predicted, detection.bbox)
    distance_ratio = math.hypot(dcx - pcx, dcy - pcy) / diag
    iou_gate = tracker.min_iou * (0.65 if loose else 1.0)
    center_gate = tracker.max_center_ratio * (1.25 if loose else 1.0)
    accepted = not (iou < iou_gate and distance_ratio > center_gate)
    appearance = appearance_similarity(track.appearance, detection.appearance)
    score = None
    if accepted:
        score = iou * 2.4 + max(0.0, 1.0 - distance_ratio / center_gate) + detection.score * 0.15
        if appearance is not None:
            score += appearance * 0.55
    return {
        "accepted": accepted,
        "reason": "candidate" if accepted else "geometry_gate",
        "iou": round(iou, 6),
        "iou_gate": round(iou_gate, 6),
        "center_ratio": round(distance_ratio, 6),
        "center_gate": round(center_gate, 6),
        "appearance": None if appearance is None else round(appearance, 6),
        "score": None if score is None else round(score, 6),
    }


class AuditedCurrentTracker(MultiObjectTracker):
    def __init__(self) -> None:
        super().__init__()
        self.audit_frame = -1
        self.audit_events: list[dict] = []

    def _associate(
        self,
        track_ids: set[int],
        detections: list[Detection],
        det_ids: set[int],
        camera_motion: tuple[float, float],
        camera_transform: tuple[float, float, float, float, float, float] | None,
        loose: bool = False,
    ) -> list[tuple[int, int]]:
        frame = ReplayFrame(
            video="audit",
            frame=self.audit_frame,
            timestamp_s=None,
            width=1,
            height=1,
            detections=(),
            camera_motion=camera_motion,
            camera_transform=camera_transform,
        )
        candidates = []
        for track_id in sorted(track_ids):
            track = self.tracks[track_id]
            for detection_id in sorted(det_ids):
                diagnostic = _current_candidate_diagnostic(
                    self,
                    track,
                    detections[detection_id],
                    frame,
                    loose,
                )
                candidates.append(
                    {
                        "track_id": track_id,
                        "detection_index": detection_id,
                        **diagnostic,
                    }
                )
        matches = super()._associate(
            track_ids,
            detections,
            det_ids,
            camera_motion,
            camera_transform,
            loose,
        )
        self.audit_events.append(
            {
                "frame": self.audit_frame,
                "event": "association",
                "stage": "low_or_unmatched" if loose else "high",
                "candidates": candidates,
                "matches": [{"track_id": tid, "detection_index": didx} for tid, didx in matches],
            }
        )
        return matches

    def _reactivate(self, detections: list[Detection], det_ids: set[int]) -> list[tuple[int, int]]:
        candidates = []
        for track_id, (track, dormant_age) in sorted(self._dormant_tracks.items()):
            for detection_id in sorted(det_ids):
                score = self._reactivation_score(track, detections[detection_id])
                appearance = appearance_similarity(track.appearance, detections[detection_id].appearance)
                candidates.append(
                    {
                        "track_id": track_id,
                        "dormant_age": dormant_age,
                        "detection_index": detection_id,
                        "appearance": None if appearance is None else round(appearance, 6),
                        "accepted": score is not None,
                        "score": None if score is None else round(score, 6),
                    }
                )
        matches = super()._reactivate(detections, det_ids)
        self.audit_events.append(
            {
                "frame": self.audit_frame,
                "event": "reactivation",
                "candidates": candidates,
                "matches": [{"track_id": tid, "detection_index": didx} for tid, didx in matches],
            }
        )
        return matches


class GlobalAssignmentCurrentTracker(MultiObjectTracker):
    """Current SpectraTrack lifecycle/gates with only assignment selection changed."""

    def _associate(
        self,
        track_ids: set[int],
        detections: list[Detection],
        det_ids: set[int],
        camera_motion: tuple[float, float],
        camera_transform: tuple[float, float, float, float, float, float] | None,
        loose: bool = False,
    ) -> list[tuple[int, int]]:
        ordered_tracks = sorted(track_ids)
        ordered_detections = sorted(det_ids)
        scores = [
            [
                self._candidate_score(
                    self.tracks[track_id],
                    detections[detection_id],
                    camera_motion,
                    camera_transform,
                    loose,
                )
                for detection_id in ordered_detections
            ]
            for track_id in ordered_tracks
        ]
        return [(ordered_tracks[row], ordered_detections[column]) for row, column in _maximize_assignment(scores)]


class GlobalAssignmentCurrentRunner:
    name = "current-global-assignment"

    def __init__(self) -> None:
        self.tracker = GlobalAssignmentCurrentTracker()

    def step(self, frame: ReplayFrame) -> list[TrackView]:
        if frame.detector_ran:
            tracks = self.tracker.update(
                frame.fresh_detections(),
                camera_motion=frame.camera_motion,
                camera_transform=frame.camera_transform,
            )
        else:
            tracks = self.tracker.predict_only(
                camera_motion=frame.camera_motion,
                camera_transform=frame.camera_transform,
            )
        return [_view(track) for track in tracks]


class AmbiguityGuardCurrentTracker(MultiObjectTracker):
    """Current tracker with global assignment only inside ambiguous score components."""

    def __init__(self, ambiguity_margin: float = 0.08) -> None:
        super().__init__()
        if ambiguity_margin < 0.0:
            raise ValueError("ambiguity_margin must be >= 0")
        self.ambiguity_margin = float(ambiguity_margin)

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
        score_by_pair: dict[tuple[int, int], float] = {}
        by_track: dict[int, list[tuple[float, int]]] = {}
        by_detection: dict[int, list[tuple[float, int]]] = {}
        for track_id in sorted(track_ids):
            track = self.tracks[track_id]
            for detection_id in sorted(det_ids):
                score = self._candidate_score(
                    track,
                    detections[detection_id],
                    camera_motion,
                    camera_transform,
                    loose,
                )
                if score is None:
                    continue
                candidates.append((score, track_id, detection_id))
                score_by_pair[(track_id, detection_id)] = score
                by_track.setdefault(track_id, []).append((score, detection_id))
                by_detection.setdefault(detection_id, []).append((score, track_id))

        if not candidates:
            return []

        competitive_edges: set[tuple[int, int]] = set()
        for track_id, rows in by_track.items():
            ranked = sorted(rows, reverse=True)
            if len(ranked) < 2 or ranked[0][0] - ranked[1][0] > self.ambiguity_margin:
                continue
            best = ranked[0][0]
            competitive_edges.update(
                (track_id, detection_id)
                for score, detection_id in ranked
                if best - score <= self.ambiguity_margin
            )
        for detection_id, rows in by_detection.items():
            ranked = sorted(rows, reverse=True)
            if len(ranked) < 2 or ranked[0][0] - ranked[1][0] > self.ambiguity_margin:
                continue
            best = ranked[0][0]
            competitive_edges.update(
                (track_id, detection_id)
                for score, track_id in ranked
                if best - score <= self.ambiguity_margin
            )

        if not competitive_edges:
            candidates.sort(reverse=True)
            remaining_tracks = set(track_ids)
            remaining_detections = set(det_ids)
            matches: list[tuple[int, int]] = []
            for _, track_id, detection_id in candidates:
                if track_id not in remaining_tracks or detection_id not in remaining_detections:
                    continue
                remaining_tracks.remove(track_id)
                remaining_detections.remove(detection_id)
                matches.append((track_id, detection_id))
            return matches

        track_neighbors: dict[int, set[int]] = {}
        detection_neighbors: dict[int, set[int]] = {}
        for track_id, detection_id in competitive_edges:
            track_neighbors.setdefault(track_id, set()).add(detection_id)
            detection_neighbors.setdefault(detection_id, set()).add(track_id)

        components: list[tuple[set[int], set[int]]] = []
        unseen_tracks = set(track_neighbors)
        while unseen_tracks:
            seed = min(unseen_tracks)
            component_tracks: set[int] = set()
            component_detections: set[int] = set()
            pending_tracks = [seed]
            pending_detections: list[int] = []
            while pending_tracks or pending_detections:
                while pending_tracks:
                    track_id = pending_tracks.pop()
                    if track_id in component_tracks:
                        continue
                    component_tracks.add(track_id)
                    unseen_tracks.discard(track_id)
                    for detection_id in track_neighbors.get(track_id, ()):
                        if detection_id not in component_detections:
                            pending_detections.append(detection_id)
                while pending_detections:
                    detection_id = pending_detections.pop()
                    if detection_id in component_detections:
                        continue
                    component_detections.add(detection_id)
                    for track_id in detection_neighbors.get(detection_id, ()):
                        if track_id not in component_tracks:
                            pending_tracks.append(track_id)
            components.append((component_tracks, component_detections))

        matches: list[tuple[int, int]] = []
        used_tracks: set[int] = set()
        used_detections: set[int] = set()
        for component_tracks, component_detections in components:
            if len(component_tracks) < 2 or len(component_detections) < 2:
                continue
            ordered_tracks = sorted(component_tracks)
            ordered_detections = sorted(component_detections)
            scores = [
                [score_by_pair.get((track_id, detection_id)) for detection_id in ordered_detections]
                for track_id in ordered_tracks
            ]
            for row, column in _maximize_assignment(scores):
                track_id = ordered_tracks[row]
                detection_id = ordered_detections[column]
                matches.append((track_id, detection_id))
                used_tracks.add(track_id)
                used_detections.add(detection_id)

        candidates.sort(reverse=True)
        for _, track_id, detection_id in candidates:
            if track_id in used_tracks or detection_id in used_detections:
                continue
            used_tracks.add(track_id)
            used_detections.add(detection_id)
            matches.append((track_id, detection_id))
        return matches


class AmbiguityGuardCurrentRunner:
    name = "current-ambiguity-guard"

    def __init__(self, ambiguity_margin: float = 0.08) -> None:
        self.tracker = AmbiguityGuardCurrentTracker(ambiguity_margin=ambiguity_margin)

    def step(self, frame: ReplayFrame) -> list[TrackView]:
        if frame.detector_ran:
            tracks = self.tracker.update(
                frame.fresh_detections(),
                camera_motion=frame.camera_motion,
                camera_transform=frame.camera_transform,
            )
        else:
            tracks = self.tracker.predict_only(
                camera_motion=frame.camera_motion,
                camera_transform=frame.camera_transform,
            )
        return [_view(track) for track in tracks]


class CurrentTrackerRunner:
    name = "current"

    def __init__(self, audit: bool = False) -> None:
        self.tracker: MultiObjectTracker
        self.tracker = AuditedCurrentTracker() if audit else MultiObjectTracker()

    @property
    def audit_events(self) -> list[dict]:
        if isinstance(self.tracker, AuditedCurrentTracker):
            return self.tracker.audit_events
        return []

    def step(self, frame: ReplayFrame) -> list[TrackView]:
        if isinstance(self.tracker, AuditedCurrentTracker):
            self.tracker.audit_frame = frame.frame
        if frame.detector_ran:
            tracks = self.tracker.update(
                frame.fresh_detections(),
                camera_motion=frame.camera_motion,
                camera_transform=frame.camera_transform,
            )
        else:
            tracks = self.tracker.predict_only(
                camera_motion=frame.camera_motion,
                camera_transform=frame.camera_transform,
            )
        return [_view(track) for track in tracks]


def _center(box: BBox) -> tuple[float, float]:
    return (box[0] + box[2]) * 0.5, (box[1] + box[3]) * 0.5


def _dimensions(box: BBox) -> tuple[float, float]:
    return max(0.0, box[2] - box[0]), max(0.0, box[3] - box[1])


def _direction_similarity(
    vx: float,
    vy: float,
    dx: float,
    dy: float,
) -> float | None:
    first = math.hypot(vx, vy)
    second = math.hypot(dx, dy)
    if first < 1e-6 or second < 1e-6:
        return None
    cosine = (vx * dx + vy * dy) / (first * second)
    return max(0.0, min(1.0, (cosine + 1.0) * 0.5))


def _hungarian_min(cost: list[list[float]]) -> list[tuple[int, int]]:
    if not cost or not cost[0]:
        return []
    rows = len(cost)
    cols = len(cost[0])
    transposed = False
    matrix = cost
    if rows > cols:
        transposed = True
        matrix = [[cost[row][col] for row in range(rows)] for col in range(cols)]
        rows, cols = cols, rows

    u = [0.0] * (rows + 1)
    v = [0.0] * (cols + 1)
    p = [0] * (cols + 1)
    way = [0] * (cols + 1)
    for i in range(1, rows + 1):
        p[0] = i
        j0 = 0
        minimum = [float("inf")] * (cols + 1)
        used = [False] * (cols + 1)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = float("inf")
            j1 = 0
            for j in range(1, cols + 1):
                if used[j]:
                    continue
                current = matrix[i0 - 1][j - 1] - u[i0] - v[j]
                if current < minimum[j]:
                    minimum[j] = current
                    way[j] = j0
                if minimum[j] < delta:
                    delta = minimum[j]
                    j1 = j
            for j in range(cols + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minimum[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break
    result = []
    for column in range(1, cols + 1):
        if p[column] == 0:
            continue
        row = p[column] - 1
        col = column - 1
        result.append((col, row) if transposed else (row, col))
    return result


def _maximize_assignment(scores: list[list[float | None]]) -> list[tuple[int, int]]:
    if not scores or not scores[0]:
        return []
    valid_values = [value for row in scores for value in row if value is not None]
    if not valid_values:
        return []
    maximum = max(valid_values)
    invalid_cost = maximum + 1_000_000.0
    cost = [[invalid_cost if value is None else maximum - value for value in row] for row in scores]
    assigned = _hungarian_min(cost)
    return [(row, column) for row, column in assigned if scores[row][column] is not None]


class ReferenceStyleTracker:
    """Dependency-free mechanism probe; not a vendored official tracker."""

    def __init__(self, mode: str) -> None:
        if mode not in {"byte", "botsort", "ocsort"}:
            raise ValueError("mode must be byte, botsort, or ocsort")
        self.mode = mode
        self.name = {
            "byte": "byte-global-reference-style",
            "botsort": "botsort-reference-style",
            "ocsort": "ocsort-style",
        }[mode]
        self.high_conf = 0.45
        self.low_conf = 0.12
        self.min_iou = 0.10
        self.max_center_ratio = 1.9
        self.max_missed = 14
        self.min_hits = 3
        self.next_id = 1
        self.tracks: dict[int, _ReferenceTrack] = {}

    def _camera_box(self, track: _ReferenceTrack, frame: ReplayFrame) -> BBox:
        if self.mode == "botsort":
            if frame.camera_transform is not None:
                return transform_box(track.bbox, frame.camera_transform)
            if frame.camera_motion != (0.0, 0.0):
                return shift_box(track.bbox, *frame.camera_motion)
        return track.bbox

    def _predict(self, track: _ReferenceTrack, frame: ReplayFrame) -> BBox:
        box = self._camera_box(track, frame)
        factor = max(0.25, 1.0 - track.missed * 0.08)
        return shift_box(box, track.vx * factor, track.vy * factor)

    def _similarity(
        self,
        track: _ReferenceTrack,
        detection: Detection,
        frame: ReplayFrame,
        loose: bool,
    ) -> float | None:
        if track.class_id != detection.class_id:
            return None
        predicted = self._predict(track, frame)
        iou = bbox_iou(predicted, detection.bbox)
        pcx, pcy = _center(predicted)
        dcx, dcy = detection.center
        width, height = _dimensions(track.bbox)
        diagonal = max(math.hypot(width, height), 24.0)
        center_ratio = math.hypot(dcx - pcx, dcy - pcy) / diagonal
        iou_gate = self.min_iou * (0.65 if loose else 1.0)
        center_gate = self.max_center_ratio * (1.25 if loose else 1.0)
        if iou < iou_gate and center_ratio > center_gate:
            return None
        center_score = max(0.0, 1.0 - center_ratio / center_gate)
        if self.mode == "byte":
            return iou * 0.72 + center_score * 0.23 + detection.score * 0.05
        if self.mode == "botsort":
            appearance = appearance_similarity(track.appearance, detection.appearance)
            if appearance is None:
                return iou * 0.70 + center_score * 0.25 + detection.score * 0.05
            return iou * 0.48 + center_score * 0.17 + appearance * 0.30 + detection.score * 0.05

        tcx, tcy = track.center
        direction = _direction_similarity(track.vx, track.vy, dcx - tcx, dcy - tcy)
        direction_score = 0.5 if direction is None else direction
        return iou * 0.55 + center_score * 0.20 + direction_score * 0.20 + detection.score * 0.05

    def _associate(
        self,
        track_ids: list[int],
        detections: list[Detection],
        detection_ids: list[int],
        frame: ReplayFrame,
        loose: bool,
    ) -> list[tuple[int, int]]:
        scores = [
            [
                self._similarity(self.tracks[track_id], detections[detection_id], frame, loose)
                for detection_id in detection_ids
            ]
            for track_id in track_ids
        ]
        return [(track_ids[row], detection_ids[column]) for row, column in _maximize_assignment(scores)]

    def _apply(self, track: _ReferenceTrack, detection: Detection, frame: ReplayFrame) -> None:
        old_center = _center(self._camera_box(track, frame))
        new_center = detection.center
        measured_vx = new_center[0] - old_center[0]
        measured_vy = new_center[1] - old_center[1]
        if self.mode == "ocsort" and track.last_observation is not None:
            if track.previous_observation is not None:
                measured_vx = track.last_observation[0] - track.previous_observation[0]
                measured_vy = track.last_observation[1] - track.previous_observation[1]
            track.previous_observation = track.last_observation
            track.last_observation = new_center
            track.vx = measured_vx
            track.vy = measured_vy
        else:
            track.vx = track.vx * 0.62 + measured_vx * 0.38
            track.vy = track.vy * 0.62 + measured_vy * 0.38
            track.previous_observation = track.last_observation
            track.last_observation = new_center
        track.bbox = detection.bbox
        track.score = detection.score
        track.appearance = detection.appearance if detection.appearance is not None else track.appearance
        track.hits += 1
        track.missed = 0
        track.confirmed = track.confirmed or track.hits >= self.min_hits

    def step(self, frame: ReplayFrame) -> list[TrackView]:
        if not frame.detector_ran:
            for track in self.tracks.values():
                track.bbox = self._predict(track, frame)
            return [_view(track) for track in sorted(self.tracks.values(), key=lambda item: item.track_id)]

        detections = [item for item in frame.fresh_detections() if item.score >= self.low_conf]
        high = [index for index, item in enumerate(detections) if item.score >= self.high_conf]
        low = [index for index, item in enumerate(detections) if item.score < self.high_conf]
        active_ids = sorted(self.tracks)
        high_matches = self._associate(active_ids, detections, high, frame, False)
        used_tracks = {track_id for track_id, _ in high_matches}
        used_detections = {detection_id for _, detection_id in high_matches}
        remaining_tracks = [track_id for track_id in active_ids if track_id not in used_tracks]
        remaining_detections = low + [item for item in high if item not in used_detections]
        low_matches = self._associate(remaining_tracks, detections, remaining_detections, frame, True)
        matches = high_matches + low_matches
        used_tracks |= {track_id for track_id, _ in low_matches}
        used_detections |= {detection_id for _, detection_id in low_matches}

        for track_id, detection_id in matches:
            self._apply(self.tracks[track_id], detections[detection_id], frame)

        for track_id in set(active_ids) - used_tracks:
            track = self.tracks[track_id]
            track.bbox = self._predict(track, frame)
            track.missed += 1

        for track_id in list(self.tracks):
            if self.tracks[track_id].missed > self.max_missed:
                del self.tracks[track_id]

        for detection_id in high:
            if detection_id in used_detections:
                continue
            detection = detections[detection_id]
            center = detection.center
            track = _ReferenceTrack(
                track_id=self.next_id,
                bbox=detection.bbox,
                score=detection.score,
                class_id=detection.class_id,
                label=detection.label,
                appearance=detection.appearance,
                confirmed=self.min_hits <= 1,
                last_observation=center,
            )
            self.tracks[self.next_id] = track
            self.next_id += 1

        return [_view(track) for track in sorted(self.tracks.values(), key=lambda item: item.track_id)]


def _match_truth(
    truth: tuple[TruthObject, ...],
    tracks: list[TrackView],
    threshold: float = DEFAULT_MATCH_IOU,
) -> dict[int, int]:
    scores = []
    for truth_item in truth:
        row: list[float | None] = []
        for track in tracks:
            if truth_item.class_id != track.class_id:
                row.append(None)
                continue
            iou = bbox_iou(truth_item.bbox, track.bbox)
            row.append(iou if iou >= threshold else None)
        scores.append(row)
    return {row: column for row, column in _maximize_assignment(scores)}


def evaluate_tracking(
    scenario: ResearchScenario,
    outputs: dict[int, list[TrackView]],
) -> dict[str, float | int | None]:
    matched_gt = 0
    total_gt = 0
    id_switches = 0
    fragmentations = 0
    false_track_ids: set[int] = set()
    identity_state: dict[str, dict[str, object]] = {}
    run_lengths: list[int] = []
    current_runs: dict[str, tuple[int | None, int]] = {}
    recovery_latencies: list[int] = []
    recovery_wait: dict[str, int] = {}
    recovered_same_id = 0
    wrong_recovery = 0
    previous_visible: set[str] = set()
    last_visible_track: dict[str, int | None] = {}
    occluded_since: set[str] = set()

    for frame in scenario.replay.frames:
        truth = scenario.truth_by_frame.get(frame.frame, ())
        tracks = outputs.get(frame.frame, [])
        total_gt += len(truth)
        matches = _match_truth(truth, tracks)
        matched_gt += len(matches)
        used_tracks = {tracks[column].track_id for column in matches.values()}
        for track in tracks:
            if track.missed == 0 and track.track_id not in used_tracks:
                false_track_ids.add(track.track_id)

        visible_ids = {item.object_id for item in truth}
        for disappeared in previous_visible - visible_ids:
            occluded_since.add(disappeared)
        previous_visible = visible_ids

        for truth_index, truth_item in enumerate(truth):
            state = identity_state.setdefault(
                truth_item.object_id,
                {"seen": False, "last_track_id": None, "gap": False},
            )
            track_column = matches.get(truth_index)
            if track_column is None:
                if state["seen"]:
                    state["gap"] = True
                    recovery_wait[truth_item.object_id] = recovery_wait.get(truth_item.object_id, 0) + 1
                previous_track, length = current_runs.get(truth_item.object_id, (None, 0))
                if length:
                    run_lengths.append(length)
                current_runs[truth_item.object_id] = (previous_track, 0)
                continue

            track_id = tracks[track_column].track_id
            if state["seen"] and state["last_track_id"] != track_id:
                id_switches += 1
            if state["seen"] and state["gap"]:
                fragmentations += 1
                if state["last_track_id"] == track_id:
                    recovered_same_id += 1
                else:
                    wrong_recovery += 1
            if truth_item.object_id in recovery_wait:
                recovery_latencies.append(recovery_wait.pop(truth_item.object_id))
            if truth_item.object_id in occluded_since:
                if last_visible_track.get(truth_item.object_id) == track_id:
                    recovered_same_id += 1
                else:
                    wrong_recovery += 1
                occluded_since.discard(truth_item.object_id)
            state["seen"] = True
            state["last_track_id"] = track_id
            state["gap"] = False
            last_visible_track[truth_item.object_id] = track_id

            previous_track, length = current_runs.get(truth_item.object_id, (None, 0))
            if previous_track == track_id:
                current_runs[truth_item.object_id] = (track_id, length + 1)
            else:
                if length:
                    run_lengths.append(length)
                current_runs[truth_item.object_id] = (track_id, 1)

    for _, length in current_runs.values():
        if length:
            run_lengths.append(length)

    return {
        "id_switches": id_switches,
        "fragmentations": fragmentations,
        "tracking_recall": matched_gt / total_gt if total_gt else 1.0,
        "false_track_creations": len(false_track_ids),
        "mean_uninterrupted_track_length": statistics.fmean(run_lengths) if run_lengths else 0.0,
        "uninterrupted_segments": len(run_lengths),
        "mean_recovery_latency_frames": statistics.fmean(recovery_latencies) if recovery_latencies else None,
        "recovery_events": len(recovery_latencies),
        "recovered_same_id": recovered_same_id,
        "wrong_recovery": wrong_recovery,
        "matched_gt": matched_gt,
        "gt": total_gt,
    }


def run_candidate(
    scenario: ResearchScenario,
    factory: Callable[[], CurrentTrackerRunner | ReferenceStyleTracker],
) -> tuple[dict[int, list[TrackView]], dict[str, float]]:
    runner = factory()
    outputs: dict[int, list[TrackView]] = {}
    wall_start = time.perf_counter()
    cpu_start = time.process_time()
    for frame in scenario.replay.frames:
        outputs[frame.frame] = runner.step(frame)
    cpu_seconds = time.process_time() - cpu_start
    wall_seconds = time.perf_counter() - wall_start
    return outputs, {"wall_seconds": wall_seconds, "cpu_seconds": cpu_seconds}


def _aggregate(metrics: Iterable[dict[str, float | int | None]]) -> dict[str, float | int | None]:
    rows = list(metrics)
    gt = sum(int(row["gt"]) for row in rows)
    matched = sum(int(row["matched_gt"]) for row in rows)
    run_segments = sum(int(row["uninterrupted_segments"]) for row in rows)
    run_length_total = sum(
        float(row["mean_uninterrupted_track_length"]) * int(row["uninterrupted_segments"]) for row in rows
    )
    recovery_events = sum(int(row["recovery_events"]) for row in rows)
    recovery_latency_total = sum(
        float(row["mean_recovery_latency_frames"]) * int(row["recovery_events"])
        for row in rows
        if row["mean_recovery_latency_frames"] is not None
    )
    return {
        "id_switches": sum(int(row["id_switches"]) for row in rows),
        "fragmentations": sum(int(row["fragmentations"]) for row in rows),
        "tracking_recall": matched / gt if gt else 1.0,
        "false_track_creations": sum(int(row["false_track_creations"]) for row in rows),
        "mean_uninterrupted_track_length": run_length_total / run_segments if run_segments else 0.0,
        "uninterrupted_segments": run_segments,
        "mean_recovery_latency_frames": (recovery_latency_total / recovery_events if recovery_events else None),
        "recovery_events": recovery_events,
        "recovered_same_id": sum(int(row["recovered_same_id"]) for row in rows),
        "wrong_recovery": sum(int(row["wrong_recovery"]) for row in rows),
        "matched_gt": matched,
        "gt": gt,
    }


def _candidate_factories() -> dict[
    str,
    Callable[
        [],
        CurrentTrackerRunner | GlobalAssignmentCurrentRunner | AmbiguityGuardCurrentRunner | ReferenceStyleTracker,
    ],
]:
    return {
        "current": lambda: CurrentTrackerRunner(),
        "current-global-assignment": lambda: GlobalAssignmentCurrentRunner(),
        "current-ambiguity-guard": lambda: AmbiguityGuardCurrentRunner(),
        "byte-global-reference-style": lambda: ReferenceStyleTracker("byte"),
        "botsort-reference-style": lambda: ReferenceStyleTracker("botsort"),
        "ocsort-style": lambda: ReferenceStyleTracker("ocsort"),
    }


def current_failure_audit(scenario: ResearchScenario) -> dict:
    runner = CurrentTrackerRunner(audit=True)
    outputs: dict[int, list[TrackView]] = {}
    for frame in scenario.replay.frames:
        outputs[frame.frame] = runner.step(frame)
    metrics = evaluate_tracking(scenario, outputs)
    return {
        "scenario": scenario.name,
        "focus": scenario.focus,
        "metrics": metrics,
        "events": runner.audit_events,
    }


def _failure_context(truth: tuple[TruthObject, ...], truth_index: int, frame: ReplayFrame) -> dict:
    target = truth[truth_index]
    tcx, tcy = _center(target.bbox)
    tw, th = _dimensions(target.bbox)
    target_diag = max(1.0, math.hypot(tw, th))
    nearest_ratio = None
    max_overlap = 0.0
    for other_index, other in enumerate(truth):
        if other_index == truth_index:
            continue
        ocx, ocy = _center(other.bbox)
        ratio = math.hypot(ocx - tcx, ocy - tcy) / target_diag
        nearest_ratio = ratio if nearest_ratio is None else min(nearest_ratio, ratio)
        max_overlap = max(max_overlap, bbox_iou(target.bbox, other.bbox))
    camera_motion_magnitude = math.hypot(*frame.camera_motion)
    return {
        "nearest_truth_center_ratio": nearest_ratio,
        "max_truth_overlap_iou": max_overlap,
        "close_competition": bool((nearest_ratio is not None and nearest_ratio <= 1.5) or max_overlap > 0.05),
        "camera_motion_magnitude": camera_motion_magnitude,
        "camera_transform_present": frame.camera_transform is not None,
    }


def mine_current_failure_windows(
    scenario: ResearchScenario,
    *,
    window_radius: int = 2,
    ambiguity_margin: float = 0.08,
) -> dict:
    if window_radius < 0:
        raise ValueError("window_radius must be >= 0")
    if ambiguity_margin < 0.0:
        raise ValueError("ambiguity_margin must be >= 0")

    runner = CurrentTrackerRunner(audit=True)
    outputs: dict[int, list[TrackView]] = {}
    for frame in scenario.replay.frames:
        outputs[frame.frame] = runner.step(frame)

    failures: list[dict] = []
    last_track: dict[str, int] = {}
    gap_start: dict[str, int] = {}
    false_track_first_frame: dict[int, int] = {}

    for frame in scenario.replay.frames:
        truth = scenario.truth_by_frame.get(frame.frame, ())
        tracks = outputs.get(frame.frame, [])
        matches = _match_truth(truth, tracks)
        used_track_ids = {tracks[column].track_id for column in matches.values()}
        for track in tracks:
            if track.missed == 0 and track.track_id not in used_track_ids:
                false_track_first_frame.setdefault(track.track_id, frame.frame)

        for truth_index, truth_item in enumerate(truth):
            track_column = matches.get(truth_index)
            if track_column is None:
                if truth_item.object_id in last_track:
                    gap_start.setdefault(truth_item.object_id, frame.frame)
                continue

            track_id = tracks[track_column].track_id
            previous_track_id = last_track.get(truth_item.object_id)
            gap_frame = gap_start.pop(truth_item.object_id, None)
            if previous_track_id is not None and previous_track_id != track_id:
                context = _failure_context(truth, truth_index, frame)
                failures.append(
                    {
                        "category": "id_switch",
                        "frame": frame.frame,
                        "window": [
                            max(scenario.replay.frames[0].frame, frame.frame - window_radius),
                            min(scenario.replay.frames[-1].frame, frame.frame + window_radius),
                        ],
                        "object_id": truth_item.object_id,
                        "previous_track_id": previous_track_id,
                        "track_id": track_id,
                        "gap_frames_before_switch": 0 if gap_frame is None else max(0, frame.frame - gap_frame),
                        **context,
                    }
                )
            if gap_frame is not None:
                failures.append(
                    {
                        "category": "fragmentation_recovery",
                        "frame": frame.frame,
                        "window": [
                            max(scenario.replay.frames[0].frame, gap_frame - window_radius),
                            min(scenario.replay.frames[-1].frame, frame.frame + window_radius),
                        ],
                        "object_id": truth_item.object_id,
                        "previous_track_id": previous_track_id,
                        "track_id": track_id,
                        "gap_frames": max(0, frame.frame - gap_frame),
                        "same_id_recovery": previous_track_id == track_id,
                        **_failure_context(truth, truth_index, frame),
                    }
                )
            last_track[truth_item.object_id] = track_id

    ambiguous_associations: list[dict] = []
    reactivations: list[dict] = []
    for event in runner.audit_events:
        if event.get("event") == "reactivation" and event.get("matches"):
            reactivations.append(
                {
                    "frame": event["frame"],
                    "matches": event["matches"],
                    "candidate_count": len(event.get("candidates", [])),
                }
            )
            continue
        if event.get("event") != "association":
            continue
        by_detection: dict[int, list[dict]] = {}
        for candidate in event.get("candidates", []):
            if not candidate.get("accepted") or candidate.get("score") is None:
                continue
            by_detection.setdefault(int(candidate["detection_index"]), []).append(candidate)
        for detection_index, candidates in by_detection.items():
            ranked = sorted(candidates, key=lambda item: float(item["score"]), reverse=True)
            if len(ranked) < 2:
                continue
            margin = float(ranked[0]["score"]) - float(ranked[1]["score"])
            if margin <= ambiguity_margin:
                ambiguous_associations.append(
                    {
                        "frame": event["frame"],
                        "stage": event.get("stage"),
                        "detection_index": detection_index,
                        "score_margin": margin,
                        "top_candidates": ranked[:2],
                    }
                )

    metrics = evaluate_tracking(scenario, outputs)
    return {
        "scenario": scenario.name,
        "focus": scenario.focus,
        "metrics": metrics,
        "summary": {
            "id_switch_events": sum(item["category"] == "id_switch" for item in failures),
            "fragmentation_recovery_events": sum(item["category"] == "fragmentation_recovery" for item in failures),
            "close_competition_id_switches": sum(
                item["category"] == "id_switch" and item["close_competition"] for item in failures
            ),
            "ambiguous_association_events": len(ambiguous_associations),
            "reactivation_matches": len(reactivations),
            "false_track_ids": len(false_track_first_frame),
        },
        "failure_windows": failures,
        "ambiguous_associations": ambiguous_associations,
        "reactivations": reactivations,
        "false_track_first_frame": false_track_first_frame,
    }


class RawBBoxSmoother:
    name = "raw"

    def update(self, bbox: BBox) -> BBox:
        return bbox


class BoundedEMABBoxSmoother:
    name = "bounded_ema"

    def __init__(self, alpha: float = 0.52, lag_fraction: float = 0.35) -> None:
        self.alpha = alpha
        self.lag_fraction = lag_fraction
        self.state: BBox | None = None

    def update(self, bbox: BBox) -> BBox:
        if self.state is None:
            self.state = bbox
            return bbox
        scx, scy = _center(self.state)
        rcx, rcy = _center(bbox)
        rw, rh = _dimensions(bbox)
        lag_limit = max(8.0, math.hypot(rw, rh) * self.lag_fraction)
        lag = math.hypot(rcx - scx, rcy - scy)
        alpha = max(self.alpha, 0.85) if lag > lag_limit else self.alpha
        self.state = tuple(previous * (1.0 - alpha) + current * alpha for previous, current in zip(self.state, bbox))
        return self.state


class MotionStateBBoxSmoother:
    name = "motion_state"

    def __init__(self, alpha: float = 0.62, beta: float = 0.18) -> None:
        self.alpha = alpha
        self.beta = beta
        self.center: tuple[float, float] | None = None
        self.size: tuple[float, float] | None = None
        self.velocity = (0.0, 0.0)
        self.size_velocity = (0.0, 0.0)

    def update(self, bbox: BBox) -> BBox:
        measurement_center = _center(bbox)
        measurement_size = _dimensions(bbox)
        if self.center is None or self.size is None:
            self.center = measurement_center
            self.size = measurement_size
            return bbox

        predicted_center = (
            self.center[0] + self.velocity[0],
            self.center[1] + self.velocity[1],
        )
        predicted_size = (
            max(1.0, self.size[0] + self.size_velocity[0]),
            max(1.0, self.size[1] + self.size_velocity[1]),
        )
        center_residual = (
            measurement_center[0] - predicted_center[0],
            measurement_center[1] - predicted_center[1],
        )
        size_residual = (
            measurement_size[0] - predicted_size[0],
            measurement_size[1] - predicted_size[1],
        )
        self.center = (
            predicted_center[0] + self.alpha * center_residual[0],
            predicted_center[1] + self.alpha * center_residual[1],
        )
        self.size = (
            max(1.0, predicted_size[0] + self.alpha * size_residual[0]),
            max(1.0, predicted_size[1] + self.alpha * size_residual[1]),
        )
        self.velocity = (
            self.velocity[0] + self.beta * center_residual[0],
            self.velocity[1] + self.beta * center_residual[1],
        )
        self.size_velocity = (
            self.size_velocity[0] + self.beta * size_residual[0],
            self.size_velocity[1] + self.beta * size_residual[1],
        )
        return _box(self.center[0], self.center[1], self.size[0], self.size[1])


def _stability_metric(
    predicted: list[BBox],
    truth: list[BBox],
    motion_change_frame: int,
) -> dict[str, float]:
    center_errors = []
    width_errors = []
    height_errors = []
    area_errors = []
    temporal_ious = []
    for pred, target in zip(predicted, truth):
        pcx, pcy = _center(pred)
        tcx, tcy = _center(target)
        pw, ph = _dimensions(pred)
        tw, th = _dimensions(target)
        center_errors.append((pcx - tcx, pcy - tcy))
        width_errors.append(pw - tw)
        height_errors.append(ph - th)
        area_errors.append((pw * ph - tw * th) / max(tw * th, 1e-6))
    center_jitter_terms = []
    for index in range(1, len(center_errors)):
        dx = center_errors[index][0] - center_errors[index - 1][0]
        dy = center_errors[index][1] - center_errors[index - 1][1]
        center_jitter_terms.append(math.hypot(dx, dy))
        truth_dx = _center(truth[index])[0] - _center(truth[index - 1])[0]
        truth_dy = _center(truth[index])[1] - _center(truth[index - 1])[1]
        compensated_previous = shift_box(predicted[index - 1], truth_dx, truth_dy)
        temporal_ious.append(bbox_iou(compensated_previous, predicted[index]))
    lag_window = range(motion_change_frame, min(len(predicted), motion_change_frame + 5))
    lag = [
        math.hypot(
            _center(predicted[index])[0] - _center(truth[index])[0],
            _center(predicted[index])[1] - _center(truth[index])[1],
        )
        for index in lag_window
    ]
    return {
        "center_jitter_px": statistics.fmean(center_jitter_terms) if center_jitter_terms else 0.0,
        "width_jitter_px": statistics.pstdev(width_errors) if len(width_errors) > 1 else 0.0,
        "height_jitter_px": statistics.pstdev(height_errors) if len(height_errors) > 1 else 0.0,
        "area_jitter_fraction": statistics.pstdev(area_errors) if len(area_errors) > 1 else 0.0,
        "temporal_iou": statistics.fmean(temporal_ious) if temporal_ious else 1.0,
        "response_lag_px": statistics.fmean(lag) if lag else 0.0,
        "mean_iou_to_truth": statistics.fmean(bbox_iou(pred, target) for pred, target in zip(predicted, truth)),
    }


def bbox_stability_probe() -> dict[str, dict[str, float]]:
    truth_boxes: list[BBox] = []
    raw_boxes: list[BBox] = []
    jitter = [(-3, 2, -2, 2), (2, -2, 3, -1), (-1, 1, -3, 1), (3, -1, 2, -2)]
    x = 120.0
    for frame in range(36):
        if frame < 18:
            x += 5.0
        else:
            x -= 11.0
        width = 40.0 + (4.0 if frame % 5 == 0 else 0.0)
        height = 96.0 + (-5.0 if frame % 7 == 0 else 0.0)
        truth = _box(x, 180.0, 40.0, 96.0)
        jx1, jy1, jx2, jy2 = jitter[frame % len(jitter)]
        raw = (
            truth[0] + jx1,
            truth[1] + jy1,
            truth[2] + jx2 + (width - 40.0),
            truth[3] + jy2 + (height - 96.0),
        )
        truth_boxes.append(truth)
        raw_boxes.append(raw)

    smoothers = [RawBBoxSmoother(), BoundedEMABBoxSmoother(), MotionStateBBoxSmoother()]
    results = {}
    for smoother in smoothers:
        predicted = [smoother.update(box) for box in raw_boxes]
        results[smoother.name] = _stability_metric(predicted, truth_boxes, motion_change_frame=18)
    return results


def run_synthetic_bakeoff(performance_repeats: int = 20) -> dict:
    scenarios = synthetic_scenarios()
    fingerprints = {scenario.name: scenario.replay.canonical_sha256() for scenario in scenarios}
    candidates = {}
    for candidate_name, factory in _candidate_factories().items():
        per_scenario = {}
        total_wall = 0.0
        total_cpu = 0.0
        for scenario in scenarios:
            outputs, timing = run_candidate(scenario, factory)
            per_scenario[scenario.name] = evaluate_tracking(scenario, outputs)
            total_wall += timing["wall_seconds"]
            total_cpu += timing["cpu_seconds"]

        performance_wall = 0.0
        performance_cpu = 0.0
        for _ in range(max(1, performance_repeats)):
            for scenario in scenarios:
                _, timing = run_candidate(scenario, factory)
                performance_wall += timing["wall_seconds"]
                performance_cpu += timing["cpu_seconds"]

        candidates[candidate_name] = {
            "aggregate": _aggregate(per_scenario.values()),
            "scenarios": per_scenario,
            "wall_seconds_single_suite": total_wall,
            "cpu_seconds_single_suite": total_cpu,
            "wall_seconds_repeated": performance_wall,
            "cpu_seconds_repeated": performance_cpu,
            "performance_repeats": max(1, performance_repeats),
        }

    audits = []
    for scenario in scenarios:
        audit = current_failure_audit(scenario)
        audits.append(
            {
                "scenario": scenario.name,
                "focus": scenario.focus,
                "metrics": audit["metrics"],
            }
        )

    return {
        "schema": "spectratrack-tracking-research-v1",
        "source_commit": RESEARCH_BASE_SHA,
        "detection_replay_schema": DETECTION_REPLAY_SCHEMA,
        "provider": "CPUExecutionProvider",
        "detector_policy_runs": 0,
        "onnx_inference_calls": 0,
        "a5_frozen_corpus_revision": None,
        "scenario_replay_sha256": fingerprints,
        "candidates": candidates,
        "bbox_stability": bbox_stability_probe(),
        "current_failure_audit": audits,
        "candidate_notes": {
            "current": "Production MultiObjectTracker unchanged; greedy two-stage high/low association plus CMC/appearance/dormant recovery.",
            "current-global-assignment": (
                "Isolated candidate: production MultiObjectTracker gates, scoring, lifecycle, CMC and dormant recovery are unchanged; "
                "only greedy one-to-one selection is replaced by global maximum-score assignment."
            ),
            "current-ambiguity-guard": (
                "Targeted candidate: current greedy association is preserved outside ambiguous score components; "
                "only connected competitions whose top alternatives are within 0.08 score use global assignment."
            ),
            "byte-global-reference-style": (
                "Clean-room mechanism probe: identical high/low stages and strong-only creation, "
                "global assignment, constant-velocity geometry; not the official ByteTrack repository."
            ),
            "botsort-reference-style": (
                "Clean-room mechanism probe: global assignment plus appearance and replay CMC; "
                "not the official BoT-SORT/FastReID stack."
            ),
            "ocsort-style": (
                "Clean-room observation-direction mechanism probe with global assignment; "
                "not the official OC-SORT implementation."
            ),
        },
    }


def run_loaded_replay(
    replay: DetectionReplay,
    candidate: str = "current",
    audit_current: bool = False,
) -> dict:
    factories = _candidate_factories()
    if candidate not in factories:
        raise ValueError(f"unknown tracker candidate: {candidate}")
    if audit_current and candidate != "current":
        raise ValueError("audit_current is available only for the current tracker")
    runner = CurrentTrackerRunner(audit=True) if audit_current else factories[candidate]()
    outputs = {}
    wall_start = time.perf_counter()
    cpu_start = time.process_time()
    for frame in replay.frames:
        outputs[str(frame.frame)] = [
            {
                "track_id": item.track_id,
                "bbox": [float(value) for value in item.bbox],
                "class_id": item.class_id,
                "label": item.label,
                "missed": item.missed,
                "confirmed": item.confirmed,
            }
            for item in runner.step(frame)
        ]
    cpu_seconds = time.process_time() - cpu_start
    wall_seconds = time.perf_counter() - wall_start
    payload = {
        "schema": "spectratrack-tracking-replay-result-v1",
        "source_commit": RESEARCH_BASE_SHA,
        "detection_replay_schema": DETECTION_REPLAY_SCHEMA,
        "replay_source_sha256": replay.source_sha256,
        "replay_canonical_sha256": replay.canonical_sha256(),
        "candidate": candidate,
        "provider": "CPUExecutionProvider",
        "detector_policy_runs": 0,
        "onnx_inference_calls": 0,
        "wall_seconds": wall_seconds,
        "cpu_seconds": cpu_seconds,
        "frames": outputs,
    }
    if audit_current and isinstance(runner, CurrentTrackerRunner):
        payload["audit_events"] = runner.audit_events
    return payload


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SpectraTrack vNext tracking research")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--synthetic", action="store_true", help="Run deterministic synthetic replay bake-off")
    source.add_argument("--replay", help="Run one candidate on a spectratrack-detection-replay-v1 JSONL file")
    parser.add_argument(
        "--candidate",
        choices=(
            "current",
            "current-global-assignment",
            "current-ambiguity-guard",
            "byte-global-reference-style",
            "botsort-reference-style",
            "ocsort-style",
        ),
        default="current",
    )
    parser.add_argument("--audit-current", action="store_true")
    parser.add_argument("--performance-repeats", type=int, default=20)
    parser.add_argument("--output")
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    if args.performance_repeats < 1:
        raise SystemExit("--performance-repeats must be >= 1")
    if args.replay:
        replay = load_detection_replay(args.replay)
        report = run_loaded_replay(replay, candidate=args.candidate, audit_current=args.audit_current)
    else:
        if args.audit_current:
            raise SystemExit("--audit-current applies only to --replay current")
        report = run_synthetic_bakeoff(args.performance_repeats)
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            handle.write(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
