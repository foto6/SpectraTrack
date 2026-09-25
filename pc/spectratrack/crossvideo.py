from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable
import math


PERSON_LABELS = {"person"}

REID_SIGNAL_WEIGHTS = {
    "appearance": 0.45,
    "color": 0.20,
    "shape": 0.12,
    "size": 0.08,
    "direction": 0.08,
    "temporal": 0.07,
}


@dataclass(slots=True)
class TrackletSummary:
    video: str
    local_track_id: int
    class_id: int
    label: str
    first_frame: int
    last_frame: int
    observations: int
    mean_score: float
    mean_quality: float
    best_frame: int
    fps: float
    descriptor: tuple[float, ...] | None
    preview_path: str | None = None
    gallery: tuple[tuple[float, ...], ...] = ()
    color_descriptor: tuple[float, ...] | None = None
    mean_aspect_ratio: float | None = None
    mean_relative_area: float | None = None
    start_center: tuple[float, float] | None = None
    end_center: tuple[float, float] | None = None
    motion_direction: tuple[float, float] | None = None

    @property
    def key(self) -> str:
        return f"{self.video}#T{self.local_track_id}"

    def to_public_dict(self) -> dict:
        data = asdict(self)
        data["key"] = self.key
        data["start_s"] = round(self.first_frame / self.fps, 3) if self.fps > 0 else None
        data["end_s"] = round(self.last_frame / self.fps, 3) if self.fps > 0 else None
        data.pop("descriptor", None)
        data.pop("gallery", None)
        data.pop("color_descriptor", None)
        return data


def normalize_descriptor(values: Iterable[float]) -> tuple[float, ...] | None:
    items = tuple(float(v) for v in values)
    norm = math.sqrt(sum(v * v for v in items))
    if not items or norm <= 1e-12:
        return None
    return tuple(v / norm for v in items)


def descriptor_similarity(a: tuple[float, ...] | None, b: tuple[float, ...] | None) -> float | None:
    if a is None or b is None or len(a) != len(b) or not a:
        return None
    na = math.sqrt(sum(v * v for v in a))
    nb = math.sqrt(sum(v * v for v in b))
    if na <= 1e-12 or nb <= 1e-12:
        return None
    return max(0.0, min(1.0, sum(x * y for x, y in zip(a, b)) / (na * nb)))


def tracklet_similarity(left: TrackletSummary, right: TrackletSummary) -> float | None:
    """Appearance/gallery similarity retained as a stable public primitive."""
    base = descriptor_similarity(left.descriptor, right.descriptor)
    if not left.gallery or not right.gallery:
        return base

    pair_scores = []
    for a in left.gallery:
        for b in right.gallery:
            score = descriptor_similarity(a, b)
            if score is not None:
                pair_scores.append(score)
    if not pair_scores:
        return base
    pair_scores.sort(reverse=True)
    count = min(3, len(left.gallery), len(right.gallery), len(pair_scores))
    gallery_score = sum(pair_scores[:count]) / count
    if base is None:
        return gallery_score
    return max(0.0, min(1.0, gallery_score * 0.70 + base * 0.30))


def _ratio_similarity(a: float | None, b: float | None) -> float | None:
    if a is None or b is None or a <= 1e-12 or b <= 1e-12:
        return None
    return min(a, b) / max(a, b)


def _direction_similarity(
    a: tuple[float, float] | None,
    b: tuple[float, float] | None,
) -> float | None:
    if a is None or b is None:
        return None
    an = math.hypot(a[0], a[1])
    bn = math.hypot(b[0], b[1])
    if an <= 1e-9 or bn <= 1e-9:
        return None
    cosine = (a[0] * b[0] + a[1] * b[1]) / (an * bn)
    return max(0.0, min(1.0, (cosine + 1.0) * 0.5))


def _same_video_context(
    left: TrackletSummary,
    right: TrackletSummary,
) -> tuple[float | None, float | None, bool]:
    if left.video != right.video:
        return None, None, False

    overlap = not (left.last_frame < right.first_frame or right.last_frame < left.first_frame)
    if overlap:
        return 0.0, None, True

    earlier, later = (left, right) if left.first_frame < right.first_frame else (right, left)
    gap_frames = max(0, later.first_frame - earlier.last_frame)
    fps = earlier.fps if earlier.fps > 0 else later.fps
    temporal = math.exp(-(gap_frames / fps) / 10.0) if fps > 0 else None

    direction_scores = []
    if earlier.end_center is not None and later.start_center is not None:
        continuation = (
            later.start_center[0] - earlier.end_center[0],
            later.start_center[1] - earlier.end_center[1],
        )
        for motion in (earlier.motion_direction, later.motion_direction):
            score = _direction_similarity(motion, continuation)
            if score is not None:
                direction_scores.append(score)
    if not direction_scores:
        direct = _direction_similarity(earlier.motion_direction, later.motion_direction)
        if direct is not None:
            direction_scores.append(direct)

    direction = sum(direction_scores) / len(direction_scores) if direction_scores else None
    return temporal, direction, False


def reid_signal_scores(left: TrackletSummary, right: TrackletSummary) -> dict[str, float | None]:
    temporal, direction, _ = _same_video_context(left, right)
    return {
        "appearance": tracklet_similarity(left, right),
        "color": descriptor_similarity(left.color_descriptor, right.color_descriptor),
        "shape": _ratio_similarity(left.mean_aspect_ratio, right.mean_aspect_ratio),
        "size": _ratio_similarity(left.mean_relative_area, right.mean_relative_area),
        "direction": direction,
        "temporal": temporal,
    }


def same_object_score(
    left: TrackletSummary,
    right: TrackletSummary,
) -> tuple[float | None, dict[str, float | None]]:
    if left.class_id != right.class_id:
        return None, reid_signal_scores(left, right)

    signals = reid_signal_scores(left, right)
    if signals["appearance"] is None:
        return None, signals

    numerator = 0.0
    weight_sum = 0.0
    for name, weight in REID_SIGNAL_WEIGHTS.items():
        value = signals[name]
        if value is None:
            continue
        numerator += weight * value
        weight_sum += weight
    if weight_sum <= 1e-12:
        return None, signals
    return max(0.0, min(1.0, numerator / weight_sum)), signals


def _relation(label: str) -> str:
    return "same_appearance_candidate" if label.lower() in PERSON_LABELS else "same_object_candidate"


def _entity_semantics(label: str) -> str:
    return "appearance_group_not_identity" if label.lower() in PERSON_LABELS else "tentative_same_object_group"


def _entity_prefix(label: str) -> str:
    cleaned = "".join(ch if ch.isalnum() else "_" for ch in label.upper()).strip("_") or "OBJECT"
    if label.lower() in PERSON_LABELS:
        return f"{cleaned}_APPEARANCE"
    return cleaned


def _decision_key(left: str, right: str) -> tuple[str, str]:
    return tuple(sorted((left, right)))


def build_cross_video_graph(
    tracklets: Iterable[TrackletSummary],
    candidate_threshold: float = 0.86,
    strong_threshold: float = 0.94,
    review_decisions: dict[tuple[str, str], str] | None = None,
) -> dict:
    if not 0.0 <= candidate_threshold <= strong_threshold <= 1.0:
        raise ValueError("Require 0 <= candidate_threshold <= strong_threshold <= 1")

    review_decisions = review_decisions or {}
    allowed_decisions = {"same", "different", "unsure"}
    if any(value not in allowed_decisions for value in review_decisions.values()):
        raise ValueError("Review decisions must be same, different, or unsure")

    items = [t for t in tracklets if t.descriptor is not None]
    items.sort(key=lambda t: (t.label.lower(), t.video.lower(), t.local_track_id))

    similarities: dict[tuple[int, int], float] = {}
    edges: list[dict] = []
    for i, left in enumerate(items):
        for j in range(i + 1, len(items)):
            right = items[j]
            if left.class_id != right.class_id:
                continue
            decision = review_decisions.get(_decision_key(left.key, right.key))
            _, _, overlaps = _same_video_context(left, right)
            if overlaps and decision != "same":
                continue

            sim, signals = same_object_score(left, right)
            if sim is not None:
                similarities[(i, j)] = sim
            if decision is not None or (sim is not None and sim >= candidate_threshold):
                rounded_signals = {
                    name: round(value, 6) if value is not None else None
                    for name, value in signals.items()
                }
                edges.append({
                    "left": left.key,
                    "right": right.key,
                    "label": left.label,
                    "similarity": round(sim, 6) if sim is not None else None,
                    "same_object_score": round(sim, 6) if sim is not None else None,
                    "appearance_similarity": rounded_signals["appearance"],
                    "signals": rounded_signals,
                    "strength": (
                        "manual"
                        if decision in {"same", "different"}
                        else ("strong" if sim is not None and sim >= strong_threshold else "review")
                    ),
                    "relation": _relation(left.label),
                    "review_decision": decision,
                })

    # Conservative complete-link clustering. A new tracklet joins an entity only
    # when it strongly matches every existing member. Same-video fragments are
    # eligible only when their frame intervals do not overlap.
    clusters: list[list[int]] = []
    for idx, item in enumerate(items):
        best_cluster: int | None = None
        best_score = -1.0
        for cidx, members in enumerate(clusters):
            if items[members[0]].class_id != item.class_id:
                continue
            scores = []
            compatible = True
            for member in members:
                existing = items[member]
                decision = review_decisions.get(_decision_key(existing.key, item.key))
                _, _, overlaps = _same_video_context(existing, item)
                if overlaps and decision != "same":
                    compatible = False
                    break
                if decision == "different":
                    compatible = False
                    break
                if decision == "same":
                    scores.append(1.0)
                    continue
                pair = (member, idx) if member < idx else (idx, member)
                sim = similarities.get(pair)
                if sim is None or sim < strong_threshold:
                    compatible = False
                    break
                scores.append(sim)
            if compatible:
                score = min(scores) if scores else 0.0
                if score > best_score:
                    best_cluster = cidx
                    best_score = score
        if best_cluster is None:
            clusters.append([idx])
        else:
            clusters[best_cluster].append(idx)

    counters: dict[str, int] = {}
    entities = []
    global_ids: dict[str, str] = {}
    for members in clusters:
        label = items[members[0]].label
        prefix = _entity_prefix(label)
        counters[prefix] = counters.get(prefix, 0) + 1
        entity_id = f"{prefix}_{counters[prefix]:03d}"
        public_members = []
        for member_index in members:
            member = items[member_index].to_public_dict()
            member["global_object_id"] = entity_id
            global_ids[items[member_index].key] = entity_id
            public_members.append(member)
        entities.append({
            "entity_id": entity_id,
            "global_object_id": entity_id,
            "label": label,
            "semantics": _entity_semantics(label),
            "members": public_members,
        })

    edges.sort(
        key=lambda e: (
            0 if e.get("review_decision") in {"same", "different"} else 1,
            -(e["similarity"] if e["similarity"] is not None else -1.0),
            e["left"],
            e["right"],
        )
    )
    public_tracklets = []
    for item in items:
        public = item.to_public_dict()
        public["global_object_id"] = global_ids.get(item.key)
        public_tracklets.append(public)

    return {
        "schema_version": 2,
        "descriptor": "spatial-hsv-gray-edge-gallery-v2",
        "scoring": {
            "name": "multi-signal-reid-v1",
            "probability_calibrated": False,
            "weights": dict(REID_SIGNAL_WEIGHTS),
        },
        "thresholds": {
            "candidate": float(candidate_threshold),
            "strong": float(strong_threshold),
        },
        "safety_semantics": {
            "person": (
                "Person links mean similar visible appearance in this batch only; "
                "they are not biometric identity or face recognition."
            ),
            "other_classes": "Object links are visual candidates and can be wrong; review ambiguous edges.",
        },
        "tracklets": public_tracklets,
        "entities": entities,
        "edges": edges,
        "review_decisions_applied": len(review_decisions),
    }
