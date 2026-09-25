from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable
import math


PERSON_LABELS = {"person"}


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
            if left.video == right.video and decision != "same":
                continue
            sim = tracklet_similarity(left, right)
            if sim is not None:
                similarities[(i, j)] = sim
            if decision is not None or (sim is not None and sim >= candidate_threshold):
                edges.append({
                    "left": left.key,
                    "right": right.key,
                    "label": left.label,
                    "similarity": round(sim, 6) if sim is not None else None,
                    "strength": (
                        "manual"
                        if decision in {"same", "different"}
                        else ("strong" if sim is not None and sim >= strong_threshold else "review")
                    ),
                    "relation": _relation(left.label),
                    "review_decision": decision,
                })

    # Conservative complete-link clustering. A new tracklet may join an entity
    # only when it strongly matches every existing member and the entity does
    # not already contain another tracklet from the same source video.
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
                if existing.video == item.video and decision != "same":
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
    for members in clusters:
        label = items[members[0]].label
        prefix = _entity_prefix(label)
        counters[prefix] = counters.get(prefix, 0) + 1
        entity_id = f"{prefix}_{counters[prefix]:03d}"
        entities.append({
            "entity_id": entity_id,
            "label": label,
            "semantics": _entity_semantics(label),
            "members": [items[m].to_public_dict() for m in members],
        })

    edges.sort(
        key=lambda e: (
            0 if e.get("review_decision") in {"same", "different"} else 1,
            -(e["similarity"] if e["similarity"] is not None else -1.0),
            e["left"],
            e["right"],
        )
    )
    return {
        "schema_version": 1,
        "descriptor": "spatial-hsv-gray-edge-gallery-v2",
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
        "tracklets": [t.to_public_dict() for t in items],
        "entities": entities,
        "edges": edges,
        "review_decisions_applied": len(review_decisions),
    }
