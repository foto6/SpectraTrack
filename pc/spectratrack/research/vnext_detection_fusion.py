from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from math import hypot
from pathlib import Path
import time
from typing import Any, Iterable, Mapping

from ..detector import YoloOnnxDetector, _tile_regions
from ..integrity import sha256_file
from ..qa_benchmark import GroundTruthFrame, GroundTruthObject, PredictedObject, evaluate_frames


PREFUSION_SCHEMA = "spectratrack-detection-prefusion-v1"
REPLAY_SCHEMA = "spectratrack-detection-replay-v1"
BBox = tuple[float, float, float, float]


@dataclass(frozen=True, slots=True)
class FusionCandidate:
    bbox: BBox
    score: float
    class_id: int
    label: str
    source_kind: str
    source_id: str
    source_region: tuple[int, int, int, int] | None = None


@dataclass(frozen=True, slots=True)
class FusionDetection:
    bbox: BBox
    score: float
    class_id: int
    label: str
    members: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class FusionConfig:
    iou_threshold: float = 0.55
    center_ratio: float = 0.20
    size_ratio: float = 1.80
    score_power: float = 1.0
    full_weight: float = 1.0
    tile_weight: float = 1.0
    evidence_weak_score: float = 0.12
    evidence_solo_score: float = 0.20
    evidence_strong_score: float = 0.35
    evidence_min_sources: int = 2

    def validate(self) -> None:
        if not 0.0 < self.iou_threshold <= 1.0:
            raise ValueError("iou_threshold must be in (0, 1]")
        if self.center_ratio <= 0.0:
            raise ValueError("center_ratio must be > 0")
        if self.size_ratio < 1.0:
            raise ValueError("size_ratio must be >= 1")
        if self.score_power <= 0.0:
            raise ValueError("score_power must be > 0")
        if self.full_weight <= 0.0 or self.tile_weight <= 0.0:
            raise ValueError("evidence weights must be > 0")
        if not 0.0 <= self.evidence_weak_score <= self.evidence_solo_score <= self.evidence_strong_score <= 1.0:
            raise ValueError("evidence score thresholds must satisfy weak <= solo <= strong in [0, 1]")
        if self.evidence_min_sources < 2:
            raise ValueError("evidence_min_sources must be >= 2")


@dataclass(frozen=True, slots=True)
class ResearchFrame:
    video: str
    frame: int
    timestamp_s: float
    width: int
    height: int
    candidates: tuple[FusionCandidate, ...]
    ground_truth: tuple[GroundTruthObject, ...] = ()
    tags: tuple[str, ...] = ()


def bbox_iou(a: BBox, b: BBox) -> float:
    xx1 = max(a[0], b[0])
    yy1 = max(a[1], b[1])
    xx2 = min(a[2], b[2])
    yy2 = min(a[3], b[3])
    inter = max(0.0, xx2 - xx1) * max(0.0, yy2 - yy1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return inter / max(area_a + area_b - inter, 1e-9)


def _center(box: BBox) -> tuple[float, float]:
    return ((box[0] + box[2]) * 0.5, (box[1] + box[3]) * 0.5)


def _size(box: BBox) -> tuple[float, float]:
    return (box[2] - box[0], box[3] - box[1])


def _score_order(items: list[FusionCandidate]) -> list[int]:
    return sorted(range(len(items)), key=lambda index: items[index].score, reverse=True)


def _conservative_match(seed: FusionCandidate, candidate: FusionCandidate, config: FusionConfig) -> bool:
    if seed.class_id != candidate.class_id:
        return False
    if bbox_iou(seed.bbox, candidate.bbox) < config.iou_threshold:
        return False
    seed_w, seed_h = _size(seed.bbox)
    cand_w, cand_h = _size(candidate.bbox)
    if min(seed_w, seed_h, cand_w, cand_h) <= 0.0:
        return False
    if max(seed_w, cand_w) / min(seed_w, cand_w) > config.size_ratio:
        return False
    if max(seed_h, cand_h) / min(seed_h, cand_h) > config.size_ratio:
        return False
    sx, sy = _center(seed.bbox)
    cx, cy = _center(candidate.bbox)
    distance = hypot(sx - cx, sy - cy)
    normalization = max(1.0, min(seed_w, seed_h, cand_w, cand_h))
    return distance <= config.center_ratio * normalization


def hard_nms_fusion(candidates: Iterable[FusionCandidate], config: FusionConfig) -> list[FusionDetection]:
    """Replicate current final cross-pass hard class-aware NMS."""
    config.validate()
    items = list(candidates)
    remaining = set(range(len(items)))
    output: list[FusionDetection] = []
    for seed_index in _score_order(items):
        if seed_index not in remaining:
            continue
        remaining.remove(seed_index)
        seed = items[seed_index]
        members = [seed_index]
        for index in list(remaining):
            item = items[index]
            if item.class_id == seed.class_id and bbox_iou(seed.bbox, item.bbox) >= config.iou_threshold:
                members.append(index)
                remaining.remove(index)
        output.append(FusionDetection(seed.bbox, seed.score, seed.class_id, seed.label, tuple(members)))
    return output


def _greedy_groups(items: list[FusionCandidate], config: FusionConfig) -> list[tuple[int, ...]]:
    """Direct seed matching only; deliberately no transitive NMM chaining."""
    remaining = set(range(len(items)))
    groups: list[tuple[int, ...]] = []
    for seed_index in _score_order(items):
        if seed_index not in remaining:
            continue
        remaining.remove(seed_index)
        seed = items[seed_index]
        members = [seed_index]
        for index in list(remaining):
            if _conservative_match(seed, items[index], config):
                members.append(index)
                remaining.remove(index)
        groups.append(tuple(members))
    return groups


def conservative_nmm_fusion(
    candidates: Iterable[FusionCandidate],
    config: FusionConfig,
) -> list[FusionDetection]:
    """GreedyNMM-style grouping with conservative gates and envelope geometry."""
    config.validate()
    items = list(candidates)
    output: list[FusionDetection] = []
    for members in _greedy_groups(items, config):
        seed_index = max(members, key=lambda index: items[index].score)
        seed = items[seed_index]
        boxes = [items[index].bbox for index in members]
        output.append(
            FusionDetection(
                (
                    min(box[0] for box in boxes),
                    min(box[1] for box in boxes),
                    max(box[2] for box in boxes),
                    max(box[3] for box in boxes),
                ),
                max(items[index].score for index in members),
                seed.class_id,
                seed.label,
                members,
            )
        )
    return output


def weighted_coordinate_fusion(
    candidates: Iterable[FusionCandidate],
    config: FusionConfig,
) -> list[FusionDetection]:
    """Use the conservative groups, but average coordinates by confidence/evidence."""
    config.validate()
    items = list(candidates)
    output: list[FusionDetection] = []
    for members in _greedy_groups(items, config):
        weights = []
        for index in members:
            item = items[index]
            evidence = config.full_weight if item.source_kind == "full" else config.tile_weight
            weights.append(max(item.score, 1e-6) ** config.score_power * evidence)
        total = sum(weights)
        coords = tuple(
            sum(items[index].bbox[axis] * weight for index, weight in zip(members, weights)) / total
            for axis in range(4)
        )
        seed_index = max(members, key=lambda index: items[index].score)
        seed = items[seed_index]
        output.append(
            FusionDetection(
                (float(coords[0]), float(coords[1]), float(coords[2]), float(coords[3])),
                max(items[index].score for index in members),
                seed.class_id,
                seed.label,
                members,
            )
        )
    return output


def evidence_aware_fusion(
    candidates: Iterable[FusionCandidate],
    config: FusionConfig,
) -> list[FusionDetection]:
    """Weighted geometry plus stricter acceptance for weak single-source evidence."""
    config.validate()
    items = list(candidates)
    output: list[FusionDetection] = []
    for members in _greedy_groups(items, config):
        group = [items[index] for index in members]
        max_score = max(item.score for item in group)
        independent_sources = {item.source_id for item in group}
        corroborated = len(independent_sources) >= config.evidence_min_sources
        has_full_frame_evidence = any(item.source_kind == "full" for item in group)
        accepted = (
            max_score >= config.evidence_strong_score
            or (has_full_frame_evidence and max_score >= config.evidence_solo_score)
            or (max_score >= config.evidence_weak_score and corroborated)
        )
        if not accepted:
            continue

        weights = []
        for item in group:
            evidence = config.full_weight if item.source_kind == "full" else config.tile_weight
            weights.append(max(item.score, 1e-6) ** config.score_power * evidence)
        total = sum(weights)
        coords = tuple(
            sum(item.bbox[axis] * weight for item, weight in zip(group, weights)) / total for axis in range(4)
        )
        seed_index = max(members, key=lambda index: items[index].score)
        seed = items[seed_index]
        output.append(
            FusionDetection(
                (float(coords[0]), float(coords[1]), float(coords[2]), float(coords[3])),
                max_score,
                seed.class_id,
                seed.label,
                members,
            )
        )
    return output


def fuse_candidates(
    candidates: Iterable[FusionCandidate],
    method: str,
    config: FusionConfig,
) -> list[FusionDetection]:
    if method == "hard-nms":
        return hard_nms_fusion(candidates, config)
    if method == "conservative-nmm":
        return conservative_nmm_fusion(candidates, config)
    if method == "weighted":
        return weighted_coordinate_fusion(candidates, config)
    if method == "evidence-aware":
        return evidence_aware_fusion(candidates, config)
    raise ValueError(f"unknown fusion method: {method}")


def collect_prefusion_people_recall(
    detector: YoloOnnxDetector,
    frame_bgr,
    *,
    person_threshold: float,
    tile_size: int,
    tile_overlap: float,
) -> list[FusionCandidate]:
    """Capture candidates after decoder-local NMS, before final cross-pass fusion."""
    if not 0.0 <= person_threshold <= 1.0:
        raise ValueError("person_threshold must be in [0, 1]")
    if tile_size <= 0:
        raise ValueError("tile_size must be > 0")
    if not 0.0 <= tile_overlap < 1.0:
        raise ValueError("tile_overlap must satisfy 0 <= overlap < 1")

    detector._reset_policy_metrics()
    thresholds = dict(detector.class_thresholds)
    thresholds["person"] = float(person_threshold)
    height, width = frame_bgr.shape[:2]
    candidates: list[FusionCandidate] = []

    for detection in detector._detect_once(frame_bgr, thresholds):
        candidates.append(
            FusionCandidate(
                tuple(float(v) for v in detection.bbox),
                float(detection.score),
                int(detection.class_id),
                detection.label,
                "full",
                "full",
                (0, 0, width, height),
            )
        )

    person_ids = {index for index, label in enumerate(detector.labels) if label.lower() == "person"}
    for region_index, (x1, y1, x2, y2) in enumerate(_tile_regions(width, height, tile_size, tile_overlap)):
        tile = frame_bgr[y1:y2, x1:x2]
        for detection in detector._detect_once(tile, thresholds):
            if detection.class_id not in person_ids:
                continue
            bx1, by1, bx2, by2 = detection.bbox
            candidates.append(
                FusionCandidate(
                    (bx1 + x1, by1 + y1, bx2 + x1, by2 + y1),
                    float(detection.score),
                    int(detection.class_id),
                    detection.label,
                    "tile",
                    f"tile:{region_index}",
                    (x1, y1, x2, y2),
                )
            )
    return candidates


def _candidate_json(item: FusionCandidate) -> dict[str, Any]:
    return {
        "bbox": list(item.bbox),
        "score": item.score,
        "class_id": item.class_id,
        "label": item.label,
        "source_kind": item.source_kind,
        "source_id": item.source_id,
        "source_region": list(item.source_region) if item.source_region else None,
    }


def _candidate_from_json(data: Mapping[str, Any]) -> FusionCandidate:
    bbox = tuple(float(value) for value in data["bbox"])
    if len(bbox) != 4:
        raise ValueError("candidate bbox must have four coordinates")
    region_data = data.get("source_region")
    region = tuple(int(value) for value in region_data) if region_data is not None else None
    if region is not None and len(region) != 4:
        raise ValueError("source_region must have four coordinates")
    return FusionCandidate(
        (bbox[0], bbox[1], bbox[2], bbox[3]),
        float(data["score"]),
        int(data["class_id"]),
        str(data["label"]),
        str(data["source_kind"]),
        str(data["source_id"]),
        None if region is None else (region[0], region[1], region[2], region[3]),
    )


def write_prefusion_dump(
    path: str | Path,
    metadata: Mapping[str, Any],
    frames: Iterable[ResearchFrame],
    summary: Mapping[str, Any],
) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        handle.write(
            json.dumps({"type": "metadata", "schema": PREFUSION_SCHEMA, **dict(metadata)}, sort_keys=True) + "\n"
        )
        for frame in frames:
            handle.write(
                json.dumps(
                    {
                        "type": "frame",
                        "video": frame.video,
                        "frame": frame.frame,
                        "timestamp_s": frame.timestamp_s,
                        "width": frame.width,
                        "height": frame.height,
                        "candidates": [_candidate_json(item) for item in frame.candidates],
                    },
                    sort_keys=True,
                )
                + "\n"
            )
        handle.write(json.dumps({"type": "summary", **dict(summary)}, sort_keys=True) + "\n")


def read_prefusion_dump(path: str | Path) -> tuple[dict[str, Any], list[ResearchFrame], dict[str, Any]]:
    metadata: dict[str, Any] | None = None
    summary: dict[str, Any] = {}
    frames: list[ResearchFrame] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            if not raw.strip():
                continue
            data = json.loads(raw)
            kind = data.get("type")
            if kind == "metadata":
                if metadata is not None:
                    raise ValueError("prefusion dump has multiple metadata records")
                if data.get("schema") != PREFUSION_SCHEMA:
                    raise ValueError(f"unsupported prefusion schema: {data.get('schema')!r}")
                metadata = dict(data)
            elif kind == "frame":
                frames.append(
                    ResearchFrame(
                        str(data["video"]),
                        int(data["frame"]),
                        float(data.get("timestamp_s", 0.0)),
                        int(data["width"]),
                        int(data["height"]),
                        tuple(_candidate_from_json(item) for item in data.get("candidates", [])),
                    )
                )
            elif kind == "summary":
                summary = dict(data)
            else:
                raise ValueError(f"{path}:{line_number}: unknown record type {kind!r}")
    if metadata is None:
        raise ValueError("prefusion dump is missing metadata")
    return metadata, frames, summary


def write_detection_replay(
    path: str | Path,
    metadata: Mapping[str, Any],
    frames: Iterable[tuple[ResearchFrame, list[FusionDetection]]],
) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    first = {
        "type": "metadata",
        "schema": REPLAY_SCHEMA,
        "source_commit": metadata["source_commit"],
        "video": metadata["video"],
        "video_sha256": metadata.get("video_sha256"),
        "detector": metadata["detector"],
        "model_sha256": metadata.get("model_sha256"),
        "provider": metadata["provider"],
        "config": metadata.get("config", {}),
        "width": int(metadata["width"]),
        "height": int(metadata["height"]),
    }
    with target.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(first, sort_keys=True) + "\n")
        for frame, detections in frames:
            handle.write(
                json.dumps(
                    {
                        "type": "frame",
                        "video": frame.video,
                        "frame": frame.frame,
                        "timestamp_s": frame.timestamp_s,
                        "width": frame.width,
                        "height": frame.height,
                        "detections": [
                            {
                                "bbox": list(item.bbox),
                                "score": item.score,
                                "class_id": item.class_id,
                                "label": item.label,
                                "appearance": None,
                            }
                            for item in detections
                        ],
                    },
                    sort_keys=True,
                )
                + "\n"
            )


def convert_prefusion_to_replay(
    prefusion_path: str | Path,
    replay_path: str | Path,
    *,
    method: str,
    config: FusionConfig,
) -> None:
    metadata, frames, summary = read_prefusion_dump(prefusion_path)
    replay_config = dict(metadata.get("config", {}))
    replay_config["cross_pass_fusion"] = {
        "method": method,
        "iou_threshold": config.iou_threshold,
        "center_ratio": config.center_ratio,
        "size_ratio": config.size_ratio,
        "score_power": config.score_power,
        "full_weight": config.full_weight,
        "tile_weight": config.tile_weight,
        "evidence_weak_score": config.evidence_weak_score,
        "evidence_solo_score": config.evidence_solo_score,
        "evidence_strong_score": config.evidence_strong_score,
        "evidence_min_sources": config.evidence_min_sources,
        "prefusion_summary": {
            "policy_runs": summary.get("policy_runs"),
            "inference_calls": summary.get("inference_calls"),
            "wall_time_s": summary.get("wall_time_s"),
        },
    }
    metadata = {**metadata, "config": replay_config}
    write_detection_replay(
        replay_path,
        metadata,
        [(frame, fuse_candidates(frame.candidates, method, config)) for frame in frames],
    )


def _match_gt(
    ground_truth: tuple[GroundTruthObject, ...],
    detections: list[FusionDetection],
    threshold: float,
) -> dict[int, int]:
    possible = []
    for gt_index, obj in enumerate(ground_truth):
        if obj.ignore:
            continue
        for pred_index, prediction in enumerate(detections):
            if obj.label != prediction.label:
                continue
            overlap = bbox_iou(obj.bbox, prediction.bbox)
            if overlap >= threshold:
                possible.append((overlap, gt_index, pred_index))
    possible.sort(reverse=True)
    matched_gt: set[int] = set()
    matched_pred: set[int] = set()
    result: dict[int, int] = {}
    for _, gt_index, pred_index in possible:
        if gt_index in matched_gt or pred_index in matched_pred:
            continue
        matched_gt.add(gt_index)
        matched_pred.add(pred_index)
        result[gt_index] = pred_index
    return result


def _candidate_assignment(
    ground_truth: tuple[GroundTruthObject, ...],
    candidates: tuple[FusionCandidate, ...],
    threshold: float,
) -> dict[int, int]:
    result: dict[int, int] = {}
    for index, candidate in enumerate(candidates):
        best_index = None
        best_overlap = threshold
        for gt_index, obj in enumerate(ground_truth):
            if obj.ignore or obj.label != candidate.label:
                continue
            overlap = bbox_iou(obj.bbox, candidate.bbox)
            if overlap >= best_overlap:
                best_overlap = overlap
                best_index = gt_index
        if best_index is not None:
            result[index] = best_index
    return result


def _translated_iou(previous: BBox, previous_gt: BBox, current: BBox, current_gt: BBox) -> float:
    prev_x, prev_y = _center(previous_gt)
    curr_x, curr_y = _center(current_gt)
    dx, dy = curr_x - prev_x, curr_y - prev_y
    shifted = (previous[0] + dx, previous[1] + dy, previous[2] + dx, previous[3] + dy)
    return bbox_iou(shifted, current)


def _jitter_metrics(
    frames: list[ResearchFrame],
    fused_by_key: Mapping[tuple[str, int], list[FusionDetection]],
    match_iou: float,
) -> dict[str, float | int | None]:
    objects: dict[tuple[str, str], list[tuple[int, BBox, BBox]]] = {}
    for frame in frames:
        fused = fused_by_key[(frame.video, frame.frame)]
        matches = _match_gt(frame.ground_truth, fused, match_iou)
        for gt_index, pred_index in matches.items():
            obj = frame.ground_truth[gt_index]
            if obj.object_id is None:
                continue
            objects.setdefault((frame.video, obj.object_id), []).append((frame.frame, obj.bbox, fused[pred_index].bbox))

    center: list[float] = []
    width: list[float] = []
    height: list[float] = []
    area_ratio: list[float] = []
    temporal: list[float] = []
    for samples in objects.values():
        samples.sort(key=lambda item: item[0])
        for previous, current in zip(samples, samples[1:]):
            if current[0] != previous[0] + 1:
                continue
            prev_gt, prev_box = previous[1], previous[2]
            curr_gt, curr_box = current[1], current[2]
            pgx, pgy = _center(prev_gt)
            pcx, pcy = _center(prev_box)
            cgx, cgy = _center(curr_gt)
            ccx, ccy = _center(curr_box)
            center.append(hypot((ccx - cgx) - (pcx - pgx), (ccy - cgy) - (pcy - pgy)))
            pw, ph = _size(prev_box)
            cw, ch = _size(curr_box)
            width.append(abs(cw - pw))
            height.append(abs(ch - ph))
            prev_area = max(1e-9, pw * ph)
            curr_area = max(0.0, cw * ch)
            area_ratio.append(abs(curr_area - prev_area) / prev_area)
            temporal.append(_translated_iou(prev_box, prev_gt, curr_box, curr_gt))

    def mean(values: list[float]) -> float | None:
        return sum(values) / len(values) if values else None

    return {
        "jitter_pair_count": len(center),
        "center_jitter_px": mean(center),
        "width_jitter_px": mean(width),
        "height_jitter_px": mean(height),
        "area_jitter_ratio": mean(area_ratio),
        "temporal_iou": mean(temporal),
    }


def evaluate_fusion(
    frames: Iterable[ResearchFrame],
    *,
    method: str,
    config: FusionConfig,
    match_iou: float = 0.5,
) -> dict[str, Any]:
    items = list(frames)
    predictions: dict[tuple[str, int], list[PredictedObject]] = {}
    qa_frames: list[GroundTruthFrame] = []
    fused_by_key: dict[tuple[str, int], list[FusionDetection]] = {}
    duplicate_count = 0
    fusion_mistakes = 0
    localization_ious: list[float] = []

    for frame in items:
        fused = fuse_candidates(frame.candidates, method, config)
        key = (frame.video, frame.frame)
        fused_by_key[key] = fused
        predictions[key] = [PredictedObject(item.bbox, item.label, item.score) for item in fused]
        qa_frames.append(GroundTruthFrame(frame.video, frame.frame, frame.tags, frame.ground_truth))

        assignment = _candidate_assignment(frame.ground_truth, frame.candidates, match_iou)
        counts: dict[int, int] = {}
        for gt_index in assignment.values():
            counts[gt_index] = counts.get(gt_index, 0) + 1
        duplicate_count += sum(max(0, count - 1) for count in counts.values())

        for detection in fused:
            member_gt = {assignment[index] for index in detection.members if index in assignment}
            if len(member_gt) > 1:
                fusion_mistakes += 1

        for gt_index, pred_index in _match_gt(frame.ground_truth, fused, match_iou).items():
            localization_ious.append(bbox_iou(frame.ground_truth[gt_index].bbox, fused[pred_index].bbox))

    metrics = evaluate_frames(qa_frames, predictions, label="person", iou_threshold=match_iou)
    precision = float(metrics["precision"])
    recall = float(metrics["recall"])
    f1 = (2.0 * precision * recall / (precision + recall)) if precision + recall > 0.0 else 0.0
    return {
        "method": method,
        "config": {
            "iou_threshold": config.iou_threshold,
            "center_ratio": config.center_ratio,
            "size_ratio": config.size_ratio,
            "score_power": config.score_power,
            "full_weight": config.full_weight,
            "tile_weight": config.tile_weight,
            "match_iou": match_iou,
        },
        "tp": metrics["tp"],
        "fp": metrics["false_positives"],
        "fn": metrics["false_negatives"],
        "recall": recall,
        "precision": precision,
        "f1": f1,
        "bbox_localization_iou": (sum(localization_ious) / len(localization_ious) if localization_ious else None),
        "duplicate_count_before_fusion": duplicate_count,
        "fusion_mistakes": fusion_mistakes,
        **_jitter_metrics(items, fused_by_key, match_iou),
    }


def _candidate(
    bbox: BBox,
    score: float,
    source_kind: str,
    source_id: str,
) -> FusionCandidate:
    return FusionCandidate(bbox, score, 0, "person", source_kind, source_id)


def _gt(object_id: str, bbox: BBox, attributes: tuple[str, ...] = ()) -> GroundTruthObject:
    return GroundTruthObject(object_id, "person", bbox, False, attributes)


def synthetic_frames() -> list[ResearchFrame]:
    frames: list[ResearchFrame] = []

    # Alternating score winner: same evidence geometry, different winner every frame.
    gt_box = (100.0, 100.0, 160.0, 220.0)
    full_box = (96.0, 96.0, 166.0, 224.0)
    tile_box = (101.0, 101.0, 161.0, 221.0)
    for frame_index in range(8):
        if frame_index % 2:
            full_score, tile_score = 0.76, 0.86
        else:
            full_score, tile_score = 0.88, 0.74
        frames.append(
            ResearchFrame(
                "synthetic/winner_flip",
                frame_index,
                frame_index / 10.0,
                640,
                360,
                (
                    _candidate(full_box, full_score, "full", "full"),
                    _candidate(tile_box, tile_score, "tile", "tile:0"),
                ),
                (_gt("p1", gt_box),),
                ("winner_flip",),
            )
        )

    frames.extend(
        [
            ResearchFrame(
                "synthetic/tile_boundary",
                0,
                0.0,
                800,
                400,
                (
                    _candidate((390, 80, 430, 200), 0.82, "full", "full"),
                    _candidate((392, 81, 431, 201), 0.78, "tile", "tile:0"),
                    _candidate((389, 79, 429, 199), 0.72, "tile", "tile:1"),
                ),
                (_gt("p1", (390, 80, 430, 200)),),
                ("tile_boundary",),
            ),
            ResearchFrame(
                "synthetic/close_people",
                0,
                0.0,
                640,
                360,
                (
                    _candidate((100, 80, 140, 210), 0.88, "full", "full"),
                    _candidate((102, 82, 141, 209), 0.75, "tile", "tile:0"),
                    _candidate((150, 82, 190, 212), 0.86, "full", "full"),
                    _candidate((149, 80, 189, 210), 0.73, "tile", "tile:0"),
                ),
                (_gt("p1", (100, 80, 140, 210)), _gt("p2", (150, 82, 190, 212))),
                ("close_people",),
            ),
            ResearchFrame(
                "synthetic/partial_full",
                0,
                0.0,
                640,
                360,
                (
                    _candidate((200, 70, 250, 250), 0.67, "full", "full"),
                    _candidate((204, 120, 247, 249), 0.83, "tile", "tile:0"),
                ),
                (_gt("p1", (200, 70, 250, 250), ("partial_vs_full",)),),
                ("partial_vs_full",),
            ),
            ResearchFrame(
                "synthetic/tiny",
                0,
                0.0,
                640,
                360,
                (
                    _candidate((300, 100, 311, 122), 0.41, "full", "full"),
                    _candidate((301, 100, 312, 123), 0.58, "tile", "tile:0"),
                ),
                (_gt("p1", (301, 100, 312, 123), ("tiny",)),),
                ("tiny",),
            ),
            ResearchFrame(
                "synthetic/edge",
                0,
                0.0,
                640,
                360,
                (
                    _candidate((0, 120, 28, 260), 0.76, "full", "full"),
                    _candidate((0, 118, 30, 259), 0.80, "tile", "tile:0"),
                    _candidate((500, 20, 530, 90), 0.30, "tile", "tile:1"),
                ),
                (_gt("p1", (0, 120, 29, 260), ("edge",)),),
                ("edge",),
            ),
            ResearchFrame(
                "synthetic/crowded_overlap",
                0,
                0.0,
                640,
                360,
                (
                    _candidate((100, 100, 160, 220), 0.92, "full", "full"),
                    _candidate((115, 100, 175, 220), 0.89, "tile", "tile:0"),
                ),
                (
                    _gt("p1", (100, 100, 160, 220), ("crowded",)),
                    _gt("p2", (115, 100, 175, 220), ("crowded",)),
                ),
                ("crowded", "high_overlap_distinct_people"),
            ),
        ]
    )
    return frames


def synthetic_report(
    config: FusionConfig | None = None,
    *,
    source_commit: str | None = None,
) -> dict[str, Any]:
    config = config or FusionConfig()
    frames = synthetic_frames()
    started = time.perf_counter()
    methods = {
        method: evaluate_fusion(frames, method=method, config=config)
        for method in ("hard-nms", "conservative-nmm", "weighted")
    }
    return {
        "schema": "spectratrack-vnext-fusion-synthetic-v1",
        "experiment_id": "a1-fusion-synthetic-001",
        "source_commit": source_commit,
        "provider": "none-synthetic",
        "detector": "synthetic-prefusion-fixture",
        "model_sha256": None,
        "config": {
            "fusion_iou": config.iou_threshold,
            "center_ratio": config.center_ratio,
            "size_ratio": config.size_ratio,
            "score_power": config.score_power,
            "full_weight": config.full_weight,
            "tile_weight": config.tile_weight,
            "match_iou": 0.5,
        },
        "policy_runs": 0,
        "inference_calls": 0,
        "wall_time_s": time.perf_counter() - started,
        "note": "Synthetic stress data only; not CCTV quality evidence.",
        "frame_count": len(frames),
        "ground_truth_count": sum(len(frame.ground_truth) for frame in frames),
        "methods": methods,
    }


def _collect_video(args: argparse.Namespace) -> int:
    import cv2

    model_path = Path(args.model)
    video_path = Path(args.video)
    if not model_path.is_file():
        raise SystemExit(f"model not found: {model_path}")
    if not video_path.is_file():
        raise SystemExit(f"video not found: {video_path}")

    detector = YoloOnnxDetector(
        model_path,
        input_size=args.input_size,
        conf_threshold=args.conf,
        iou_threshold=args.decoder_iou,
        prefer_gpu=not args.cpu,
    )
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise SystemExit(f"cannot open video: {video_path}")

    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    frames: list[ResearchFrame] = []
    inference_calls = 0
    stage_ms: dict[str, float] = {}
    started = time.perf_counter()
    frame_index = 0

    try:
        while args.max_frames <= 0 or frame_index < args.max_frames:
            ok, frame_bgr = capture.read()
            if not ok or frame_bgr is None:
                break
            timestamp_s = float(capture.get(cv2.CAP_PROP_POS_MSEC)) / 1000.0
            if timestamp_s <= 0.0 and fps > 0.0:
                timestamp_s = frame_index / fps
            candidates = collect_prefusion_people_recall(
                detector,
                frame_bgr,
                person_threshold=args.person_conf,
                tile_size=args.tile_size,
                tile_overlap=args.tile_overlap,
            )
            inference_calls += detector.last_inference_calls
            for name, value in detector.last_stage_ms.items():
                stage_ms[name] = stage_ms.get(name, 0.0) + float(value)
            height, width = frame_bgr.shape[:2]
            frames.append(
                ResearchFrame(
                    args.video_id or video_path.name,
                    frame_index,
                    timestamp_s,
                    width,
                    height,
                    tuple(candidates),
                )
            )
            frame_index += 1
    finally:
        capture.release()

    if not frames:
        raise SystemExit("no frames decoded")
    wall_time_s = time.perf_counter() - started
    metadata = {
        "source_commit": args.source_commit,
        "video": args.video_id or video_path.name,
        "video_sha256": sha256_file(video_path),
        "detector": "current-yolo-onnx-prefusion",
        "model_sha256": sha256_file(model_path),
        "provider": ",".join(detector.providers),
        "config": {
            "input_size": args.input_size,
            "actual_input_width": detector.input_w,
            "actual_input_height": detector.input_h,
            "conf": args.conf,
            "decoder_iou": args.decoder_iou,
            "person_conf": args.person_conf,
            "tile_size": args.tile_size,
            "tile_overlap": args.tile_overlap,
            "decoder_local_nms": True,
            "final_cross_pass_fusion": "not_applied",
            "enhancement": "off",
        },
        "width": frames[0].width,
        "height": frames[0].height,
    }
    summary = {
        "policy_runs": len(frames),
        "inference_calls": inference_calls,
        "wall_time_s": wall_time_s,
        "stage_ms": stage_ms,
    }
    write_prefusion_dump(args.output, metadata, frames, summary)
    print(json.dumps(summary, sort_keys=True))
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SpectraTrack vNext A1 detection-fusion research")
    sub = parser.add_subparsers(dest="command", required=True)

    synthetic = sub.add_parser("synthetic", help="Run deterministic fusion stress cases")
    synthetic.add_argument("--output", required=True)
    synthetic.add_argument("--source-commit", required=True)

    collect = sub.add_parser("collect", help="Dump full-frame/tile candidates before final fusion")
    collect.add_argument("--model", required=True)
    collect.add_argument("--video", required=True)
    collect.add_argument("--output", required=True)
    collect.add_argument("--source-commit", required=True)
    collect.add_argument("--video-id", default="")
    collect.add_argument("--input-size", type=int, default=640)
    collect.add_argument("--conf", type=float, default=0.35)
    collect.add_argument("--decoder-iou", type=float, default=0.45)
    collect.add_argument("--person-conf", type=float, default=0.12)
    collect.add_argument("--tile-size", type=int, default=640)
    collect.add_argument("--tile-overlap", type=float, default=0.20)
    collect.add_argument("--max-frames", type=int, default=0)
    collect.add_argument("--cpu", action="store_true")

    replay = sub.add_parser("replay", help="Fuse prefusion dump into canonical tracker replay JSONL")
    replay.add_argument("--input", required=True)
    replay.add_argument("--output", required=True)
    replay.add_argument(
        "--method",
        choices=("hard-nms", "conservative-nmm", "weighted", "evidence-aware"),
        required=True,
    )
    replay.add_argument("--fusion-iou", type=float, default=0.55)
    replay.add_argument("--center-ratio", type=float, default=0.20)
    replay.add_argument("--size-ratio", type=float, default=1.80)
    replay.add_argument("--score-power", type=float, default=1.0)
    replay.add_argument("--full-weight", type=float, default=1.0)
    replay.add_argument("--tile-weight", type=float, default=1.0)
    replay.add_argument("--evidence-weak-score", type=float, default=0.12)
    replay.add_argument("--evidence-solo-score", type=float, default=0.20)
    replay.add_argument("--evidence-strong-score", type=float, default=0.35)
    replay.add_argument("--evidence-min-sources", type=int, default=2)
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    if args.command == "synthetic":
        report = synthetic_report(source_commit=args.source_commit)
        target = Path(args.output)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(report["methods"], sort_keys=True))
        return 0
    if args.command == "collect":
        return _collect_video(args)

    config = FusionConfig(
        iou_threshold=args.fusion_iou,
        center_ratio=args.center_ratio,
        size_ratio=args.size_ratio,
        score_power=args.score_power,
        full_weight=args.full_weight,
        tile_weight=args.tile_weight,
        evidence_weak_score=args.evidence_weak_score,
        evidence_solo_score=args.evidence_solo_score,
        evidence_strong_score=args.evidence_strong_score,
        evidence_min_sources=args.evidence_min_sources,
    )
    config.validate()
    convert_prefusion_to_replay(args.input, args.output, method=args.method, config=config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
