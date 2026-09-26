from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any, Iterable

from .integrity import sha256_file
from .tracker import bbox_iou

SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class GroundTruthObject:
    object_id: str | None
    label: str
    bbox: tuple[float, float, float, float]
    ignore: bool = False
    attributes: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class GroundTruthFrame:
    video: str
    frame: int
    tags: tuple[str, ...]
    objects: tuple[GroundTruthObject, ...]
    source: str | None = None
    source_frame: int | None = None
    source_sequence: str | None = None
    source_fps: float | None = None


@dataclass(frozen=True, slots=True)
class PredictedObject:
    bbox: tuple[float, float, float, float]
    label: str
    score: float
    track_id: int | None = None


def _validate_bbox(value: Any, where: str) -> tuple[float, float, float, float]:
    if not isinstance(value, list) or len(value) != 4:
        raise ValueError(f"{where}: bbox must be [x1, y1, x2, y2]")
    try:
        box = tuple(float(v) for v in value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{where}: bbox contains a non-numeric value") from exc
    if not all(math.isfinite(v) for v in box):
        raise ValueError(f"{where}: bbox must contain finite values")
    if box[2] <= box[0] or box[3] <= box[1]:
        raise ValueError(f"{where}: bbox must have positive width and height")
    return box


def load_ground_truth(path: str | Path) -> list[GroundTruthFrame]:
    source = Path(path)
    frames: list[GroundTruthFrame] = []
    seen: set[tuple[str, int]] = set()
    with source.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            raw = raw.strip()
            if not raw or raw.startswith("#"):
                continue
            try:
                data = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{source}:{line_number}: invalid JSON") from exc
            video = data.get("video")
            frame = data.get("frame")
            if not isinstance(video, str) or not video.strip():
                raise ValueError(f"{source}:{line_number}: video must be a non-empty string")
            if not isinstance(frame, int) or isinstance(frame, bool) or frame < 0:
                raise ValueError(f"{source}:{line_number}: frame must be a non-negative integer")
            source_path = data.get("source")
            if source_path is not None and (not isinstance(source_path, str) or not source_path.strip()):
                raise ValueError(f"{source}:{line_number}: source must be a non-empty string when present")
            source_frame = data.get("source_frame")
            if source_frame is not None and (
                not isinstance(source_frame, int) or isinstance(source_frame, bool) or source_frame < 0
            ):
                raise ValueError(f"{source}:{line_number}: source_frame must be a non-negative integer when present")
            source_sequence = data.get("source_sequence")
            if source_sequence is not None and (
                not isinstance(source_sequence, str) or not source_sequence.strip()
            ):
                raise ValueError(f"{source}:{line_number}: source_sequence must be a non-empty string when present")
            source_fps_raw = data.get("source_fps")
            source_fps = None
            if source_fps_raw is not None:
                if (
                    isinstance(source_fps_raw, bool)
                    or not isinstance(source_fps_raw, (int, float))
                    or not math.isfinite(float(source_fps_raw))
                    or float(source_fps_raw) <= 0.0
                ):
                    raise ValueError(f"{source}:{line_number}: source_fps must be finite and > 0 when present")
                source_fps = float(source_fps_raw)

            key = (video, frame)
            if key in seen:
                raise ValueError(f"{source}:{line_number}: duplicate frame {video!r}#{frame}")
            seen.add(key)

            tags_raw = data.get("tags", [])
            if not isinstance(tags_raw, list) or not all(isinstance(tag, str) and tag for tag in tags_raw):
                raise ValueError(f"{source}:{line_number}: tags must be a list of non-empty strings")

            objects_raw = data.get("objects", [])
            if not isinstance(objects_raw, list):
                raise ValueError(f"{source}:{line_number}: objects must be a list")
            objects: list[GroundTruthObject] = []
            object_ids: set[str] = set()
            for index, item in enumerate(objects_raw):
                where = f"{source}:{line_number}:objects[{index}]"
                if not isinstance(item, dict):
                    raise ValueError(f"{where}: object must be a JSON object")
                label = item.get("label")
                if not isinstance(label, str) or not label:
                    raise ValueError(f"{where}: label must be a non-empty string")
                object_id = item.get("id")
                if object_id is not None and (not isinstance(object_id, str) or not object_id):
                    raise ValueError(f"{where}: id must be a non-empty string when present")
                if object_id is not None:
                    if object_id in object_ids:
                        raise ValueError(f"{where}: duplicate object id {object_id!r} in frame")
                    object_ids.add(object_id)
                ignore = item.get("ignore", False)
                if not isinstance(ignore, bool):
                    raise ValueError(f"{where}: ignore must be a boolean")
                attributes_raw = item.get("attributes", [])
                if not isinstance(attributes_raw, list) or not all(
                    isinstance(attribute, str) and attribute for attribute in attributes_raw
                ):
                    raise ValueError(f"{where}: attributes must be a list of non-empty strings")
                objects.append(
                    GroundTruthObject(
                        object_id=object_id,
                        label=label,
                        bbox=_validate_bbox(item.get("bbox"), where),
                        ignore=ignore,
                        attributes=tuple(sorted(set(attributes_raw))),
                    )
                )
            frames.append(
                GroundTruthFrame(
                    video=video,
                    frame=frame,
                    tags=tuple(sorted(set(tags_raw))),
                    objects=tuple(objects),
                    source=source_path,
                    source_frame=source_frame,
                    source_sequence=source_sequence,
                    source_fps=source_fps,
                )
            )
    if not frames:
        raise ValueError(f"{source}: no ground-truth frames found")
    return sorted(frames, key=lambda item: (item.video, item.frame))


def ground_truth_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


_IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".webp"}


def image_sequence_files(path: str | Path) -> list[Path]:
    root = Path(path)
    if not root.is_dir():
        return []
    files = [
        item
        for item in root.iterdir()
        if item.is_file() and item.suffix.lower() in _IMAGE_EXTENSIONS
    ]
    return sorted(files, key=lambda item: item.name)


def qa_source_sha256(path: str | Path) -> str:
    source = Path(path)
    if source.is_file():
        return sha256_file(source)
    files = image_sequence_files(source)
    if not files:
        raise FileNotFoundError(f"QA source is neither a file nor an image-sequence directory: {source}")
    digest = hashlib.sha256()
    for item in files:
        relative = item.relative_to(source).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        with item.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def inspect_qa_source(path: str | Path, *, fps_override: float | None = None) -> dict[str, Any]:
    import cv2

    source = Path(path)
    if source.is_dir():
        files = image_sequence_files(source)
        if not files:
            raise RuntimeError(f"Image-sequence directory is empty: {source}")
        first = cv2.imread(str(files[0]), cv2.IMREAD_COLOR)
        if first is None:
            raise RuntimeError(f"Cannot read first image-sequence frame: {files[0]}")
        height, width = first.shape[:2]
        return {
            "kind": "image_sequence",
            "width": int(width),
            "height": int(height),
            "fps": fps_override,
            "frame_count": len(files),
            "sha256": qa_source_sha256(source),
        }

    if not source.is_file():
        raise FileNotFoundError(source)
    if source.suffix.lower() in _IMAGE_EXTENSIONS:
        image = cv2.imread(str(source), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Cannot read benchmark image: {source}")
        height, width = image.shape[:2]
        return {
            "kind": "image",
            "width": int(width),
            "height": int(height),
            "fps": None,
            "frame_count": 1,
            "sha256": sha256_file(source),
        }

    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open benchmark video: {source}")
    try:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        cap.release()
    if width <= 0 or height <= 0:
        raise RuntimeError(f"Invalid benchmark source dimensions for {source}: {width}x{height}")
    measured_fps = fps if math.isfinite(fps) and fps > 0.0 else None
    return {
        "kind": "video",
        "width": width,
        "height": height,
        "fps": fps_override if fps_override is not None else measured_fps,
        "frame_count": frame_count if frame_count > 0 else None,
        "sha256": sha256_file(source),
    }


def _truth_key(frame: GroundTruthFrame, index: int, obj: GroundTruthObject) -> str:
    suffix = obj.object_id if obj.object_id is not None else f"index:{index}"
    return f"{frame.video}#{frame.frame}#{suffix}"


def _size_bin(obj: GroundTruthObject) -> str:
    height = obj.bbox[3] - obj.bbox[1]
    if height < 24:
        return "height_lt_24"
    if height < 48:
        return "height_24_47"
    if height < 96:
        return "height_48_95"
    return "height_ge_96"


def _match_objects(
    truth: list[tuple[int, GroundTruthObject]],
    predictions: list[PredictedObject],
    iou_threshold: float,
) -> tuple[dict[int, int], set[int]]:
    candidates: list[tuple[float, int, int]] = []
    for truth_index, truth_obj in truth:
        if truth_obj.ignore:
            continue
        for pred_index, pred in enumerate(predictions):
            if pred.label != truth_obj.label:
                continue
            overlap = bbox_iou(truth_obj.bbox, pred.bbox)
            if overlap >= iou_threshold:
                candidates.append((overlap, truth_index, pred_index))
    candidates.sort(reverse=True)
    matched_truth: set[int] = set()
    matched_predictions: set[int] = set()
    matches: dict[int, int] = {}
    for _, truth_index, pred_index in candidates:
        if truth_index in matched_truth or pred_index in matched_predictions:
            continue
        matched_truth.add(truth_index)
        matched_predictions.add(pred_index)
        matches[truth_index] = pred_index

    ignored_truth = [obj for _, obj in truth if obj.ignore]
    for pred_index, pred in enumerate(predictions):
        if pred_index in matched_predictions:
            continue
        if any(pred.label == ignored.label and bbox_iou(pred.bbox, ignored.bbox) >= iou_threshold for ignored in ignored_truth):
            matched_predictions.add(pred_index)
    return matches, matched_predictions


def _empty_counts() -> dict[str, int]:
    return {"tp": 0, "fp": 0, "fn": 0, "gt": 0, "predictions": 0}


def _rates(counts: dict[str, int]) -> dict[str, float | int]:
    tp = counts["tp"]
    fp = counts["fp"]
    fn = counts["fn"]
    precision = tp / (tp + fp) if tp + fp else 1.0
    recall = tp / (tp + fn) if tp + fn else 1.0
    return {**counts, "precision": precision, "recall": recall}


def _mean(values: list[float] | list[int]) -> float | None:
    if not values:
        return None
    return float(sum(values)) / len(values)


def _gt_relative_box(
    truth_bbox: tuple[float, float, float, float],
    predicted_bbox: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    tx1, ty1, tx2, ty2 = truth_bbox
    px1, py1, px2, py2 = predicted_bbox
    truth_w = max(tx2 - tx1, 1e-12)
    truth_h = max(ty2 - ty1, 1e-12)
    truth_cx = (tx1 + tx2) * 0.5
    truth_cy = (ty1 + ty2) * 0.5
    return (
        (px1 - truth_cx) / truth_w,
        (py1 - truth_cy) / truth_h,
        (px2 - truth_cx) / truth_w,
        (py2 - truth_cy) / truth_h,
    )


def _bbox_stability_metrics(
    samples: dict[
        tuple[str, str],
        list[
            tuple[
                int,
                tuple[float, float, float, float],
                tuple[float, float, float, float],
            ]
        ],
    ],
) -> dict[str, Any]:
    center_jitter: list[float] = []
    width_jitter: list[float] = []
    height_jitter: list[float] = []
    area_jitter: list[float] = []
    temporal_iou: list[float] = []

    for identity_samples in samples.values():
        normalized: list[tuple[int, tuple[float, float, float, float]]] = []
        for frame_index, truth_bbox, predicted_bbox in sorted(identity_samples):
            normalized.append((frame_index, _gt_relative_box(truth_bbox, predicted_bbox)))
        for (_, previous), (_, current) in zip(normalized, normalized[1:]):
            prev_w = max(previous[2] - previous[0], 1e-12)
            prev_h = max(previous[3] - previous[1], 1e-12)
            curr_w = max(current[2] - current[0], 1e-12)
            curr_h = max(current[3] - current[1], 1e-12)
            prev_cx = (previous[0] + previous[2]) * 0.5
            prev_cy = (previous[1] + previous[3]) * 0.5
            curr_cx = (current[0] + current[2]) * 0.5
            curr_cy = (current[1] + current[3]) * 0.5
            center_jitter.append(math.hypot(curr_cx - prev_cx, curr_cy - prev_cy))
            width_jitter.append(abs(math.log(curr_w / prev_w)))
            height_jitter.append(abs(math.log(curr_h / prev_h)))
            area_jitter.append(abs(math.log((curr_w * curr_h) / (prev_w * prev_h))))
            temporal_iou.append(bbox_iou(previous, current))

    return {
        "pair_count": len(center_jitter),
        "normalized_center_jitter_mean": _mean(center_jitter),
        "width_log_jitter_mean": _mean(width_jitter),
        "height_log_jitter_mean": _mean(height_jitter),
        "area_log_jitter_mean": _mean(area_jitter),
        "temporal_iou_mean": _mean(temporal_iou),
        "note": (
            "Consecutive matched boxes are expressed relative to each frame's GT center/size before "
            "differencing, reducing genuine person/camera motion from the stability signal."
        ),
    }


def evaluate_frames(
    frames: Iterable[GroundTruthFrame],
    predictions_by_frame: dict[tuple[str, int], list[PredictedObject]],
    tracks_by_frame: dict[tuple[str, int], list[PredictedObject]] | None = None,
    *,
    label: str = "person",
    iou_threshold: float = 0.5,
) -> dict[str, Any]:
    if not 0.0 < iou_threshold <= 1.0:
        raise ValueError("iou_threshold must be in (0, 1]")
    ordered = sorted(frames, key=lambda item: (item.video, item.frame))
    overall = _empty_counts()
    by_tag: dict[str, dict[str, int]] = defaultdict(_empty_counts)
    by_size: dict[str, dict[str, int]] = defaultdict(_empty_counts)
    by_attribute: dict[str, dict[str, int]] = defaultdict(_empty_counts)
    matched_keys: list[str] = []
    missed_keys: list[str] = []
    missing_prediction_frames: list[str] = []

    track_tp = 0
    track_fn = 0
    id_switches = 0
    fragmentations = 0
    identity_state: dict[tuple[str, str], dict[str, Any]] = {}
    detection_stability_samples: dict[
        tuple[str, str],
        list[
            tuple[
                int,
                tuple[float, float, float, float],
                tuple[float, float, float, float],
            ]
        ],
    ] = defaultdict(list)
    tracking_stability_samples: dict[
        tuple[str, str],
        list[
            tuple[
                int,
                tuple[float, float, float, float],
                tuple[float, float, float, float],
            ]
        ],
    ] = defaultdict(list)
    uninterrupted_track_lengths: list[int] = []
    recovery_latencies: list[int] = []

    for frame in ordered:
        key = (frame.video, frame.frame)
        if key not in predictions_by_frame:
            missing_prediction_frames.append(f"{frame.video}#{frame.frame}")
        predictions = [item for item in predictions_by_frame.get(key, []) if item.label == label]
        truth = [(index, obj) for index, obj in enumerate(frame.objects) if obj.label == label]
        valid_truth = [(index, obj) for index, obj in truth if not obj.ignore]
        matches, used_predictions = _match_objects(truth, predictions, iou_threshold)

        tp = len(matches)
        fn = len(valid_truth) - tp
        fp = len(predictions) - len(used_predictions)
        overall["tp"] += tp
        overall["fn"] += fn
        overall["fp"] += fp
        overall["gt"] += len(valid_truth)
        overall["predictions"] += len(predictions)

        for tag in frame.tags:
            counts = by_tag[tag]
            counts["tp"] += tp
            counts["fn"] += fn
            counts["fp"] += fp
            counts["gt"] += len(valid_truth)
            counts["predictions"] += len(predictions)

        for truth_index, obj in valid_truth:
            object_key = _truth_key(frame, truth_index, obj)
            matched = truth_index in matches
            if matched:
                matched_keys.append(object_key)
                if obj.object_id is not None:
                    detection_stability_samples[(frame.video, obj.object_id)].append(
                        (frame.frame, obj.bbox, predictions[matches[truth_index]].bbox)
                    )
            else:
                missed_keys.append(object_key)
            size_counts = by_size[_size_bin(obj)]
            size_counts["gt"] += 1
            if matched:
                size_counts["tp"] += 1
            else:
                size_counts["fn"] += 1
            for attribute in obj.attributes:
                counts = by_attribute[attribute]
                counts["gt"] += 1
                if matched:
                    counts["tp"] += 1
                else:
                    counts["fn"] += 1

        if tracks_by_frame is None:
            continue
        tracks = [item for item in tracks_by_frame.get(key, []) if item.label == label and item.track_id is not None]
        track_matches, _ = _match_objects(truth, tracks, iou_threshold)
        track_tp += len(track_matches)
        track_fn += len(valid_truth) - len(track_matches)
        for truth_index, obj in valid_truth:
            if obj.object_id is None:
                continue
            state_key = (frame.video, obj.object_id)
            state = identity_state.setdefault(
                state_key,
                {
                    "last_track_id": None,
                    "gap": False,
                    "seen_match": False,
                    "current_run": 0,
                    "last_match_frame": None,
                },
            )
            pred_index = track_matches.get(truth_index)
            if pred_index is None:
                if state["current_run"] > 0:
                    uninterrupted_track_lengths.append(int(state["current_run"]))
                    state["current_run"] = 0
                if state["seen_match"]:
                    state["gap"] = True
                continue
            current_track_id = tracks[pred_index].track_id
            tracking_stability_samples[state_key].append(
                (frame.frame, obj.bbox, tracks[pred_index].bbox)
            )
            if state["seen_match"] and state["last_track_id"] != current_track_id:
                id_switches += 1
                if state["current_run"] > 0:
                    uninterrupted_track_lengths.append(int(state["current_run"]))
                    state["current_run"] = 0
            if state["seen_match"] and state["gap"]:
                fragmentations += 1
                if state["last_match_frame"] is not None:
                    recovery_latencies.append(frame.frame - int(state["last_match_frame"]))
            state["last_track_id"] = current_track_id
            state["seen_match"] = True
            state["gap"] = False
            state["current_run"] += 1
            state["last_match_frame"] = frame.frame

    for state in identity_state.values():
        if state["current_run"] > 0:
            uninterrupted_track_lengths.append(int(state["current_run"]))

    metrics = _rates(overall)
    metrics["false_negatives"] = overall["fn"]
    metrics["false_positives"] = overall["fp"]
    metrics["matched_ground_truth"] = sorted(matched_keys)
    metrics["missed_ground_truth"] = sorted(missed_keys)
    metrics["missing_prediction_frames"] = sorted(missing_prediction_frames)
    metrics["by_tag"] = {name: _rates(counts) for name, counts in sorted(by_tag.items())}
    metrics["by_size"] = {name: _rates(counts) for name, counts in sorted(by_size.items())}
    metrics["by_attribute"] = {
        name: {
            "tp": counts["tp"],
            "fn": counts["fn"],
            "gt": counts["gt"],
            "recall": counts["tp"] / counts["gt"] if counts["gt"] else 1.0,
        }
        for name, counts in sorted(by_attribute.items())
    }
    track_total = track_tp + track_fn
    metrics["tracking"] = {
        "matched_gt": track_tp,
        "missed_gt": track_fn,
        "recall": track_tp / track_total if track_total else None,
        "id_switches": id_switches,
        "fragmentations": fragmentations,
        "mean_uninterrupted_track_length_annotated_frames": _mean(uninterrupted_track_lengths),
        "uninterrupted_track_segments": len(uninterrupted_track_lengths),
        "mean_recovery_latency_frames": _mean(recovery_latencies),
        "max_recovery_latency_frames": max(recovery_latencies) if recovery_latencies else None,
        "recovery_events": len(recovery_latencies),
        "note": (
            "ID/continuity metrics require stable GT ids. Track length counts consecutive annotated "
            "matches; recovery latency uses source-frame index distance and unannotated intervals are "
            "otherwise not scored."
        ),
    }
    metrics["bbox_stability"] = {
        "detection": _bbox_stability_metrics(detection_stability_samples),
        "tracking": _bbox_stability_metrics(tracking_stability_samples),
    }
    return metrics


def compare_results(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    *,
    max_new_false_negatives: int = 0,
    max_recall_drop: float = 0.0,
    max_precision_drop: float = 0.0,
    max_id_switch_increase: int = 0,
    max_fragmentation_increase: int = 0,
) -> dict[str, Any]:
    if baseline.get("ground_truth_sha256") != candidate.get("ground_truth_sha256"):
        raise ValueError("Cannot compare results produced from different ground-truth files")
    for key in ("label", "match_iou"):
        if baseline.get("evaluation", {}).get(key) != candidate.get("evaluation", {}).get(key):
            raise ValueError(f"Cannot compare results with different evaluation setting: {key}")

    base_metrics = baseline["metrics"]
    cand_metrics = candidate["metrics"]
    base_matched = set(base_metrics.get("matched_ground_truth", []))
    cand_matched = set(cand_metrics.get("matched_ground_truth", []))
    new_false_negatives = sorted(base_matched - cand_matched)
    recovered = sorted(cand_matched - base_matched)
    recall_drop = float(base_metrics["recall"]) - float(cand_metrics["recall"])
    precision_drop = float(base_metrics["precision"]) - float(cand_metrics["precision"])
    base_tracking = base_metrics.get("tracking", {})
    cand_tracking = cand_metrics.get("tracking", {})
    id_switch_increase = int(cand_tracking.get("id_switches", 0)) - int(base_tracking.get("id_switches", 0))
    fragmentation_increase = int(cand_tracking.get("fragmentations", 0)) - int(base_tracking.get("fragmentations", 0))

    reasons: list[str] = []
    if len(new_false_negatives) > max_new_false_negatives:
        reasons.append(
            f"new false negatives {len(new_false_negatives)} > allowed {max_new_false_negatives}"
        )
    if recall_drop > max_recall_drop + 1e-12:
        reasons.append(f"recall drop {recall_drop:.6f} > allowed {max_recall_drop:.6f}")
    if precision_drop > max_precision_drop + 1e-12:
        reasons.append(f"precision drop {precision_drop:.6f} > allowed {max_precision_drop:.6f}")
    if id_switch_increase > max_id_switch_increase:
        reasons.append(f"ID-switch increase {id_switch_increase} > allowed {max_id_switch_increase}")
    if fragmentation_increase > max_fragmentation_increase:
        reasons.append(
            f"fragmentation increase {fragmentation_increase} > allowed {max_fragmentation_increase}"
        )

    base_perf = baseline.get("performance", {})
    cand_perf = candidate.get("performance", {})
    return {
        "schema_version": SCHEMA_VERSION,
        "baseline": baseline.get("run_name"),
        "candidate": candidate.get("run_name"),
        "passed": not reasons,
        "reasons": reasons,
        "new_false_negatives": new_false_negatives,
        "recovered_ground_truth": recovered,
        "delta": {
            "recall": float(cand_metrics["recall"]) - float(base_metrics["recall"]),
            "precision": float(cand_metrics["precision"]) - float(base_metrics["precision"]),
            "false_negatives": int(cand_metrics["false_negatives"]) - int(base_metrics["false_negatives"]),
            "false_positives": int(cand_metrics["false_positives"]) - int(base_metrics["false_positives"]),
            "id_switches": id_switch_increase,
            "fragmentations": fragmentation_increase,
            "fps": _optional_delta(cand_perf.get("fps"), base_perf.get("fps")),
            "peak_vram_mb": _optional_delta(cand_perf.get("peak_vram_mb"), base_perf.get("peak_vram_mb")),
        },
    }


def _optional_delta(candidate: Any, baseline: Any) -> float | None:
    if candidate is None or baseline is None:
        return None
    return float(candidate) - float(baseline)


def _json_dump(path: str | Path, payload: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _load_result(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if data.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"Unsupported benchmark result schema in {path}")
    return data


def _detect_with_runtime_policy(detector, frame, args: argparse.Namespace):
    detector_mode = getattr(args, "detector_mode", "standard")
    if detector_mode == "standard":
        return detector.detect(frame)
    if detector_mode != "people-recall":
        raise ValueError("detector_mode must be standard or people-recall")

    enhancement_mode = getattr(args, "people_recall_enhancement", "off")
    if enhancement_mode not in {"off", "adaptive"}:
        raise ValueError("people_recall_enhancement must be off or adaptive")
    return detector.detect_people_recall(
        frame,
        person_threshold=float(getattr(args, "person_conf", 0.12)),
        tile_size=int(getattr(args, "person_tile_size", 640)),
        tile_overlap=float(getattr(args, "person_tile_overlap", 0.20)),
        merge_iou_threshold=float(getattr(args, "person_merge_iou", 0.55)),
        enhancement_mode=enhancement_mode,
    )


def run_current_pipeline(args: argparse.Namespace) -> dict[str, Any]:
    import cv2

    from .appearance import attach_appearance
    from .detector import YoloOnnxDetector
    from .integrity import sha256_file
    from .motion import GlobalMotionEstimator
    from .tracker import MultiObjectTracker

    ground_truth = load_ground_truth(args.ground_truth)
    grouped: dict[str, list[GroundTruthFrame]] = defaultdict(list)
    for item in ground_truth:
        grouped[item.video].append(item)

    detector = YoloOnnxDetector(
        args.model,
        input_size=args.input_size,
        conf_threshold=args.conf,
        iou_threshold=args.nms_iou,
        prefer_gpu=not args.cpu,
    )
    predictions_by_frame: dict[tuple[str, int], list[PredictedObject]] = {}
    tracks_by_frame: dict[tuple[str, int], list[PredictedObject]] = {}
    processed_frames = 0
    inference_seconds = 0.0
    detector_policy_runs = 0
    onnx_inference_calls = 0
    source_seconds = 0.0
    source_seconds_known = True
    input_videos: list[dict[str, Any]] = []
    total_start = time.perf_counter()

    for video, video_annotations in sorted(grouped.items()):
        source_names = {item.source or item.video for item in video_annotations}
        source_fps_values = {item.source_fps for item in video_annotations if item.source_fps is not None}
        if len(source_names) != 1:
            raise ValueError(f"Ground truth {video!r} references multiple physical sources: {sorted(source_names)}")
        if len(source_fps_values) > 1:
            raise ValueError(f"Ground truth {video!r} has conflicting source_fps values")
        source_name = next(iter(source_names))
        source_path = Path(args.video_root) / source_name
        fps_override = next(iter(source_fps_values)) if source_fps_values else None
        meta = inspect_qa_source(source_path, fps_override=fps_override)

        annotated_frames = {item.frame for item in video_annotations}
        last_needed = max(annotated_frames)
        frame_count = meta.get("frame_count")
        if frame_count is not None and last_needed >= int(frame_count):
            raise RuntimeError(
                f"Source {source_path} ends before annotated frame {last_needed}; frame_count={frame_count}"
            )

        motion = GlobalMotionEstimator()
        tracker = MultiObjectTracker()
        video_processed_frames = 0

        def process_frame(frame_index: int, frame) -> None:
            nonlocal processed_frames, inference_seconds, detector_policy_runs
            nonlocal onnx_inference_calls, video_processed_frames
            cam = motion.update(frame)
            infer_start = time.perf_counter()
            detections = _detect_with_runtime_policy(detector, frame, args)
            inference_seconds += time.perf_counter() - infer_start
            detector_policy_runs += 1
            onnx_inference_calls += int(getattr(detector, "last_inference_calls", 0))
            if not args.no_appearance:
                attach_appearance(frame, detections)
            camera_motion = (cam.dx, cam.dy) if cam.valid else (0.0, 0.0)
            camera_transform = cam.affine if cam.valid else None
            tracks = tracker.update(
                detections,
                camera_motion=camera_motion,
                camera_transform=camera_transform,
            )
            processed_frames += 1
            video_processed_frames += 1
            if frame_index in annotated_frames:
                key = (video, frame_index)
                predictions_by_frame[key] = [
                    PredictedObject(tuple(det.bbox), det.label, float(det.score)) for det in detections
                ]
                tracks_by_frame[key] = [
                    PredictedObject(tuple(track.bbox), track.label, float(track.score), int(track.track_id))
                    for track in tracks
                ]

        if meta["kind"] == "image":
            if last_needed != 0:
                raise RuntimeError(f"Still-image source {source_path} can only use canonical frame 0")
            frame = cv2.imread(str(source_path), cv2.IMREAD_COLOR)
            if frame is None:
                raise RuntimeError(f"Cannot read benchmark image: {source_path}")
            process_frame(0, frame)
        elif meta["kind"] == "image_sequence":
            sequence_files = image_sequence_files(source_path)
            for frame_index in range(last_needed + 1):
                frame = cv2.imread(str(sequence_files[frame_index]), cv2.IMREAD_COLOR)
                if frame is None:
                    raise RuntimeError(f"Cannot read image-sequence frame: {sequence_files[frame_index]}")
                process_frame(frame_index, frame)
        else:
            cap = cv2.VideoCapture(str(source_path))
            if not cap.isOpened():
                raise RuntimeError(f"Cannot open benchmark video: {source_path}")
            frame_index = 0
            try:
                while frame_index <= last_needed:
                    ok, frame = cap.read()
                    if not ok or frame is None:
                        raise RuntimeError(
                            f"Video {source_path} ended before annotated frame {last_needed}; stopped at {frame_index}"
                        )
                    process_frame(frame_index, frame)
                    frame_index += 1
            finally:
                cap.release()

        fps = meta.get("fps")
        if fps is not None and float(fps) > 0.0:
            source_seconds += video_processed_frames / float(fps)
        else:
            source_seconds_known = False
        input_videos.append(
            {
                "video": video,
                "source": source_name,
                "source_kind": meta["kind"],
                "sha256": meta["sha256"],
                "width": int(meta["width"]),
                "height": int(meta["height"]),
                "fps": fps,
                "frame_count": frame_count,
                "processed_frames": video_processed_frames,
            }
        )

    wall_seconds = time.perf_counter() - total_start
    measured_source_seconds = source_seconds if source_seconds_known and source_seconds > 0.0 else None
    metrics = evaluate_frames(
        ground_truth,
        predictions_by_frame,
        tracks_by_frame,
        label=args.label,
        iou_threshold=args.match_iou,
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "run_name": args.run_name,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "revision": args.revision,
        "ground_truth_sha256": ground_truth_sha256(args.ground_truth),
        "input_videos": input_videos,
        "model": {
            "path": str(args.model),
            "sha256": sha256_file(args.model),
            "providers": list(detector.providers),
        },
        "evaluation": {"label": args.label, "match_iou": args.match_iou},
        "settings": {
            "requested_input_size": args.input_size,
            "actual_input_width": detector.input_w,
            "actual_input_height": detector.input_h,
            "conf": args.conf,
            "nms_iou": args.nms_iou,
            "prefer_gpu": not args.cpu,
            "appearance": not args.no_appearance,
            "detector_mode": args.detector_mode,
            "person_conf": args.person_conf if args.detector_mode == "people-recall" else None,
            "person_tile_size": args.person_tile_size if args.detector_mode == "people-recall" else None,
            "person_tile_overlap": args.person_tile_overlap if args.detector_mode == "people-recall" else None,
            "person_merge_iou": args.person_merge_iou if args.detector_mode == "people-recall" else None,
            "people_recall_enhancement": (
                args.people_recall_enhancement if args.detector_mode == "people-recall" else None
            ),
        },
        "performance": {
            "processed_frames": processed_frames,
            "wall_seconds": wall_seconds,
            "fps": processed_frames / wall_seconds if wall_seconds > 0 else None,
            "detector_seconds": inference_seconds,
            "detector_fps": processed_frames / inference_seconds if inference_seconds > 0 else None,
            "detector_policy_runs": detector_policy_runs,
            "onnx_inference_calls": onnx_inference_calls,
            "onnx_calls_per_frame": (
                onnx_inference_calls / processed_frames if processed_frames > 0 else None
            ),
            "source_seconds": measured_source_seconds,
            "processing_seconds_per_source_second": (
                wall_seconds / measured_source_seconds if measured_source_seconds else None
            ),
            "peak_vram_mb": args.peak_vram_mb,
            "vram_source": args.vram_source if args.peak_vram_mb is not None else None,
        },
        "metrics": metrics,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SpectraTrack annotated QA benchmark and regression checker")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="Run the current detector/tracker on annotated local videos")
    run.add_argument("--ground-truth", required=True)
    run.add_argument("--video-root", required=True)
    run.add_argument("--model", required=True)
    run.add_argument("--output", required=True)
    run.add_argument("--run-name", required=True)
    run.add_argument("--revision", default=None, help="Git SHA/branch label recorded as metadata")
    run.add_argument("--label", default="person")
    run.add_argument("--match-iou", type=float, default=0.5)
    run.add_argument("--input-size", type=int, default=640)
    run.add_argument("--conf", type=float, default=0.35)
    run.add_argument("--nms-iou", type=float, default=0.45)
    run.add_argument("--detector-mode", choices=("standard", "people-recall"), default="standard")
    run.add_argument("--person-conf", type=float, default=0.12)
    run.add_argument("--person-tile-size", type=int, default=640)
    run.add_argument("--person-tile-overlap", type=float, default=0.20)
    run.add_argument("--person-merge-iou", type=float, default=0.55)
    run.add_argument("--people-recall-enhancement", choices=("off", "adaptive"), default="off")
    run.add_argument("--cpu", action="store_true")
    run.add_argument("--no-appearance", action="store_true")
    run.add_argument(
        "--peak-vram-mb",
        type=float,
        default=None,
        help="Optional externally measured peak VRAM; no value is fabricated when unavailable",
    )
    run.add_argument("--vram-source", default=None, help="How --peak-vram-mb was measured")

    compare = sub.add_parser("compare", help="Compare candidate result against a baseline result")
    compare.add_argument("--baseline", required=True)
    compare.add_argument("--candidate", required=True)
    compare.add_argument("--output")
    compare.add_argument("--max-new-fn", type=int, default=0)
    compare.add_argument("--max-recall-drop", type=float, default=0.0)
    compare.add_argument("--max-precision-drop", type=float, default=0.0)
    compare.add_argument("--max-id-switch-increase", type=int, default=0)
    compare.add_argument("--max-fragmentation-increase", type=int, default=0)
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    if args.command == "run":
        if args.peak_vram_mb is not None and not args.vram_source:
            parser.error("--vram-source is required when --peak-vram-mb is supplied")
        if not 0.0 <= args.person_conf <= 1.0:
            parser.error("--person-conf must be in [0, 1]")
        if args.person_tile_size <= 0:
            parser.error("--person-tile-size must be > 0")
        if not 0.0 <= args.person_tile_overlap < 1.0:
            parser.error("--person-tile-overlap must satisfy 0 <= overlap < 1")
        if not 0.0 < args.person_merge_iou <= 1.0:
            parser.error("--person-merge-iou must be in (0, 1]")
        if args.people_recall_enhancement == "adaptive" and args.detector_mode != "people-recall":
            parser.error("--people-recall-enhancement adaptive requires --detector-mode people-recall")
        if args.people_recall_enhancement == "adaptive" and args.person_conf <= 0.0:
            parser.error("adaptive people-recall requires --person-conf > 0")
        result = run_current_pipeline(args)
        _json_dump(args.output, result)
        metrics = result["metrics"]
        perf = result["performance"]
        print(
            f"run={result['run_name']} person_recall={metrics['recall']:.4f} "
            f"person_precision={metrics['precision']:.4f} fn={metrics['false_negatives']} "
            f"fp={metrics['false_positives']} id_switches={metrics['tracking']['id_switches']} "
            f"fragmentations={metrics['tracking']['fragmentations']} fps={perf['fps']:.2f}"
        )
        if perf["peak_vram_mb"] is None:
            print("peak_vram_mb=unavailable")
        return 0

    baseline = _load_result(args.baseline)
    candidate = _load_result(args.candidate)
    comparison = compare_results(
        baseline,
        candidate,
        max_new_false_negatives=args.max_new_fn,
        max_recall_drop=args.max_recall_drop,
        max_precision_drop=args.max_precision_drop,
        max_id_switch_increase=args.max_id_switch_increase,
        max_fragmentation_increase=args.max_fragmentation_increase,
    )
    if args.output:
        _json_dump(args.output, comparison)
    for item in comparison["new_false_negatives"][:50]:
        print(f"NEW FALSE NEGATIVE: {item}")
    if len(comparison["new_false_negatives"]) > 50:
        print(f"... {len(comparison['new_false_negatives']) - 50} more")
    print(json.dumps({key: comparison[key] for key in ("baseline", "candidate", "passed", "reasons", "delta")}, indent=2))
    return 0 if comparison["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
