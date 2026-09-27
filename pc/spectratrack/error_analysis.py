from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable

from .qa_benchmark import (
    GroundTruthFrame,
    GroundTruthObject,
    PredictedObject,
    _match_objects,
    _truth_key,
    ground_truth_sha256,
    load_ground_truth,
)

SCHEMA_VERSION = 1
MAX_TOP_K = 25
DEFAULT_TOP_K = 5
LOCALIZATION_MIN_IOU = 0.10
BACKGROUND_MAX_IOU = 0.05

TAXONOMY = (
    "missed_small_person",
    "low_contrast_night_darkness",
    "occlusion_crowd",
    "localization_iou_miss",
    "duplicate_fragmented_detection",
    "background_false_positive",
    "edge_of_frame",
    "scale_extreme_aspect",
    "unknown",
)

_LOW_LIGHT_TERMS = {
    "dark",
    "darkness",
    "night",
    "nighttime",
    "lowlight",
    "low_light",
    "low-contrast",
    "low_contrast",
    "lowcontrast",
}
_OCCLUSION_TERMS = {"occluded", "occlusion", "crowd", "crowded", "crowding"}
_SMALL_TERMS = {"small", "tiny", "distant", "small_people", "small_person"}
_EDGE_TERMS = {"edge", "edge_of_frame", "truncated", "cropped"}
_SCALE_TERMS = {"extreme_scale", "extreme_aspect", "unusual_aspect"}


@dataclass(frozen=True, slots=True, order=True)
class FrameKey:
    video: str
    frame: int

    @property
    def frame_id(self) -> str:
        return f"{self.video}#{self.frame}"


@dataclass(frozen=True, slots=True)
class ManifestSelection:
    frames: tuple[FrameKey, ...]
    file_sha256: str
    frame_ids_sha256: str


@dataclass(frozen=True, slots=True)
class ObservationFrame:
    key: FrameKey
    width: int | None
    height: int | None
    predictions: tuple[PredictedObject, ...]


@dataclass(frozen=True, slots=True)
class FailureEvent:
    event_id: str
    kind: str
    frame_id: str | None
    categories: tuple[str, ...]
    severity: float
    evidence: dict[str, Any]


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _normalize_term(value: str) -> str:
    return value.strip().lower().replace(" ", "_")


def _parse_frame_entry(entry: Any, where: str) -> FrameKey:
    if isinstance(entry, str):
        video, sep, raw_frame = entry.rpartition("#")
        if not sep or not video:
            raise ValueError(f"{where}: frame string must be VIDEO#FRAME")
        try:
            frame = int(raw_frame)
        except ValueError as exc:
            raise ValueError(f"{where}: invalid frame number") from exc
        if frame < 0:
            raise ValueError(f"{where}: frame must be non-negative")
        return FrameKey(video, frame)
    if not isinstance(entry, dict):
        raise ValueError(f"{where}: frame entry must be an object or VIDEO#FRAME string")
    video = entry.get("video")
    frame = entry.get("frame")
    if not isinstance(video, str) or not video:
        raise ValueError(f"{where}: video must be a non-empty string")
    if not isinstance(frame, int) or isinstance(frame, bool) or frame < 0:
        raise ValueError(f"{where}: frame must be a non-negative integer")
    return FrameKey(video, frame)


def load_frame_manifest(path: str | Path) -> ManifestSelection:
    source = Path(path)
    raw = source.read_bytes()
    stripped = raw.decode("utf-8-sig").strip()
    if not stripped:
        raise ValueError(f"{source}: empty frame manifest")

    entries: list[Any]
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        entries = []
        for line_number, line in enumerate(stripped.splitlines(), start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{source}:{line_number}: invalid manifest JSONL") from exc
    else:
        if isinstance(parsed, list):
            entries = parsed
        elif isinstance(parsed, dict):
            candidates = parsed.get("frames", parsed.get("selected_frames"))
            if not isinstance(candidates, list):
                raise ValueError(f"{source}: JSON manifest must contain a frames list")
            entries = candidates
        else:
            raise ValueError(f"{source}: manifest must be a JSON object, list, or JSONL")

    frames = [_parse_frame_entry(entry, f"{source}:frames[{index}]") for index, entry in enumerate(entries)]
    if not frames:
        raise ValueError(f"{source}: no frame IDs found")
    if len(set(frames)) != len(frames):
        raise ValueError(f"{source}: duplicate frame IDs")
    ordered = tuple(sorted(frames))
    frame_ids_sha256 = _canonical_sha256([item.frame_id for item in ordered])
    return ManifestSelection(ordered, hashlib.sha256(raw).hexdigest(), frame_ids_sha256)


def _parse_prediction(entry: Any, where: str) -> PredictedObject:
    if not isinstance(entry, dict):
        raise ValueError(f"{where}: prediction must be an object")
    bbox = entry.get("bbox")
    if not isinstance(bbox, list) or len(bbox) != 4:
        raise ValueError(f"{where}: bbox must contain four numbers")
    try:
        box = tuple(float(value) for value in bbox)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{where}: bbox must contain four numbers") from exc
    if not all(math.isfinite(value) for value in box) or box[2] <= box[0] or box[3] <= box[1]:
        raise ValueError(f"{where}: bbox must be finite with positive width and height")
    label = entry.get("label")
    score = entry.get("score", 1.0)
    if not isinstance(label, str) or not label:
        raise ValueError(f"{where}: label must be a non-empty string")
    if not isinstance(score, (int, float)) or isinstance(score, bool) or not math.isfinite(float(score)):
        raise ValueError(f"{where}: score must be finite")
    return PredictedObject(box, label, float(score), None)


def load_observations(path: str | Path) -> dict[FrameKey, ObservationFrame]:
    source = Path(path)
    frames: dict[FrameKey, ObservationFrame] = {}
    with source.open("r", encoding="utf-8-sig") as handle:
        for line_number, raw in enumerate(handle, start=1):
            raw = raw.strip()
            if not raw or raw.startswith("#"):
                continue
            try:
                data = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{source}:{line_number}: invalid JSON") from exc
            key = _parse_frame_entry(data, f"{source}:{line_number}")
            if key in frames:
                raise ValueError(f"{source}:{line_number}: duplicate frame {key.frame_id}")
            width = data.get("width")
            height = data.get("height")
            if width is not None and (not isinstance(width, int) or isinstance(width, bool) or width <= 0):
                raise ValueError(f"{source}:{line_number}: width must be a positive integer")
            if height is not None and (not isinstance(height, int) or isinstance(height, bool) or height <= 0):
                raise ValueError(f"{source}:{line_number}: height must be a positive integer")
            predictions_raw = data.get("predictions", data.get("detections", []))
            if not isinstance(predictions_raw, list):
                raise ValueError(f"{source}:{line_number}: predictions must be a list")
            predictions = tuple(
                _parse_prediction(item, f"{source}:{line_number}:predictions[{index}]")
                for index, item in enumerate(predictions_raw)
            )
            frames[key] = ObservationFrame(key, width, height, predictions)
    if not frames:
        raise ValueError(f"{source}: no observation frames found")
    return frames


def _load_result(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    with source.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if data.get("schema_version") != 1:
        raise ValueError(f"{source}: unsupported qa_benchmark schema")
    if not isinstance(data.get("metrics"), dict):
        raise ValueError(f"{source}: missing metrics")
    return data


def _result_selection_hashes(result: dict[str, Any]) -> set[str]:
    hashes: set[str] = set()
    containers = [
        result,
        result.get("selection"),
        result.get("frame_manifest"),
        result.get("benchmark_selection"),
    ]
    keys = {
        "selected_frame_ids_sha256",
        "frame_ids_sha256",
        "frame_manifest_sha256",
        "manifest_sha256",
    }
    for container in containers:
        if not isinstance(container, dict):
            continue
        for key in keys:
            value = container.get(key)
            if isinstance(value, str) and value:
                hashes.add(value.lower())
    return hashes


def _prove_selection(
    result: dict[str, Any],
    manifest: ManifestSelection,
    all_gt_keys: set[FrameKey],
    observations: dict[FrameKey, ObservationFrame] | None,
) -> str:
    selected = set(manifest.frames)
    if observations is not None:
        observed = set(observations)
        if observed != selected:
            missing = sorted(item.frame_id for item in selected - observed)
            extra = sorted(item.frame_id for item in observed - selected)
            raise ValueError(
                "observation frames do not exactly match manifest"
                f" (missing={missing[:5]}, extra={extra[:5]})"
            )
    if selected == all_gt_keys:
        return "manifest_covers_all_ground_truth"
    hashes = _result_selection_hashes(result)
    if manifest.frame_ids_sha256.lower() in hashes or manifest.file_sha256.lower() in hashes:
        return "result_selection_hash"
    raise ValueError(
        "cannot prove qa_benchmark result was scored on exactly the supplied manifest; "
        "require result selection-hash metadata unless the manifest covers all ground-truth frames"
    )


def _frame_size(observation: ObservationFrame | None) -> tuple[int, int] | None:
    if observation is None or observation.width is None or observation.height is None:
        return None
    return observation.width, observation.height


def _touches_edge(
    bbox: tuple[float, float, float, float],
    frame_size: tuple[int, int] | None,
) -> bool:
    if frame_size is None:
        return False
    width, height = frame_size
    margin_x = max(4.0, width * 0.02)
    margin_y = max(4.0, height * 0.02)
    return (
        bbox[0] <= margin_x
        or bbox[1] <= margin_y
        or bbox[2] >= width - margin_x
        or bbox[3] >= height - margin_y
    )


def _extreme_scale_or_aspect(
    bbox: tuple[float, float, float, float],
    frame_size: tuple[int, int] | None,
) -> bool:
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    aspect = width / height
    if aspect < 0.12 or aspect > 1.25:
        return True
    if frame_size is None:
        return False
    frame_width, frame_height = frame_size
    area_fraction = (width * height) / float(frame_width * frame_height)
    return area_fraction < 0.00015 or area_fraction > 0.45


def _metadata_terms(frame: GroundTruthFrame, obj: GroundTruthObject | None) -> set[str]:
    values = {_normalize_term(value) for value in frame.tags}
    if obj is not None:
        values.update(_normalize_term(value) for value in obj.attributes)
    return values


def _best_iou(
    obj: GroundTruthObject,
    predictions: Iterable[PredictedObject],
) -> tuple[float, int | None]:
    from .tracker import bbox_iou

    best = 0.0
    best_index: int | None = None
    for index, pred in enumerate(predictions):
        if pred.label != obj.label:
            continue
        overlap = bbox_iou(obj.bbox, pred.bbox)
        if overlap > best:
            best = overlap
            best_index = index
    return best, best_index


def _classify_false_negative(
    frame: GroundTruthFrame,
    obj: GroundTruthObject,
    observation: ObservationFrame | None,
    match_iou: float,
) -> tuple[tuple[str, ...], dict[str, Any], float]:
    categories: set[str] = set()
    terms = _metadata_terms(frame, obj)
    box_width = obj.bbox[2] - obj.bbox[0]
    box_height = obj.bbox[3] - obj.bbox[1]
    frame_size = _frame_size(observation)

    if obj.label == "person" and (box_height < 48.0 or terms & _SMALL_TERMS):
        categories.add("missed_small_person")
    if terms & _LOW_LIGHT_TERMS:
        categories.add("low_contrast_night_darkness")
    if terms & _OCCLUSION_TERMS:
        categories.add("occlusion_crowd")
    if terms & _EDGE_TERMS or _touches_edge(obj.bbox, frame_size):
        categories.add("edge_of_frame")
    if terms & _SCALE_TERMS or _extreme_scale_or_aspect(obj.bbox, frame_size):
        categories.add("scale_extreme_aspect")

    best_iou = None
    best_index = None
    if observation is not None:
        best_iou, best_index = _best_iou(obj, observation.predictions)
        if LOCALIZATION_MIN_IOU <= best_iou < match_iou:
            categories.add("localization_iou_miss")

    if not categories:
        categories.add("unknown")
    severity = 100.0 + max(0.0, 48.0 - box_height)
    if "localization_iou_miss" in categories and best_iou is not None:
        severity += best_iou * 10.0
    evidence = {
        "bbox": list(obj.bbox),
        "box_width": box_width,
        "box_height": box_height,
        "tags": list(frame.tags),
        "attributes": list(obj.attributes),
        "best_same_label_iou": best_iou,
        "best_prediction_index": best_index,
    }
    return tuple(name for name in TAXONOMY if name in categories), evidence, severity


def _classify_false_positive(
    frame: GroundTruthFrame,
    prediction: PredictedObject,
    pred_index: int,
    truth: list[tuple[int, GroundTruthObject]],
    matches: dict[int, int],
    observation: ObservationFrame,
    match_iou: float,
) -> tuple[tuple[str, ...], dict[str, Any], float]:
    from .tracker import bbox_iou

    categories: set[str] = set()
    frame_size = _frame_size(observation)
    same_label = [(truth_index, obj) for truth_index, obj in truth if not obj.ignore and obj.label == prediction.label]
    overlaps = [(bbox_iou(obj.bbox, prediction.bbox), truth_index) for truth_index, obj in same_label]
    overlaps.sort(reverse=True)
    best_iou = overlaps[0][0] if overlaps else 0.0
    best_truth_index = overlaps[0][1] if overlaps else None

    duplicate = False
    if best_truth_index is not None and best_iou >= LOCALIZATION_MIN_IOU:
        matched_pred = matches.get(best_truth_index)
        duplicate = matched_pred is not None and matched_pred != pred_index
    if duplicate:
        categories.add("duplicate_fragmented_detection")
    elif LOCALIZATION_MIN_IOU <= best_iou < match_iou:
        categories.add("localization_iou_miss")
    elif best_iou < BACKGROUND_MAX_IOU:
        categories.add("background_false_positive")

    if _touches_edge(prediction.bbox, frame_size):
        categories.add("edge_of_frame")
    if _extreme_scale_or_aspect(prediction.bbox, frame_size):
        categories.add("scale_extreme_aspect")
    if not categories:
        categories.add("unknown")
    evidence = {
        "bbox": list(prediction.bbox),
        "score": prediction.score,
        "best_same_label_iou": best_iou,
        "best_truth_index": best_truth_index,
    }
    return tuple(name for name in TAXONOMY if name in categories), evidence, 50.0 + prediction.score * 10.0


def _selected_ground_truth(
    frames: list[GroundTruthFrame],
    manifest: ManifestSelection,
) -> tuple[list[GroundTruthFrame], set[FrameKey]]:
    by_key = {FrameKey(frame.video, frame.frame): frame for frame in frames}
    missing = [item.frame_id for item in manifest.frames if item not in by_key]
    if missing:
        raise ValueError(f"manifest contains frames absent from ground truth: {missing[:10]}")
    return [by_key[key] for key in manifest.frames], set(by_key)


def _analyze_run(
    result: dict[str, Any],
    selected_frames: list[GroundTruthFrame],
    manifest: ManifestSelection,
    all_gt_keys: set[FrameKey],
    observations: dict[FrameKey, ObservationFrame] | None,
    *,
    label: str,
    top_k: int,
) -> dict[str, Any]:
    metrics = result["metrics"]
    match_iou = float(result.get("evaluation", {}).get("match_iou", 0.5))
    result_label = result.get("evaluation", {}).get("label", label)
    if result_label != label:
        raise ValueError(f"result label {result_label!r} does not match requested label {label!r}")
    proof = _prove_selection(result, manifest, all_gt_keys, observations)
    matched_keys = set(metrics.get("matched_ground_truth", []))
    missed_keys = set(metrics.get("missed_ground_truth", []))
    events: list[FailureEvent] = []
    selected_object_keys: set[str] = set()

    for frame in selected_frames:
        key = FrameKey(frame.video, frame.frame)
        observation = observations.get(key) if observations is not None else None
        truth = [(index, obj) for index, obj in enumerate(frame.objects) if obj.label == label]
        valid_truth = [(index, obj) for index, obj in truth if not obj.ignore]
        for truth_index, obj in valid_truth:
            object_key = _truth_key(frame, truth_index, obj)
            selected_object_keys.add(object_key)
            in_matched = object_key in matched_keys
            in_missed = object_key in missed_keys
            if in_matched == in_missed:
                raise ValueError(
                    f"result does not uniquely classify selected GT object {object_key}; "
                    "expected exactly one of matched_ground_truth/missed_ground_truth"
                )
            if in_missed:
                categories, evidence, severity = _classify_false_negative(
                    frame, obj, observation, match_iou
                )
                events.append(
                    FailureEvent(
                        event_id=object_key,
                        kind="false_negative",
                        frame_id=key.frame_id,
                        categories=categories,
                        severity=severity,
                        evidence=evidence,
                    )
                )

        if observation is None:
            continue
        predictions = [pred for pred in observation.predictions if pred.label == label]
        matches, used_predictions = _match_objects(truth, predictions, match_iou)
        geometry_matched = {
            _truth_key(frame, truth_index, frame.objects[truth_index]) for truth_index in matches
        }
        expected_matched = matched_keys & {
            _truth_key(frame, truth_index, obj) for truth_index, obj in valid_truth
        }
        if geometry_matched != expected_matched:
            raise ValueError(
                f"observation geometry disagrees with qa_benchmark result on {key.frame_id}"
            )
        for pred_index, prediction in enumerate(predictions):
            if pred_index in used_predictions:
                continue
            categories, evidence, severity = _classify_false_positive(
                frame,
                prediction,
                pred_index,
                truth,
                matches,
                observation,
                match_iou,
            )
            events.append(
                FailureEvent(
                    event_id=f"{key.frame_id}#prediction:{pred_index}",
                    kind="false_positive",
                    frame_id=key.frame_id,
                    categories=categories,
                    severity=severity,
                    evidence=evidence,
                )
            )

    classified_keys = matched_keys | missed_keys
    if classified_keys != selected_object_keys:
        extra = sorted(classified_keys - selected_object_keys)
        missing = sorted(selected_object_keys - classified_keys)
        raise ValueError(
            "qa_benchmark matched/missed GT IDs do not exactly match the selected manifest "
            f"(missing={missing[:5]}, extra={extra[:5]})"
        )
    selected_fn = sum(event.kind == "false_negative" for event in events)
    expected_fn = len(missed_keys)
    if selected_fn != expected_fn:
        raise ValueError("internal false-negative accounting mismatch")

    result_fp = int(metrics.get("false_positives", 0))
    detailed_fp = sum(event.kind == "false_positive" for event in events)
    if observations is not None:
        if detailed_fp != result_fp:
            raise ValueError(
                f"observation false positives ({detailed_fp}) disagree with qa_benchmark result ({result_fp})"
            )
    else:
        for index in range(result_fp):
            events.append(
                FailureEvent(
                    event_id=f"summary:false_positive:{index + 1}",
                    kind="false_positive",
                    frame_id=None,
                    categories=("unknown",),
                    severity=0.0,
                    evidence={"reason": "qa_benchmark result has no per-frame prediction geometry"},
                )
            )

    category_events: dict[str, list[FailureEvent]] = defaultdict(list)
    for event in events:
        for category in event.categories:
            category_events[category].append(event)

    total = len(events)
    categories: dict[str, Any] = {}
    for category in TAXONOMY:
        items = sorted(category_events.get(category, []), key=lambda item: (-item.severity, item.event_id))
        categories[category] = {
            "count": len(items),
            "rate": len(items) / total if total else 0.0,
            "false_negative_count": sum(item.kind == "false_negative" for item in items),
            "false_positive_count": sum(item.kind == "false_positive" for item in items),
            "representatives": [item.event_id for item in items[:top_k]],
        }

    serialized_events = [
        {
            "event_id": event.event_id,
            "kind": event.kind,
            "frame_id": event.frame_id,
            "categories": list(event.categories),
            "severity": event.severity,
            "evidence": event.evidence,
        }
        for event in sorted(events, key=lambda item: (item.kind, item.event_id))
    ]
    return {
        "run_name": result.get("run_name"),
        "revision": result.get("revision"),
        "selection_proof": proof,
        "match_iou": match_iou,
        "failure_event_count": total,
        "false_negative_count": selected_fn,
        "false_positive_count": result_fp,
        "taxonomy_is_multilabel": True,
        "taxonomy": categories,
        "events": serialized_events,
    }


def _recovered_in_category(control: dict[str, Any], candidate: dict[str, Any], category: str) -> list[str]:
    control_events = {
        event["event_id"]
        for event in control["events"]
        if event["kind"] == "false_negative" and category in event["categories"]
    }
    candidate_missed = {
        event["event_id"] for event in candidate["events"] if event["kind"] == "false_negative"
    }
    return sorted(control_events - candidate_missed)


def _comparison(control: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    control_fn = {
        event["event_id"] for event in control["events"] if event["kind"] == "false_negative"
    }
    candidate_fn = {
        event["event_id"] for event in candidate["events"] if event["kind"] == "false_negative"
    }
    recovered = sorted(control_fn - candidate_fn)
    new_misses = sorted(candidate_fn - control_fn)
    category_delta = {
        category: candidate["taxonomy"][category]["count"] - control["taxonomy"][category]["count"]
        for category in TAXONOMY
    }
    lowlight_recovered = _recovered_in_category(
        control, candidate, "low_contrast_night_darkness"
    )
    small_recovered = _recovered_in_category(control, candidate, "missed_small_person")
    background_delta = category_delta["background_false_positive"]
    guidance: list[str] = []

    if background_delta < 0 and not lowlight_recovered:
        guidance.append(
            f"candidate reduces background false-positive events by {-background_delta} "
            "but recovers 0 low-contrast/night ground-truth persons on the identical selected frames"
        )
    elif candidate["false_positive_count"] < control["false_positive_count"] and not lowlight_recovered:
        guidance.append(
            f"candidate reduces benchmark false positives by "
            f"{control['false_positive_count'] - candidate['false_positive_count']} but recovers "
            "0 low-contrast/night ground-truth persons; per-frame geometry is unavailable to call "
            "those reductions background false positives"
        )
    if small_recovered:
        guidance.append(
            f"candidate recovers {len(small_recovered)} ground-truth persons carrying the "
            "missed-small-person heuristic"
        )
    if new_misses:
        guidance.append(
            f"candidate introduces {len(new_misses)} new missed ground-truth objects on the "
            "same selected frames"
        )
    if not guidance:
        guidance.append(
            "candidate changes no classified failure category in a way that supports a more "
            "specific error-mechanism claim"
        )

    return {
        "recovered_ground_truth": recovered,
        "new_missed_ground_truth": new_misses,
        "category_count_delta": category_delta,
        "false_negative_delta": candidate["false_negative_count"] - control["false_negative_count"],
        "false_positive_delta": candidate["false_positive_count"] - control["false_positive_count"],
        "guidance": guidance,
    }


def build_error_report(
    *,
    ground_truth_path: str | Path,
    manifest_path: str | Path,
    control_result_path: str | Path,
    candidate_result_path: str | Path | None = None,
    control_observations_path: str | Path | None = None,
    candidate_observations_path: str | Path | None = None,
    label: str = "person",
    top_k: int = DEFAULT_TOP_K,
) -> dict[str, Any]:
    if top_k < 1 or top_k > MAX_TOP_K:
        raise ValueError(f"top_k must be in [1, {MAX_TOP_K}]")
    manifest = load_frame_manifest(manifest_path)
    all_frames = load_ground_truth(ground_truth_path)
    selected_frames, all_gt_keys = _selected_ground_truth(all_frames, manifest)
    control_result = _load_result(control_result_path)
    expected_gt_sha = ground_truth_sha256(ground_truth_path)
    if control_result.get("ground_truth_sha256") != expected_gt_sha:
        raise ValueError("control result ground-truth SHA-256 does not match supplied JSONL")
    control_observations = (
        load_observations(control_observations_path) if control_observations_path else None
    )
    control = _analyze_run(
        control_result,
        selected_frames,
        manifest,
        all_gt_keys,
        control_observations,
        label=label,
        top_k=top_k,
    )

    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "label": label,
        "frame_count": len(manifest.frames),
        "manifest_sha256": manifest.file_sha256,
        "selected_frame_ids_sha256": manifest.frame_ids_sha256,
        "ground_truth_sha256": expected_gt_sha,
        "input_hashes": {
            "control_result_sha256": _sha256_file(control_result_path),
            "control_observations_sha256": (
                _sha256_file(control_observations_path) if control_observations_path else None
            ),
        },
        "control": control,
        "candidate": None,
        "comparison": None,
        "heuristics": {
            "small_person_height_lt_px": 48.0,
            "localization_iou_min": LOCALIZATION_MIN_IOU,
            "background_fp_max_iou": BACKGROUND_MAX_IOU,
            "edge_margin_fraction": 0.02,
            "edge_margin_min_px": 4.0,
            "aspect_ratio_min": 0.12,
            "aspect_ratio_max": 1.25,
            "area_fraction_min": 0.00015,
            "area_fraction_max": 0.45,
        },
        "limits": {
            "top_k_representatives_per_category": top_k,
            "contact_sheet_generation": "not performed",
        },
    }

    if candidate_result_path is not None:
        candidate_result = _load_result(candidate_result_path)
        if candidate_result.get("ground_truth_sha256") != expected_gt_sha:
            raise ValueError("candidate result ground-truth SHA-256 does not match supplied JSONL")
        for key in ("label", "match_iou"):
            if candidate_result.get("evaluation", {}).get(key) != control_result.get("evaluation", {}).get(key):
                raise ValueError(f"candidate/control evaluation mismatch: {key}")
        candidate_observations = (
            load_observations(candidate_observations_path) if candidate_observations_path else None
        )
        candidate = _analyze_run(
            candidate_result,
            selected_frames,
            manifest,
            all_gt_keys,
            candidate_observations,
            label=label,
            top_k=top_k,
        )
        payload["candidate"] = candidate
        payload["comparison"] = _comparison(control, candidate)
        payload["input_hashes"]["candidate_result_sha256"] = _sha256_file(candidate_result_path)
        payload["input_hashes"]["candidate_observations_sha256"] = (
            _sha256_file(candidate_observations_path) if candidate_observations_path else None
        )

    payload["report_sha256"] = _canonical_sha256(payload)
    return payload


def _write_report(path: str | Path, payload: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Deterministic offline QA failure taxonomy for frozen benchmark frames"
    )
    parser.add_argument("--ground-truth", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--control-result", required=True)
    parser.add_argument("--candidate-result")
    parser.add_argument("--control-observations")
    parser.add_argument("--candidate-observations")
    parser.add_argument("--label", default="person")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--output", required=True)
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    report = build_error_report(
        ground_truth_path=args.ground_truth,
        manifest_path=args.manifest,
        control_result_path=args.control_result,
        candidate_result_path=args.candidate_result,
        control_observations_path=args.control_observations,
        candidate_observations_path=args.candidate_observations,
        label=args.label,
        top_k=args.top_k,
    )
    _write_report(args.output, report)
    summary = {
        "frame_count": report["frame_count"],
        "report_sha256": report["report_sha256"],
        "control_failures": report["control"]["failure_event_count"],
        "candidate_failures": (
            report["candidate"]["failure_event_count"] if report["candidate"] is not None else None
        ),
        "guidance": report["comparison"]["guidance"] if report["comparison"] else [],
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
