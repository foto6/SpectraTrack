from __future__ import annotations

import argparse
import hashlib
import json
from math import hypot
from pathlib import Path
import time
from typing import Any, Callable, Iterable, Mapping

import cv2

from ..detector import YoloOnnxDetector
from ..integrity import sha256_file
from ..qa_benchmark import GroundTruthFrame, GroundTruthObject, load_ground_truth
from .vnext_detection_corpus import _source_image_path, _timestamp_s
from .vnext_detection_fusion import (
    FusionCandidate,
    FusionConfig,
    ResearchFrame,
    _match_gt,
    bbox_iou,
    collect_prefusion_people_recall,
    evaluate_fusion,
    fuse_candidates,
)
from .vnext_detection_hypothesis_lock import (
    EXPECTED_CANDIDATE_ID,
    EXPECTED_LOCK_DIGEST,
    load_lock,
    validate_lock,
)
from .vnext_detector_backend_lab import _load_benchmark_source_records

RESULT_SCHEMA = "spectratrack-a1-r2-result-v1"
RUNTIME_SCHEMA = "spectratrack-a1-r2-runtime-v1"
R2_REVISION = "nightowls-public-smoke400-r2"
R2_FRAME_COUNT = 400
R2_CORPUS_SHA256 = "9d145b4dda780052388f3b663203c519ded48f61457adef14654f84fb5549eff"
R2_GT_SHA256 = "75eba2c9dc3b36d0a2389bfbb080a709621ef690680ae655d85d8732e7bc6097"
R2_MANIFEST_SHA256 = "9270d46c2776aa531e1b979a1a7ebc16eaed0f6483b2c1095da834af383c4e83"
R2_PROOF_SHA256 = "df11e9dc6ca1f63019cba071ed82420de27c706b41d1d3b46df880bb3ce50faf"
A6_POLICY_COMMIT = "1a63c0a981c21e57dbe679e388046eaaaa0d5577"
A6_POLICY_CI = 36317780050


def _canonical_digest(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _json_object(path: str | Path) -> dict[str, Any]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"expected JSON object: {path}")
    return data


def _find_key(data: Any, key: str) -> list[Any]:
    found: list[Any] = []
    if isinstance(data, dict):
        for name, value in data.items():
            if name == key:
                found.append(value)
            found.extend(_find_key(value, key))
    elif isinstance(data, list):
        for value in data:
            found.extend(_find_key(value, key))
    return found


def _require_any_value(data: Mapping[str, Any], key: str, expected: Any) -> None:
    values = _find_key(data, key)
    if expected not in values:
        raise ValueError(f"{key} does not contain expected frozen value {expected!r}")


def verify_frozen_inputs(
    *,
    lock_path: str | Path,
    ground_truth_path: str | Path,
    manifest_path: str | Path,
    proof_path: str | Path,
) -> dict[str, Any]:
    lock = load_lock(lock_path)
    lock_validation = validate_lock(lock)
    if lock_validation["lock_digest"] != EXPECTED_LOCK_DIGEST:
        raise ValueError("lock digest mismatch")

    actual = {
        "ground_truth_sha256": sha256_file(ground_truth_path),
        "manifest_sha256": sha256_file(manifest_path),
        "selection_proof_sha256": sha256_file(proof_path),
    }
    expected = {
        "ground_truth_sha256": R2_GT_SHA256,
        "manifest_sha256": R2_MANIFEST_SHA256,
        "selection_proof_sha256": R2_PROOF_SHA256,
    }
    for name, value in expected.items():
        if actual[name] != value:
            raise ValueError(
                f"R2 preflight mismatch for {name}: "
                f"expected {value}, got {actual[name]}"
            )

    manifest = _json_object(manifest_path)
    proof = _json_object(proof_path)
    _require_any_value(manifest, "revision", R2_REVISION)
    _require_any_value(manifest, "corpus_sha256", R2_CORPUS_SHA256)
    _require_any_value(proof, "revision", R2_REVISION)

    return {
        "candidate_id": EXPECTED_CANDIDATE_ID,
        "lock_digest": EXPECTED_LOCK_DIGEST,
        "revision": R2_REVISION,
        "frame_count": R2_FRAME_COUNT,
        "corpus_sha256": R2_CORPUS_SHA256,
        **actual,
    }


def _center(box: tuple[float, float, float, float]) -> tuple[float, float]:
    return ((box[0] + box[2]) * 0.5, (box[1] + box[3]) * 0.5)


def _height(box: tuple[float, float, float, float]) -> float:
    return max(0.0, box[3] - box[1])


def _pair_related(left: FusionCandidate, right: FusionCandidate) -> bool:
    if left.label != right.label:
        return False
    if bbox_iou(left.bbox, right.bbox) >= 0.15:
        return True
    left_height = _height(left.bbox)
    right_height = _height(right.bbox)
    if min(left_height, right_height) <= 0.0:
        return False
    lx, ly = _center(left.bbox)
    rx, ry = _center(right.bbox)
    distance = hypot(lx - rx, ly - ry)
    return distance / min(left_height, right_height) <= 0.75


def _canonical_group_signature(group: Iterable[FusionCandidate]) -> str:
    records = [
        (
            item.source_id,
            tuple(round(float(value), 6) for value in item.bbox),
            round(float(item.score), 8),
        )
        for item in group
    ]
    return json.dumps(sorted(records), separators=(",", ":"))


def select_trigger_group(
    candidates: Iterable[FusionCandidate],
) -> tuple[FusionCandidate, ...] | None:
    people = [item for item in candidates if item.label.lower() == "person"]
    if len(people) < 2:
        return None

    adjacency: dict[int, set[int]] = {index: set() for index in range(len(people))}
    for left in range(len(people)):
        for right in range(left + 1, len(people)):
            if _pair_related(people[left], people[right]):
                adjacency[left].add(right)
                adjacency[right].add(left)

    groups: list[tuple[FusionCandidate, ...]] = []
    remaining = set(adjacency)
    while remaining:
        start = min(remaining)
        stack = [start]
        component: list[int] = []
        while stack:
            index = stack.pop()
            if index not in remaining:
                continue
            remaining.remove(index)
            component.append(index)
            stack.extend(sorted(adjacency[index] & remaining, reverse=True))
        group = tuple(people[index] for index in sorted(component))
        if len(group) < 2:
            continue
        if len({item.source_id for item in group}) < 2:
            continue
        groups.append(group)

    if not groups:
        return None

    def rank(group: tuple[FusionCandidate, ...]) -> tuple[Any, ...]:
        heights = sorted(_height(item.bbox) for item in group)
        midpoint = len(heights) // 2
        if len(heights) % 2:
            median_height = heights[midpoint]
        else:
            median_height = (heights[midpoint - 1] + heights[midpoint]) * 0.5
        return (
            median_height,
            -len({item.source_id for item in group}),
            -len(group),
            _canonical_group_signature(group),
        )

    return min(groups, key=rank)


def _union_box(
    group: Iterable[FusionCandidate],
) -> tuple[float, float, float, float]:
    items = list(group)
    return (
        min(item.bbox[0] for item in items),
        min(item.bbox[1] for item in items),
        max(item.bbox[2] for item in items),
        max(item.bbox[3] for item in items),
    )


def microtile_region(
    group: Iterable[FusionCandidate],
    *,
    frame_width: int,
    frame_height: int,
) -> tuple[int, int, int, int] | None:
    if frame_width < 512 or frame_height < 512:
        return None
    union = _union_box(group)
    cx, cy = _center(union)
    x1 = int(round(cx - 256.0))
    y1 = int(round(cy - 256.0))
    x1 = max(0, min(x1, frame_width - 512))
    y1 = max(0, min(y1, frame_height - 512))
    return (x1, y1, x1 + 512, y1 + 512)


def _expanded_trigger_box(
    group: Iterable[FusionCandidate],
) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = _union_box(group)
    width = max(0.0, x2 - x1)
    height = max(0.0, y2 - y1)
    return (
        x1 - 0.15 * width,
        y1 - 0.15 * height,
        x2 + 0.15 * width,
        y2 + 0.15 * height,
    )


def _inside(box: tuple[float, float, float, float], region: tuple[float, ...]) -> bool:
    cx, cy = _center(box)
    return region[0] <= cx <= region[2] and region[1] <= cy <= region[3]


def add_locked_microtile(
    detector: YoloOnnxDetector,
    frame_bgr,
    baseline: Iterable[FusionCandidate],
) -> tuple[list[FusionCandidate], dict[str, Any]]:
    candidates = list(baseline)
    group = select_trigger_group(candidates)
    if group is None:
        return candidates, {
            "activated": False,
            "extra_onnx_calls": 0,
            "admitted_detections": 0,
        }

    height, width = frame_bgr.shape[:2]
    region = microtile_region(group, frame_width=width, frame_height=height)
    if region is None:
        return candidates, {
            "activated": False,
            "extra_onnx_calls": 0,
            "admitted_detections": 0,
        }

    thresholds = dict(detector.class_thresholds)
    thresholds["person"] = 0.12
    person_ids = {
        index
        for index, label in enumerate(detector.labels)
        if label.lower() == "person"
    }
    x1, y1, x2, y2 = region
    crop = frame_bgr[y1:y2, x1:x2]
    before_calls = int(detector.last_inference_calls)
    detections = detector._detect_once(crop, thresholds)
    extra_calls = int(detector.last_inference_calls) - before_calls
    if extra_calls != 1:
        raise RuntimeError(f"locked microtile must use exactly one ONNX call, got {extra_calls}")

    trigger_box = _expanded_trigger_box(group)
    admitted = 0
    for detection in detections:
        if detection.class_id not in person_ids:
            continue
        bx1, by1, bx2, by2 = detection.bbox
        global_box = (
            float(bx1 + x1),
            float(by1 + y1),
            float(bx2 + x1),
            float(by2 + y1),
        )
        if not _inside(global_box, trigger_box):
            continue
        candidates.append(
            FusionCandidate(
                global_box,
                float(detection.score),
                int(detection.class_id),
                detection.label,
                "microtile",
                "microtile:0",
                region,
            )
        )
        admitted += 1

    return candidates, {
        "activated": True,
        "extra_onnx_calls": 1,
        "admitted_detections": admitted,
        "trigger_group_size": len(group),
        "trigger_distinct_sources": len({item.source_id for item in group}),
    }


def _is_small_person(obj: GroundTruthObject) -> bool:
    return obj.label == "person" and not obj.ignore and _height(obj.bbox) < 48.0


def _is_occlusion_crowd(obj: GroundTruthObject) -> bool:
    if obj.label != "person" or obj.ignore:
        return False
    attributes = {str(value).lower() for value in obj.attributes}
    return any(value.endswith("occluded_true") for value in attributes)


def _cluster_metric(
    frames: Iterable[ResearchFrame],
    *,
    predicate: Callable[[GroundTruthObject], bool],
    config: FusionConfig,
) -> dict[str, Any]:
    total = tp = fn = 0
    for frame in frames:
        fused = fuse_candidates(frame.candidates, "hard-nms", config)
        matches = _match_gt(frame.ground_truth, fused, 0.5)
        for index, obj in enumerate(frame.ground_truth):
            if not predicate(obj):
                continue
            total += 1
            if index in matches:
                tp += 1
            else:
                fn += 1
    return {
        "ground_truth": total,
        "tp": tp,
        "fn": fn,
        "recall": (tp / total if total else None),
    }


def targeted_cluster_metrics(
    frames: Iterable[ResearchFrame],
    *,
    config: FusionConfig,
) -> dict[str, Any]:
    items = list(frames)
    return {
        "small_person_height_lt_48": _cluster_metric(
            items,
            predicate=_is_small_person,
            config=config,
        ),
        "occlusion_crowd_occluded_true": _cluster_metric(
            items,
            predicate=_is_occlusion_crowd,
            config=config,
        ),
    }


def _delta(candidate: Any, control: Any) -> float | None:
    if candidate is None or control is None:
        return None
    return float(candidate) - float(control)


def _relative_increase(candidate: int, control: int) -> float:
    if control == 0:
        return 0.0 if candidate == 0 else float("inf")
    return (candidate - control) / control


def apply_prelocked_gates(
    *,
    control: Mapping[str, Any],
    candidate: Mapping[str, Any],
    control_clusters: Mapping[str, Any],
    candidate_clusters: Mapping[str, Any],
    compute: Mapping[str, Any],
) -> dict[str, Any]:
    deltas = {
        "tp": int(candidate["tp"]) - int(control["tp"]),
        "fp": int(candidate["fp"]) - int(control["fp"]),
        "fn": int(candidate["fn"]) - int(control["fn"]),
        "precision": float(candidate["precision"]) - float(control["precision"]),
        "recall": float(candidate["recall"]) - float(control["recall"]),
        "f1": float(candidate["f1"]) - float(control["f1"]),
        "bbox_localization_iou": _delta(
            candidate.get("bbox_localization_iou"),
            control.get("bbox_localization_iou"),
        ),
        "center_error_gt_diag_ratio": _delta(
            candidate.get("center_error_gt_diag_ratio"),
            control.get("center_error_gt_diag_ratio"),
        ),
        "prefusion_duplicate_pressure": (
            int(candidate["duplicate_count_before_fusion"])
            - int(control["duplicate_count_before_fusion"])
        ),
    }
    fp_relative = _relative_increase(int(candidate["fp"]), int(control["fp"]))
    duplicate_relative = _relative_increase(
        int(candidate["duplicate_count_before_fusion"]),
        int(control["duplicate_count_before_fusion"]),
    )

    hard_reject_reasons: list[str] = []
    if deltas["f1"] <= -0.005:
        hard_reject_reasons.append("f1_delta_at_most_-0.005")
    if deltas["recall"] <= -0.005:
        hard_reject_reasons.append("recall_delta_at_most_-0.005")
    if deltas["precision"] <= -0.01:
        hard_reject_reasons.append("precision_delta_at_most_-0.01")
    if deltas["fp"] >= 10 and fp_relative >= 0.15:
        hard_reject_reasons.append("false_positive_increase_gate")
    if (
        deltas["prefusion_duplicate_pressure"] >= 20
        and duplicate_relative >= 0.5
    ):
        hard_reject_reasons.append("prefusion_duplicate_increase_gate")
    if int(compute["max_extra_onnx_calls_observed_per_frame"]) > 1:
        hard_reject_reasons.append("compute_budget_violation")

    small_control = control_clusters["small_person_height_lt_48"]
    small_candidate = candidate_clusters["small_person_height_lt_48"]
    crowd_control = control_clusters["occlusion_crowd_occluded_true"]
    crowd_candidate = candidate_clusters["occlusion_crowd_occluded_true"]
    small_fn_delta = int(small_candidate["fn"]) - int(small_control["fn"])
    crowd_fn_delta = int(crowd_candidate["fn"]) - int(crowd_control["fn"])

    bbox_delta = deltas["bbox_localization_iou"]
    center_delta = deltas["center_error_gt_diag_ratio"]
    promote_checks = {
        "compute_budget_must_pass": (
            int(compute["max_extra_onnx_calls_observed_per_frame"]) <= 1
        ),
        "bbox_localization_iou": (
            bbox_delta is None or bbox_delta >= -0.01
        ),
        "center_error_gt_diag_ratio": (
            center_delta is None or center_delta <= 0.02
        ),
        "false_positive_relative_increase": fp_relative <= 0.05,
        "precision": deltas["precision"] >= -0.005,
        "prefusion_duplicate_relative_increase": duplicate_relative <= 0.25,
        "f1": deltas["f1"] >= 0.005,
        "recall": deltas["recall"] >= 0.01,
        "small_person_fn": small_fn_delta <= 0,
        "occlusion_crowd_fn": (
            True if int(crowd_control["fn"]) == 0 else crowd_fn_delta <= -1
        ),
    }

    if hard_reject_reasons:
        decision = "REJECT_R2"
    elif all(promote_checks.values()):
        decision = "PROMOTE_TO_HOLDOUT4200_REQUEST"
    else:
        decision = "AMBIGUOUS_R2"

    return {
        "decision": decision,
        "deltas_vs_control": deltas,
        "relative_increases": {
            "false_positive": fp_relative,
            "prefusion_duplicate_pressure": duplicate_relative,
        },
        "targeted_cluster_deltas": {
            "small_person_fn": small_fn_delta,
            "occlusion_crowd_fn": crowd_fn_delta,
        },
        "hard_reject_reasons": hard_reject_reasons,
        "promote_checks": promote_checks,
    }


def _stage_delta(
    after: Mapping[str, float],
    before: Mapping[str, float],
) -> dict[str, float]:
    return {
        key: float(after.get(key, 0.0)) - float(before.get(key, 0.0))
        for key in sorted(set(after) | set(before))
    }


def run_r2(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    provenance = verify_frozen_inputs(
        lock_path=args.lock,
        ground_truth_path=args.ground_truth,
        manifest_path=args.manifest,
        proof_path=args.selection_proof,
    )

    annotations = load_ground_truth(args.ground_truth)
    if len(annotations) != R2_FRAME_COUNT:
        raise ValueError(
            f"R2 ground truth must contain {R2_FRAME_COUNT} frames, "
            f"got {len(annotations)}"
        )
    source_records = _load_benchmark_source_records(args.ground_truth)
    detector = YoloOnnxDetector(
        args.model,
        input_size=960,
        conf_threshold=0.35,
        iou_threshold=0.45,
        prefer_gpu=not args.cpu,
    )
    model_sha = sha256_file(args.model)
    lock = load_lock(args.lock)
    candidate_config = {
        "candidate_id": lock["candidate_id"],
        "control_contract": lock["control_contract"],
        "algorithm": lock["algorithm"],
        "compute_budget": lock["compute_budget"],
    }
    candidate_config_digest = _canonical_digest(candidate_config)

    control_frames: list[ResearchFrame] = []
    candidate_frames: list[ResearchFrame] = []
    control_calls = 0
    extra_calls = 0
    activations = 0
    admitted = 0
    max_extra_calls = 0
    control_stage_ms: dict[str, float] = {}
    extra_stage_ms: dict[str, float] = {}
    extra_wall_s = 0.0
    started = time.perf_counter()

    ordered = sorted(annotations, key=lambda item: (item.video, item.frame))
    for annotation in ordered:
        key = (annotation.video, annotation.frame)
        record = source_records.get(key)
        if record is None:
            raise ValueError(
                f"missing source record for {annotation.video}#{annotation.frame}"
            )
        image_path, _ = _source_image_path(args.video_root, record)
        frame_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if frame_bgr is None:
            raise RuntimeError(f"cannot read R2 benchmark image: {image_path}")

        baseline = collect_prefusion_people_recall(
            detector,
            frame_bgr,
            person_threshold=0.12,
            tile_size=640,
            tile_overlap=0.20,
        )
        frame_control_calls = int(detector.last_inference_calls)
        baseline_stage = dict(detector.last_stage_ms)
        control_calls += frame_control_calls
        for name, value in baseline_stage.items():
            control_stage_ms[name] = control_stage_ms.get(name, 0.0) + float(value)

        height, width = frame_bgr.shape[:2]
        common = {
            "video": annotation.video,
            "frame": annotation.frame,
            "timestamp_s": _timestamp_s(record, annotation.frame),
            "width": width,
            "height": height,
            "ground_truth": annotation.objects,
            "tags": annotation.tags,
        }
        control_frames.append(
            ResearchFrame(candidates=tuple(baseline), **common)
        )

        microtile_started = time.perf_counter()
        augmented, microtile = add_locked_microtile(
            detector,
            frame_bgr,
            baseline,
        )
        microtile_elapsed = time.perf_counter() - microtile_started
        frame_extra_calls = int(microtile["extra_onnx_calls"])
        extra_calls += frame_extra_calls
        max_extra_calls = max(max_extra_calls, frame_extra_calls)
        if microtile["activated"]:
            activations += 1
            extra_wall_s += microtile_elapsed
        admitted += int(microtile["admitted_detections"])
        for name, value in _stage_delta(
            detector.last_stage_ms,
            baseline_stage,
        ).items():
            extra_stage_ms[name] = extra_stage_ms.get(name, 0.0) + value
        candidate_frames.append(
            ResearchFrame(candidates=tuple(augmented), **common)
        )

    config = FusionConfig(iou_threshold=0.55)
    control_metrics = evaluate_fusion(
        control_frames,
        method="hard-nms",
        config=config,
        match_iou=0.5,
    )
    candidate_metrics = evaluate_fusion(
        candidate_frames,
        method="hard-nms",
        config=config,
        match_iou=0.5,
    )
    control_clusters = targeted_cluster_metrics(control_frames, config=config)
    candidate_clusters = targeted_cluster_metrics(candidate_frames, config=config)
    compute = {
        "control_onnx_calls": control_calls,
        "extra_onnx_calls": extra_calls,
        "candidate_represented_onnx_calls": control_calls + extra_calls,
        "activations": activations,
        "activation_rate": activations / R2_FRAME_COUNT,
        "admitted_microtile_detections": admitted,
        "max_extra_onnx_calls_observed_per_frame": max_extra_calls,
        "locked_max_extra_onnx_calls_per_frame": 1,
    }
    gate_result = apply_prelocked_gates(
        control=control_metrics,
        candidate=candidate_metrics,
        control_clusters=control_clusters,
        candidate_clusters=candidate_clusters,
        compute=compute,
    )

    provenance_payload = {
        **provenance,
        "a6_policy_commit": A6_POLICY_COMMIT,
        "a6_policy_ci": A6_POLICY_CI,
        "model_sha256": model_sha,
        "source_commit": args.source_commit,
        "candidate_config_digest": candidate_config_digest,
    }
    provenance_digest = _canonical_digest(provenance_payload)
    result = {
        "schema": RESULT_SCHEMA,
        "candidate_id": EXPECTED_CANDIDATE_ID,
        "source_commit": args.source_commit,
        "lock_digest": EXPECTED_LOCK_DIGEST,
        "candidate_config_digest": candidate_config_digest,
        "r2": {
            "revision": R2_REVISION,
            "frame_count": R2_FRAME_COUNT,
            "corpus_sha256": R2_CORPUS_SHA256,
            "ground_truth_sha256": R2_GT_SHA256,
            "manifest_sha256": R2_MANIFEST_SHA256,
            "selection_proof_sha256": R2_PROOF_SHA256,
        },
        "a6_stage_policy": {
            "commit": A6_POLICY_COMMIT,
            "ci": A6_POLICY_CI,
            "first_allowed_r2_exposure": True,
            "holdout4200_inspected": False,
            "holdout4200_scored": False,
            "full5000_run": False,
            "retuned_after_r2": False,
        },
        "model_sha256": model_sha,
        "production_control_changed": False,
        "control": {
            "metrics": control_metrics,
            "clusters": control_clusters,
        },
        "candidate": {
            "metrics": candidate_metrics,
            "clusters": candidate_clusters,
        },
        "compute": compute,
        "gates": {
            key: value
            for key, value in gate_result.items()
            if key != "decision"
        },
        "decision": gate_result["decision"],
        "provenance_digest": provenance_digest,
    }
    result["result_digest"] = _canonical_digest(result)

    runtime = {
        "schema": RUNTIME_SCHEMA,
        "source_commit": args.source_commit,
        "result_digest": result["result_digest"],
        "control_stage_ms": control_stage_ms,
        "extra_microtile_stage_ms": extra_stage_ms,
        "extra_microtile_wall_s": extra_wall_s,
        "total_wall_s": time.perf_counter() - started,
    }
    return result, runtime


def validate_result_artifact(result: Mapping[str, Any]) -> dict[str, Any]:
    if result.get("schema") != RESULT_SCHEMA:
        raise ValueError("unsupported R2 result schema")
    if result.get("candidate_id") != EXPECTED_CANDIDATE_ID:
        raise ValueError("R2 result candidate mismatch")
    if result.get("lock_digest") != EXPECTED_LOCK_DIGEST:
        raise ValueError("R2 result lock digest mismatch")
    r2 = result.get("r2")
    if not isinstance(r2, Mapping):
        raise ValueError("R2 result provenance is missing")
    expected = {
        "revision": R2_REVISION,
        "frame_count": R2_FRAME_COUNT,
        "corpus_sha256": R2_CORPUS_SHA256,
        "ground_truth_sha256": R2_GT_SHA256,
        "manifest_sha256": R2_MANIFEST_SHA256,
        "selection_proof_sha256": R2_PROOF_SHA256,
    }
    for key, value in expected.items():
        if r2.get(key) != value:
            raise ValueError(f"R2 result provenance mismatch: {key}")
    stage = result.get("a6_stage_policy")
    if not isinstance(stage, Mapping):
        raise ValueError("A6 stage policy provenance is missing")
    if stage.get("commit") != A6_POLICY_COMMIT:
        raise ValueError("A6 stage policy commit mismatch")
    for field in (
        "holdout4200_inspected",
        "holdout4200_scored",
        "full5000_run",
        "retuned_after_r2",
    ):
        if stage.get(field) is not False:
            raise ValueError(f"forbidden post-R2 state in result: {field}")
    decision = result.get("decision")
    if decision not in {
        "REJECT_R2",
        "PROMOTE_TO_HOLDOUT4200_REQUEST",
        "AMBIGUOUS_R2",
    }:
        raise ValueError("invalid R2 decision")
    recorded_digest = result.get("result_digest")
    payload = dict(result)
    payload.pop("result_digest", None)
    if recorded_digest != _canonical_digest(payload):
        raise ValueError("R2 result digest mismatch")
    return {
        "valid": True,
        "candidate_id": EXPECTED_CANDIDATE_ID,
        "result_digest": recorded_digest,
        "decision": decision,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the frozen A1 blind-R2 candidate without retuning"
    )
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    run.add_argument("--lock", required=True)
    run.add_argument("--ground-truth", required=True)
    run.add_argument("--manifest", required=True)
    run.add_argument("--selection-proof", required=True)
    run.add_argument("--video-root", required=True)
    run.add_argument("--model", required=True)
    run.add_argument("--source-commit", required=True)
    run.add_argument("--output", required=True)
    run.add_argument("--runtime-output")
    run.add_argument("--cpu", action="store_true")
    verify = sub.add_parser("validate-result")
    verify.add_argument("--result", required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.command == "validate-result":
        result = validate_result_artifact(_json_object(args.result))
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0

    result, runtime = run_r2(args)
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if args.runtime_output:
        runtime_target = Path(args.runtime_output)
        runtime_target.parent.mkdir(parents=True, exist_ok=True)
        runtime_target.write_text(
            json.dumps(runtime, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(
        json.dumps(
            {
                "decision": result["decision"],
                "result_digest": result["result_digest"],
                "candidate_config_digest": result["candidate_config_digest"],
                "provenance_digest": result["provenance_digest"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
