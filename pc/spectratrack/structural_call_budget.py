"""A4's single prelocked structural invocation experiment; never used by production.

No temporal tracking evidence is inferred from sparse NightOwls selected frames.
The authoritative quality accounting is spectratrack.qa_benchmark.evaluate_frames.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
from typing import Any

import cv2

from .detector import YoloOnnxDetector, _merge_detections, _tile_regions
from .integrity import sha256_file
from .performance_smoke import load_selected_image_frames
from .qa_benchmark import PredictedObject, evaluate_frames, load_ground_truth
from .types import Detection


LOCK_PATH = Path(__file__).resolve().parents[1] / (
    "benchmarks/vnext/performance/R2_STRUCTURAL_BUDGET_LOCK.v1.json"
)
# Exact Git blob from the separate, pre-data lock commit 0f910b6050c40c22e0607d548402389d528592e3.
LOCK_GIT_BLOB_SHA1 = "351bae3ce60f7365f79cd989a98551e962049c49"
CANDIDATE_ID = "global_plus_least_supported_one_raw_tile_v1"


def locked_contract() -> dict[str, Any]:
    raw = LOCK_PATH.read_bytes()
    # Git stores this authored lock with LF. Windows checkout can materialize CRLF;
    # reconstruct the canonical committed blob without altering the JSON contract.
    git_bytes = raw.replace(b"\r\n", b"\n")
    blob = b"blob " + str(len(git_bytes)).encode("ascii") + b"\0" + git_bytes
    if hashlib.sha1(blob).hexdigest() != LOCK_GIT_BLOB_SHA1:  # noqa: S324 - Git blob integrity
        raise ValueError("pre-R2 immutable A4 lock blob differs from the predeclared commit")
    lock = json.loads(raw)
    if (
        lock["one_candidate"]["id"] != CANDIDATE_ID
        or lock["one_candidate"]["maximum_onnx_calls_per_selected_frame"] != 2
        or lock["one_candidate"]["admissible_tile_count"] != 2
        or lock["shared_workload"]["enhancement_mode"] != "off"
    ):
        raise ValueError("unexpected immutable A4 invocation contract")
    return lock


def least_supported_tile(
    frame_key: str,
    full_detections: list[Detection],
    regions: list[tuple[int, int, int, int]],
) -> tuple[int, tuple[int, int]]:
    """Use full-frame detections only: target fewer already-explained person centers."""
    if len(regions) != 2:
        raise ValueError("LOCK_GEOMETRY_MISMATCH: exactly two production tiles are required")
    if not frame_key:
        raise ValueError("frame key must be nonempty")
    support = [0, 0]
    for detection in full_detections:
        if detection.label.lower() != "person":
            continue
        center_x, center_y = detection.center
        for index, (x1, y1, x2, y2) in enumerate(regions):
            if x1 <= center_x < x2 and y1 <= center_y < y2:
                support[index] += 1
    if support[0] == support[1]:
        index = hashlib.sha256(frame_key.encode("utf-8")).digest()[0] % 2
    else:
        index = 0 if support[0] < support[1] else 1
    return index, (support[0], support[1])


def detect_global_plus_one_tile(
    detector: YoloOnnxDetector,
    frame_bgr: Any,
    *,
    frame_key: str,
    person_threshold: float,
    tile_size: int,
    overlap: float,
    merge_iou: float,
) -> tuple[list[Detection], int]:
    """Experimental A4 invocation schedule; uses production per-call decode and fusion."""
    if not 0.0 < person_threshold <= 1.0:
        raise ValueError("invalid person threshold")
    if not 0.0 <= overlap < 1.0 or tile_size <= 0:
        raise ValueError("invalid tile geometry")
    if not 0.0 < merge_iou <= 1.0:
        raise ValueError("invalid merge IoU")
    h, w = frame_bgr.shape[:2]
    regions = _tile_regions(w, h, tile_size, overlap)
    if len(regions) != 2:
        raise ValueError("LOCK_GEOMETRY_MISMATCH: exactly two production tiles are required")
    person_ids = {i for i, label in enumerate(detector.labels) if label.lower() == "person"}
    if not person_ids:
        raise ValueError("LOCK_MODEL_MISMATCH: person label absent")
    thresholds = dict(detector.class_thresholds)
    thresholds["person"] = person_threshold

    detector._reset_policy_metrics()
    detector.last_policy_counts["tile_count"] = 2
    started = time.perf_counter()
    combined = detector._detect_once(frame_bgr, thresholds)
    detector._add_policy_ms("full_pass", (time.perf_counter() - started) * 1000.0)
    detector._increment_policy_count("full_frame_calls")
    selected_index, _ = least_supported_tile(frame_key, combined, regions)

    x1, y1, x2, y2 = regions[selected_index]
    started = time.perf_counter()
    tile_detections = detector._detect_once(frame_bgr[y1:y2, x1:x2], thresholds)
    detector._add_policy_ms("raw_roi_passes", (time.perf_counter() - started) * 1000.0)
    detector._increment_policy_count("raw_roi_calls")
    for detection in tile_detections:
        if detection.class_id not in person_ids:
            continue
        bx1, by1, bx2, by2 = detection.bbox
        combined.append(
            Detection(
                (bx1 + x1, by1 + y1, bx2 + x1, by2 + y1),
                detection.score,
                detection.class_id,
                detection.label,
                detection.appearance,
            )
        )
    detector.last_policy_counts["fusion_inputs"] = len(combined)
    started = time.perf_counter()
    fused = _merge_detections(combined, merge_iou)
    detector._add_policy_ms("fusion", (time.perf_counter() - started) * 1000.0)
    detector.last_policy_counts["fusion_outputs"] = len(fused)
    if detector.last_inference_calls != 2:
        raise ValueError("LOCK_BUDGET_VIOLATION: candidate did not perform exactly two ONNX calls")
    if detector.last_policy_counts.get("enhanced_roi_calls", 0):
        raise ValueError("LOCK_POLICY_VIOLATION: enhancement must be OFF")
    return fused, selected_index


def _hash_detections(detections: list[Detection]) -> str:
    data = [
        {
            "bbox": [float(value) for value in detection.bbox],
            "score": float(detection.score),
            "class_id": int(detection.class_id),
            "label": detection.label,
        }
        for detection in detections
    ]
    encoded = json.dumps(data, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _scored_quality(metrics: dict[str, Any]) -> dict[str, Any]:
    if metrics["missing_prediction_frames"]:
        raise ValueError("missing prediction frames must not be counted as observed quality")
    return {
        key: metrics[key]
        for key in ("tp", "fp", "fn", "gt", "predictions", "precision", "recall")
    }


def verify_pair(report: dict[str, Any], lock: dict[str, Any]) -> dict[str, Any]:
    """Fail closed on pairing/accounting drift BEFORE calculating candidate gates."""
    if report.get("candidate_id") != lock["one_candidate"]["id"]:
        raise ValueError("LOCK_CANDIDATE_ID_MISMATCH")
    entries = report["paired_frames"]
    count = int(report["frames"])
    if count != len(entries) or count <= 0:
        raise ValueError("PAIR_FRAME_COUNT_MISMATCH")
    keys = [item["key"] for item in entries]
    if keys != report["frame_keys"] or len(set(keys)) != count:
        raise ValueError("PAIR_SELECTED_WINDOW_MISMATCH")
    for entry in entries:
        if entry["control"]["key"] != entry["candidate"]["key"] or entry["control"]["key"] != entry["key"]:
            raise ValueError("PAIR_FRAME_KEY_MISMATCH")
        if entry["control"]["input_sha256"] != entry["candidate"]["input_sha256"]:
            raise ValueError("PAIR_INPUT_BYTES_MISMATCH")
        if entry["control"]["input_sha256"] != entry["input_sha256"]:
            raise ValueError("PAIR_INPUT_BYTES_MISMATCH")
        control_counts = entry["control"]["calls"]
        candidate_counts = entry["candidate"]["calls"]
        if (control_counts["full"], control_counts["raw"], control_counts["enhanced"], control_counts["total"]) != (1, 2, 0, 3):
            raise ValueError("CONTROL_INVOCATION_DRIFT")
        if (candidate_counts["full"], candidate_counts["raw"], candidate_counts["enhanced"], candidate_counts["total"]) != (1, 1, 0, 2):
            raise ValueError("CANDIDATE_INVOCATION_DRIFT")
        if entry["selected_tile"] not in (0, 1):
            raise ValueError("CANDIDATE_TILE_INDEX_INVALID")

    calls = report["calls"]
    if calls["control"] != 3 * count or calls["candidate"] != 2 * count:
        raise ValueError("AGGREGATE_CALL_COUNT_DRIFT")
    if report["providers"]["control"] != report["providers"]["candidate"]:
        raise ValueError("PAIR_PROVIDER_MISMATCH")
    if report["providers"]["control"] != lock["r1_evidence"]["providers"]:
        raise ValueError("LOCK_PROVIDER_MISMATCH")
    if report["model_sha256"] != lock["r1_evidence"]["model_sha256"]:
        raise ValueError("LOCK_MODEL_SHA_MISMATCH")
    locked_settings = {
        key: lock["shared_workload"][key] for key in (
            "input_size", "person_threshold", "tile_size", "tile_overlap",
            "merge_iou", "enhancement_mode", "label", "matching_iou",
        )
    }
    if report["settings"] != locked_settings:
        raise ValueError("LOCK_SETTINGS_DRIFT")
    if report["control"]["ground_truth_sha256"] != report["candidate"]["ground_truth_sha256"]:
        raise ValueError("PAIR_GT_SHA_MISMATCH")
    if report["control"]["selection_manifest_sha256"] != report["candidate"]["selection_manifest_sha256"]:
        raise ValueError("PAIR_SELECTION_SHA_MISMATCH")
    if report["control"]["selected_keys_sha256"] != report["candidate"]["selected_keys_sha256"]:
        raise ValueError("PAIR_SELECTION_KEYS_SHA_MISMATCH")
    for name in ("control", "candidate"):
        quality = report[name]["quality"]
        tp, fp, fn, gt = [quality[key] for key in ("tp", "fp", "fn", "gt")]
        if any(type(value) is not int or value < 0 for value in (tp, fp, fn, gt)):
            raise ValueError("QUALITY_COUNT_TYPE_DRIFT")
        if tp + fn != gt:
            raise ValueError("QUALITY_GT_ACCOUNTING_DRIFT")
        precision = tp / (tp + fp) if tp + fp else 1.0
        recall = tp / gt if gt else 1.0
        if abs(float(quality["precision"]) - precision) > 1e-12:
            raise ValueError("QUALITY_PRECISION_ACCOUNTING_DRIFT")
        if abs(float(quality["recall"]) - recall) > 1e-12:
            raise ValueError("QUALITY_RECALL_ACCOUNTING_DRIFT")
        if quality["predictions"] < tp + fp:
            raise ValueError("QUALITY_PREDICTION_ACCOUNTING_DRIFT")
    control = report["control"]["quality"]
    candidate = report["candidate"]["quality"]
    if control["gt"] != candidate["gt"]:
        raise ValueError("PAIR_GT_DENOMINATOR_MISMATCH")

    if report["phase"] == "r1":
        expected = lock["r1_evidence"]
        if (
            count != expected["frames"]
            or control["tp"] != expected["true_positive"]
            or control["fp"] != expected["false_positive"]
            or control["fn"] != expected["false_negative"]
            or control["gt"] != expected["scored_gt"]
            or report["control"]["ground_truth_sha256"] != expected["ground_truth_sha256"]
            or report["control"]["selection_manifest_sha256"] != expected["selection_proof_sha256"]
            or report["control"]["selected_keys_sha256"] != expected["selected_keys_sha256"]
        ):
            raise ValueError("R1_CONTROL_REPRODUCIBILITY_MISMATCH")

    gate = lock["predeclared_reject_gates"]
    wall_control = float(report["control"]["detector_wall_seconds"])
    wall_candidate = float(report["candidate"]["detector_wall_seconds"])
    if not 0.0 < wall_control or not 0.0 < wall_candidate:
        raise ValueError("DETECTOR_LATENCY_UNMEASURED")
    speedup = wall_control / wall_candidate
    reduction = (calls["control"] - calls["candidate"]) / calls["control"]
    delta = {
        "recall": candidate["recall"] - control["recall"],
        "precision": candidate["precision"] - control["precision"],
        "fn": candidate["fn"] - control["fn"],
        "fp": candidate["fp"] - control["fp"],
        "tp": candidate["tp"] - control["tp"],
    }
    rejected = []
    if speedup < gate["minimum_measured_end_to_end_detector_speedup_x"]:
        rejected.append("SPEEDUP_LT_1_10")
    if reduction + 1e-12 < gate["minimum_onnx_call_reduction_fraction"]:
        rejected.append("CALL_REDUCTION_BELOW_LOCK")
    if delta["recall"] < gate["recall_delta_minimum"]:
        rejected.append("RECALL_REGRESSION")
    if delta["precision"] < gate["precision_delta_minimum"]:
        rejected.append("PRECISION_REGRESSION")
    if delta["fn"] > gate["fn_delta_maximum"]:
        rejected.append("NEW_FALSE_NEGATIVES")
    if delta["fp"] > gate["fp_delta_maximum"]:
        rejected.append("NEW_FALSE_POSITIVES")
    return {
        "decision": "REJECT" if rejected else gate["decision_if_every_gate_passes"],
        "rejected_gates": rejected,
        "speedup_x": speedup,
        "call_reduction_fraction": reduction,
        "quality_delta": delta,
        "scope": "paired detector policy only; no source-second/temporal GT claim",
    }


def _validate_input_provenance(
    *,
    phase: str,
    selection: dict[str, Any],
    frames: list[Any],
    lock: dict[str, Any],
    ground_truth_path: str,
    selection_manifest_path: str,
    expected_r2_gt_sha256: str | None = None,
    expected_r2_selection_sha256: str | None = None,
    expected_r2_keys_sha256: str | None = None,
    r1_ground_truth_path: str | None = None,
    r1_selection_manifest_path: str | None = None,
) -> None:
    if phase == "r1":
        for key, locked_key in (
            ("ground_truth_sha256", "ground_truth_sha256"),
            ("selection_manifest_sha256", "selection_proof_sha256"),
            ("selection_keys_sha256", "selected_keys_sha256"),
        ):
            if selection[key] != lock["r1_evidence"][locked_key]:
                raise ValueError(f"R1_PROOF_SHA_MISMATCH: {key}")
        return
    if phase != "r2":
        raise ValueError("phase must be r1 or r2")
    provided = (
        expected_r2_gt_sha256,
        expected_r2_selection_sha256,
        expected_r2_keys_sha256,
        r1_ground_truth_path,
        r1_selection_manifest_path,
    )
    if not all(provided):
        raise ValueError("R2_REQUIRES_EXPLICIT_FROZEN_HASHES_AND_R1_DISJOINTNESS_PROOF")
    if selection["ground_truth_sha256"] != expected_r2_gt_sha256:
        raise ValueError("R2_GT_SHA_MISMATCH")
    if selection["selection_manifest_sha256"] != expected_r2_selection_sha256:
        raise ValueError("R2_SELECTION_SHA_MISMATCH")
    if selection["selection_keys_sha256"] != expected_r2_keys_sha256:
        raise ValueError("R2_KEYS_SHA_MISMATCH")
    r1_frames, r1_proof = load_selected_image_frames(
        r1_ground_truth_path, r1_selection_manifest_path, expected_frames=lock["r1_evidence"]["frames"]
    )
    for key, locked_key in (
        ("ground_truth_sha256", "ground_truth_sha256"),
        ("selection_manifest_sha256", "selection_proof_sha256"),
        ("selection_keys_sha256", "selected_keys_sha256"),
    ):
        if r1_proof[key] != lock["r1_evidence"][locked_key]:
            raise ValueError("R1_DISJOINTNESS_PROOF_UNTRUSTED")
    r1_sources = {(frame.source.replace("\\", "/").casefold(), frame.key) for frame in r1_frames}
    r1_source_paths = {src for src, _ in r1_sources}
    r1_keys = {key for _, key in r1_sources}
    if r1_keys.intersection(frame.key for frame in frames) or r1_source_paths.intersection(
        frame.source.replace("\\", "/").casefold() for frame in frames
    ):
        raise ValueError("R2_R1_FRAME_OR_SOURCE_LEAKAGE")
    if ground_truth_path == r1_ground_truth_path and selection_manifest_path == r1_selection_manifest_path:
        raise ValueError("R2_MUST_BE_INDEPENDENT_OF_R1")


def _calls(detector: YoloOnnxDetector) -> dict[str, int]:
    counts = detector.last_policy_counts
    result = {
        "full": int(counts.get("full_frame_calls", 0)),
        "raw": int(counts.get("raw_roi_calls", 0)),
        "enhanced": int(counts.get("enhanced_roi_calls", 0)),
        "total": int(detector.last_inference_calls),
    }
    if result["total"] != result["full"] + result["raw"] + result["enhanced"]:
        raise ValueError("LOW_LEVEL_ONNX_COUNTER_MISMATCH")
    return result


def _detector_stage(detector: YoloOnnxDetector) -> dict[str, float]:
    return {name: float(detector.last_stage_ms.get(name, 0.0)) for name in (
        "preprocess", "inference", "postprocess"
    )}


def run_paired_selected_images(
    *,
    phase: str,
    ground_truth_path: str,
    selection_manifest_path: str,
    source_root: str,
    model_path: str,
    source_commit: str,
    expected_r2_gt_sha256: str | None = None,
    expected_r2_selection_sha256: str | None = None,
    expected_r2_keys_sha256: str | None = None,
    r1_ground_truth_path: str | None = None,
    r1_selection_manifest_path: str | None = None,
    detector_factory: Any = YoloOnnxDetector,
) -> dict[str, Any]:
    """Run same ordered images/GT through control and prelocked candidate in one process.

    CLI never discovers holdout/full paths. No model binary or R2 fixture is shipped.
    """
    lock = locked_contract()
    if len(source_commit) != 40 or any(c not in "0123456789abcdef" for c in source_commit.lower()):
        raise ValueError("source_commit must be a complete 40-character SHA")
    frames, selection = load_selected_image_frames(
        ground_truth_path, selection_manifest_path, expected_frames=lock["r1_evidence"]["frames"]
    )
    _validate_input_provenance(
        phase=phase,
        selection=selection,
        frames=frames,
        lock=lock,
        ground_truth_path=ground_truth_path,
        selection_manifest_path=selection_manifest_path,
        expected_r2_gt_sha256=expected_r2_gt_sha256,
        expected_r2_selection_sha256=expected_r2_selection_sha256,
        expected_r2_keys_sha256=expected_r2_keys_sha256,
        r1_ground_truth_path=r1_ground_truth_path,
        r1_selection_manifest_path=r1_selection_manifest_path,
    )
    model_hash = sha256_file(model_path)
    if model_hash != lock["r1_evidence"]["model_sha256"]:
        raise ValueError("LOCK_MODEL_SHA_MISMATCH")
    settings = lock["shared_workload"]
    constructor_options = {
        "input_size": settings["input_size"],
        "conf_threshold": 0.35,
        "iou_threshold": 0.45,
        "prefer_gpu": True,
    }
    control_detector = detector_factory(model_path, **constructor_options)
    candidate_detector = detector_factory(model_path, **constructor_options)
    if control_detector.providers != candidate_detector.providers:
        raise ValueError("PAIR_PROVIDER_MISMATCH")
    if control_detector.providers != lock["r1_evidence"]["providers"]:
        raise ValueError("LOCK_PROVIDER_MISMATCH")

    root = Path(source_root).resolve()
    paired_frames = []
    predictions: dict[str, dict[tuple[str, int], list[PredictedObject]]] = {
        "control": {}, "candidate": {}
    }
    totals = {"control": 0.0, "candidate": 0.0}
    counts = {"control": 0, "candidate": 0}
    stages = {"control": {"preprocess": 0.0, "inference": 0.0, "postprocess": 0.0},
              "candidate": {"preprocess": 0.0, "inference": 0.0, "postprocess": 0.0}}
    ordered_image_hashes = []
    for frame_index, item in enumerate(frames):
        path = (root / item.source).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError(f"invalid selected source path: {item.key}")
        input_sha = sha256_file(path)
        image = cv2.imread(str(path))
        if image is None:
            raise ValueError(f"unreadable locked image: {item.key}")
        if len(_tile_regions(image.shape[1], image.shape[0], settings["tile_size"], settings["tile_overlap"])) != 2:
            raise ValueError(f"LOCK_GEOMETRY_MISMATCH: {item.key}")
        ordered_image_hashes.append({"key": item.key, "sha256": input_sha})

        if frame_index == 0:
            # Equal one-call warmup for both sessions; excluded from scored invocations
            # and detector-policy wall time, explicitly recorded in the report.
            control_detector.detect(image)
            if control_detector.last_inference_calls != 1:
                raise ValueError("CONTROL_WARMUP_CALL_COUNT_DRIFT")
            candidate_detector.detect(image)
            if candidate_detector.last_inference_calls != 1:
                raise ValueError("CANDIDATE_WARMUP_CALL_COUNT_DRIFT")

        per_frame: dict[str, dict[str, Any]] = {}
        chosen_tile = None
        # Deterministic AB/BA order avoids consistently crediting warm host/cache to one policy.
        execution_order = ("control", "candidate") if frame_index % 2 == 0 else ("candidate", "control")
        for label in execution_order:
            detector = control_detector if label == "control" else candidate_detector
            started = time.perf_counter()
            if label == "control":
                result = detector.detect_people_recall(
                    image, person_threshold=settings["person_threshold"],
                    tile_size=settings["tile_size"], tile_overlap=settings["tile_overlap"],
                    merge_iou_threshold=settings["merge_iou"], enhancement_mode="off"
                )
            else:
                result, chosen_tile = detect_global_plus_one_tile(
                    detector, image, frame_key=item.key,
                    person_threshold=settings["person_threshold"],
                    tile_size=settings["tile_size"], overlap=settings["tile_overlap"],
                    merge_iou=settings["merge_iou"],
                )
            elapsed = time.perf_counter() - started
            call_counts = _calls(detector)
            expected_calls = 3 if label == "control" else 2
            if call_counts["total"] != expected_calls:
                raise ValueError(f"LOCK_BUDGET_VIOLATION: {label}, {item.key}")
            metrics = _detector_stage(detector)
            totals[label] += elapsed
            counts[label] += call_counts["total"]
            for name, value in metrics.items():
                stages[label][name] += value
            predictions[label][(item.video, item.frame)] = [
                PredictedObject(tuple(det.bbox), det.label, float(det.score))
                for det in result
            ]
            per_frame[label] = {
                "key": item.key,
                "input_sha256": input_sha,
                "wall_ms": elapsed * 1000.0,
                "calls": call_counts,
                "stage_ms": metrics,
                "detections": len(result),
                "detection_sha256": _hash_detections(result),
            }
        paired_frames.append({
            "key": item.key,
            "input_sha256": input_sha,
            "selected_tile": chosen_tile,
            "control": per_frame["control"],
            "candidate": per_frame["candidate"],
        })

    selected_keys = {(item.video, item.frame) for item in frames}
    truth = [frame for frame in load_ground_truth(ground_truth_path)
             if (frame.video, frame.frame) in selected_keys]
    if len(truth) != len(frames):
        raise ValueError("selected frame-to-GT one-to-one mapping failed")
    quality = {
        label: _scored_quality(
            evaluate_frames(truth, predictions[label], None, label=settings["label"],
                            iou_threshold=settings["matching_iou"])
        )
        for label in ("control", "candidate")
    }
    digest = hashlib.sha256(
        json.dumps(ordered_image_hashes, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    report = {
        "schema": "spectratrack.a4.structural_detector_call_budget.paired_v1",
        "phase": phase,
        "candidate_id": CANDIDATE_ID,
        "scope": "research-only; not production; no tracking GT",
        "lock_git_blob_sha1": LOCK_GIT_BLOB_SHA1,
        "lock_commit": "0f910b6050c40c22e0607d548402389d528592e3",
        "source_commit": source_commit,
        "model_sha256": model_hash,
        "providers": {
            "control": control_detector.providers,
            "candidate": candidate_detector.providers,
        },
        "settings": {
            key: settings[key] for key in (
                "input_size", "person_threshold", "tile_size", "tile_overlap",
                "merge_iou", "enhancement_mode", "label", "matching_iou",
            )
        },
        "frames": len(frames),
        "frame_keys": [item.key for item in frames],
        "ordered_image_bytes_sha256": digest,
        "paired_frames": paired_frames,
        "calls": counts,
        "warmup_calls_excluded_from_budget": {"control": 1, "candidate": 1},
        "control": {
            "ground_truth_sha256": selection["ground_truth_sha256"],
            "selection_manifest_sha256": selection["selection_manifest_sha256"],
            "selected_keys_sha256": selection["selection_keys_sha256"],
            "detector_wall_seconds": totals["control"],
            "stage_totals_ms": stages["control"],
            "quality": quality["control"],
        },
        "candidate": {
            "ground_truth_sha256": selection["ground_truth_sha256"],
            "selection_manifest_sha256": selection["selection_manifest_sha256"],
            "selected_keys_sha256": selection["selection_keys_sha256"],
            "detector_wall_seconds": totals["candidate"],
            "stage_totals_ms": stages["candidate"],
            "quality": quality["candidate"],
        },
        "source_fps": None,
        "processing_seconds_per_source_second": None,
        "source_fps_unavailable_reason": "Sparse selected NightOwls still images are not a contiguous video.",
        "gpu_vram": None,
        "gpu_vram_reason": "No trustworthy per-process DirectML telemetry.",
    }
    report["gate"] = verify_pair(report, lock)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="A4 prelocked structural invocation paired validation")
    parser.add_argument("--phase", choices=("r1", "r2"), required=True)
    parser.add_argument("--ground-truth", required=True)
    parser.add_argument("--selection-manifest", required=True)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--expected-r2-gt-sha256")
    parser.add_argument("--expected-r2-selection-sha256")
    parser.add_argument("--expected-r2-keys-sha256")
    parser.add_argument("--r1-ground-truth")
    parser.add_argument("--r1-selection-manifest")
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    result = run_paired_selected_images(
        phase=args.phase,
        ground_truth_path=args.ground_truth,
        selection_manifest_path=args.selection_manifest,
        source_root=args.source_root,
        model_path=args.model,
        source_commit=args.source_commit,
        expected_r2_gt_sha256=args.expected_r2_gt_sha256,
        expected_r2_selection_sha256=args.expected_r2_selection_sha256,
        expected_r2_keys_sha256=args.expected_r2_keys_sha256,
        r1_ground_truth_path=args.r1_ground_truth,
        r1_selection_manifest_path=args.r1_selection_manifest,
    )
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "decision": result["gate"]["decision"],
        "calls": result["calls"],
        "speedup_x": result["gate"]["speedup_x"],
        "output": str(target),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
