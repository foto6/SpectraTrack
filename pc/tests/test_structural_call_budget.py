from __future__ import annotations

import copy
import hashlib
import json

import cv2
import numpy as np
import pytest

from spectratrack.detector import YoloOnnxDetector
from spectratrack.performance_smoke import load_selected_image_frames
from spectratrack.qa_benchmark import (
    GroundTruthFrame,
    GroundTruthObject,
    PredictedObject,
    evaluate_frames,
)
from spectratrack.structural_call_budget import (
    _scored_quality,
    _validate_input_provenance,
    detect_global_plus_one_tile,
    least_supported_tile,
    locked_contract,
    run_paired_selected_images,
    verify_pair,
)
from spectratrack.types import Detection


def _settings(lock):
    return {
        key: lock["shared_workload"][key] for key in (
            "input_size", "person_threshold", "tile_size", "tile_overlap",
            "merge_iou", "enhancement_mode", "label", "matching_iou",
        )
    }


def _quality(tp=1, fp=0, fn=0, gt=1):
    assert tp + fn == gt
    return {
        "tp": tp, "fp": fp, "fn": fn, "gt": gt, "predictions": tp + fp,
        "precision": tp / (tp + fp) if tp + fp else 1.0,
        "recall": tp / gt if gt else 1.0,
    }


def _synthetic_pair():
    lock = locked_contract()
    entries = []
    keys = ["synthetic#a", "synthetic#b"]
    for key in keys:
        entries.append({
            "key": key, "input_sha256": "a" * 64, "selected_tile": 0,
            "control": {
                "key": key, "input_sha256": "a" * 64,
                "calls": {"full": 1, "raw": 2, "enhanced": 0, "total": 3},
            },
            "candidate": {
                "key": key, "input_sha256": "a" * 64,
                "calls": {"full": 1, "raw": 1, "enhanced": 0, "total": 2},
            },
        })
    proof = {"ground_truth_sha256": "c" * 64, "selection_manifest_sha256": "d" * 64,
             "selected_keys_sha256": "e" * 64}
    return lock, {
        "schema": "spectratrack.a4.structural_detector_call_budget.paired_v1",
        "phase": "r2",
        "frames": 2,
        "frame_keys": keys,
        "paired_frames": entries,
        "calls": {"control": 6, "candidate": 4},
        "providers": {"control": lock["r1_evidence"]["providers"],
                      "candidate": lock["r1_evidence"]["providers"]},
        "model_sha256": lock["r1_evidence"]["model_sha256"],
        "settings": _settings(lock),
        "control": {
            **proof, "detector_wall_seconds": 2.0, "quality": _quality(),
        },
        "candidate": {
            **proof, "detector_wall_seconds": 1.5, "quality": _quality(),
        },
    }


def test_pre_data_lock_git_blob_is_frozen_and_micro_candidate_is_rejected(monkeypatch, tmp_path):
    lock = locked_contract()
    assert lock["starting_head"] == "5c880267498a18b51b2e5c5f92f6c5360fafc212"
    assert lock["r1_evidence"]["micro_candidate_contiguous_speedup_x"] == 1.027
    assert lock["r1_evidence"]["micro_candidate_gate"] == "REJECTED_LT_1.10"
    assert lock["one_candidate"]["maximum_onnx_calls_per_selected_frame"] == 2
    assert lock["one_candidate"]["expected_calls_for_400_frames"] == 800
    assert lock["one_candidate"]["control_calls_for_400_frames"] == 1200
    assert lock["predeclared_reject_gates"]["minimum_measured_end_to_end_detector_speedup_x"] == 1.10

    modified = tmp_path / "different-lock.json"
    modified.write_text(json.dumps(lock), encoding="utf-8")
    monkeypatch.setattr("spectratrack.structural_call_budget.LOCK_PATH", modified)
    with pytest.raises(ValueError, match="immutable A4 lock blob differs"):
        locked_contract()


def test_least_supported_tile_uses_only_full_frame_person_support():
    regions = [(0, 0, 64, 64), (64, 0, 128, 64)]
    left_person = Detection((5.0, 5.0, 25.0, 45.0), 0.9, 0, "person")
    unrelated_car = Detection((70.0, 5.0, 95.0, 45.0), 0.9, 1, "car")
    chosen, counts = least_supported_tile("f#1", [left_person, unrelated_car], regions)
    assert chosen == 1
    assert counts == (1, 0)

    equal_a = least_supported_tile("f#tie", [], regions)
    equal_b = least_supported_tile("f#tie", [], regions)
    assert equal_a == equal_b
    assert equal_a[0] in (0, 1)
    with pytest.raises(ValueError, match="exactly two"):
        least_supported_tile("f#1", [], regions[:1])


class _FakeDetector(YoloOnnxDetector):
    """Synthetic invocation counter, not an ONNX latency/quality claim."""

    def __init__(self, _model, **_options):
        self.labels = ["person", "car"]
        self.providers = ["DmlExecutionProvider", "CPUExecutionProvider"]
        self.class_thresholds = {}
        self.last_stage_ms = {}
        self.last_stage_cpu_ms = {}
        self.last_policy_ms = {}
        self.last_policy_counts = {}
        self.last_inference_calls = 0

    def _detect_once(self, frame_bgr, _thresholds=None, allowed_class_ids=None):
        self.last_inference_calls += 1
        self.last_stage_ms["inference"] = self.last_stage_ms.get("inference", 0.0) + 0.01
        # In this fixture the full frame misses a small target. Only the right tile finds it.
        if frame_bgr.shape[1] > 64 or float(np.mean(frame_bgr)) < 200.0:
            return []
        return [Detection((12.0, 4.0, 30.0, 45.0), 0.9, 0, "person")]


def _image():
    image = np.zeros((64, 96, 3), dtype=np.uint8)
    image[:, 32:, :] = 255
    return image


def test_actual_production_control_vs_one_tile_calls_without_changing_fusion():
    frame = _image()
    control = _FakeDetector("synthetic")
    candidate = _FakeDetector("synthetic")
    selected_key = next(
        f"test#{value}" for value in range(100)
        if hashlib.sha256(f"test#{value}".encode()).digest()[0] % 2 == 0
    )
    baseline = control.detect_people_recall(
        frame, person_threshold=0.12, tile_size=64,
        tile_overlap=0.20, merge_iou_threshold=0.55, enhancement_mode="off"
    )
    candidate_detections, chosen = detect_global_plus_one_tile(
        candidate, frame, frame_key=selected_key,
        person_threshold=0.12, tile_size=64, overlap=0.20, merge_iou=0.55,
    )
    assert control.last_inference_calls == 3
    assert candidate.last_inference_calls == 2
    assert (control.last_policy_counts["full_frame_calls"],
            control.last_policy_counts["raw_roi_calls"]) == (1, 2)
    assert (candidate.last_policy_counts["full_frame_calls"],
            candidate.last_policy_counts["raw_roi_calls"]) == (1, 1)
    assert chosen == 0
    assert len(baseline) == 1
    assert candidate_detections == []  # Must reject quality loss, not mislabel calls as quality win.


def test_non_two_tile_geometry_blocks_before_any_inference():
    detector = _FakeDetector("synthetic")
    with pytest.raises(ValueError, match="LOCK_GEOMETRY_MISMATCH"):
        detect_global_plus_one_tile(
            detector, np.zeros((1080, 1920, 3), dtype=np.uint8),
            frame_key="v#0", person_threshold=0.12,
            tile_size=640, overlap=0.20, merge_iou=0.55,
        )
    assert detector.last_inference_calls == 0


def test_paired_validator_requires_same_window_same_bytes_and_exact_calls():
    lock, pair = _synthetic_pair()
    gate = verify_pair(pair, lock)
    assert gate["call_reduction_fraction"] == pytest.approx(1 / 3)
    assert gate["speedup_x"] == pytest.approx(4 / 3)
    assert gate["decision"] == "ELIGIBLE_FOR_INDEPENDENT_REVIEW_ONLY_NO_MERGE"
    cases = [
        ("candidate", "key", "DIFFERENT", "PAIR_FRAME_KEY_MISMATCH"),
        ("candidate", "input_sha256", "b" * 64, "PAIR_INPUT_BYTES_MISMATCH"),
        ("candidate", "calls", {"full": 1, "raw": 2, "enhanced": 0, "total": 3},
         "CANDIDATE_INVOCATION_DRIFT"),
    ]
    for group, field, value, expected in cases:
        bad = copy.deepcopy(pair)
        bad["paired_frames"][0][group][field] = value
        with pytest.raises(ValueError, match=expected):
            verify_pair(bad, lock)
    bad = copy.deepcopy(pair)
    bad["frame_keys"].reverse()
    with pytest.raises(ValueError, match="PAIR_SELECTED_WINDOW_MISMATCH"):
        verify_pair(bad, lock)
    bad = copy.deepcopy(pair)
    bad["calls"]["candidate"] = 5
    with pytest.raises(ValueError, match="AGGREGATE_CALL_COUNT_DRIFT"):
        verify_pair(bad, lock)


def test_paired_validator_refuses_precision_recall_and_gt_accounting_drift():
    lock, pair = _synthetic_pair()
    for field, value, expected in [
        ("recall", 0.0, "QUALITY_RECALL_ACCOUNTING_DRIFT"),
        ("precision", 0.0, "QUALITY_PRECISION_ACCOUNTING_DRIFT"),
        ("fn", 1, "QUALITY_GT_ACCOUNTING_DRIFT"),
        ("predictions", 0, "QUALITY_PREDICTION_ACCOUNTING_DRIFT"),
    ]:
        bad = copy.deepcopy(pair)
        bad["candidate"]["quality"][field] = value
        with pytest.raises(ValueError, match=expected):
            verify_pair(bad, lock)
    bad = copy.deepcopy(pair)
    bad["candidate"]["ground_truth_sha256"] = "z" * 64
    with pytest.raises(ValueError, match="PAIR_GT_SHA_MISMATCH"):
        verify_pair(bad, lock)


def test_paired_quality_nonregression_and_speed_gates_are_independent():
    lock, pair = _synthetic_pair()
    bad = copy.deepcopy(pair)
    bad["candidate"]["quality"] = _quality(tp=0, fp=0, fn=1, gt=1)
    gate = verify_pair(bad, lock)
    assert gate["decision"] == "REJECT"
    assert "RECALL_REGRESSION" in gate["rejected_gates"]
    assert "NEW_FALSE_NEGATIVES" in gate["rejected_gates"]

    fp_bad = copy.deepcopy(pair)
    fp_bad["candidate"]["quality"] = _quality(tp=1, fp=1, fn=0, gt=1)
    gate = verify_pair(fp_bad, lock)
    assert "PRECISION_REGRESSION" in gate["rejected_gates"]
    assert "NEW_FALSE_POSITIVES" in gate["rejected_gates"]

    slow = copy.deepcopy(pair)
    slow["candidate"]["detector_wall_seconds"] = 1.95
    gate = verify_pair(slow, lock)
    assert gate["decision"] == "REJECT"
    assert "SPEEDUP_LT_1_10" in gate["rejected_gates"]
    assert gate["call_reduction_fraction"] == pytest.approx(1 / 3)


def test_canonical_evaluator_preserves_ignored_gt_and_fp_fn_denominator():
    truth = [GroundTruthFrame(
        video="synthetic", frame=0, tags=(),
        objects=(
            GroundTruthObject("person-a", "person", (40.0, 4.0, 60.0, 45.0)),
            GroundTruthObject("ignore-a", "person", (3.0, 4.0, 23.0, 45.0), ignore=True),
        ),
    )]
    control = {(truth[0].video, truth[0].frame): [
        PredictedObject((40.0, 4.0, 60.0, 45.0), "person", 0.90),
        PredictedObject((3.0, 4.0, 23.0, 45.0), "person", 0.80),
    ]}
    candidate = {(truth[0].video, truth[0].frame): []}
    control_quality = _scored_quality(evaluate_frames(truth, control, None, label="person", iou_threshold=0.5))
    candidate_quality = _scored_quality(
        evaluate_frames(truth, candidate, None, label="person", iou_threshold=0.5)
    )
    assert (control_quality["tp"], control_quality["fp"], control_quality["fn"],
            control_quality["gt"]) == (1, 0, 0, 1)
    assert (candidate_quality["tp"], candidate_quality["fp"], candidate_quality["fn"],
            candidate_quality["gt"]) == (0, 0, 1, 1)
    assert control_quality["precision"] == control_quality["recall"] == 1.0
    assert candidate_quality["recall"] == 0.0


def test_r2_requires_explicit_frozen_hashes_and_proven_r1_disjointness(monkeypatch):
    lock = locked_contract()
    frame = type("Selected", (), {"key": "r2#0", "source": "r2.png"})()
    proof = {
        "ground_truth_sha256": "a" * 64,
        "selection_manifest_sha256": "b" * 64,
        "selection_keys_sha256": "c" * 64,
    }
    with pytest.raises(ValueError, match="R2_REQUIRES_EXPLICIT"):
        _validate_input_provenance(
            phase="r2", selection=proof, frames=[frame], lock=lock,
            ground_truth_path="r2.jsonl", selection_manifest_path="r2.json"
        )

    r1 = type("Selected", (), {"key": "r2#0", "source": "r2.png"})()
    r1_proof = {
        "ground_truth_sha256": lock["r1_evidence"]["ground_truth_sha256"],
        "selection_manifest_sha256": lock["r1_evidence"]["selection_proof_sha256"],
        "selection_keys_sha256": lock["r1_evidence"]["selected_keys_sha256"],
    }
    monkeypatch.setattr(
        "spectratrack.structural_call_budget.load_selected_image_frames",
        lambda *args, **kwargs: ([r1], r1_proof),
    )
    with pytest.raises(ValueError, match="R2_R1_FRAME_OR_SOURCE_LEAKAGE"):
        _validate_input_provenance(
            phase="r2", selection=proof, frames=[frame], lock=lock,
            ground_truth_path="r2.jsonl", selection_manifest_path="r2.json",
            expected_r2_gt_sha256=proof["ground_truth_sha256"],
            expected_r2_selection_sha256=proof["selection_manifest_sha256"],
            expected_r2_keys_sha256=proof["selection_keys_sha256"],
            r1_ground_truth_path="r1.jsonl", r1_selection_manifest_path="r1.json",
        )


def test_synthetic_paired_runner_consumes_one_selection_and_one_gt(monkeypatch, tmp_path):
    import spectratrack.structural_call_budget as experiment

    img_root = tmp_path / "images"
    img_root.mkdir()
    gt = tmp_path / "gt.jsonl"
    manifest = tmp_path / "selection.json"
    frames = [
        {"video": "fixture", "frame": i, "source": f"frame-{i}.png",
         "source_sequence": f"fixture-{i}",
         "objects": [{"id": "p", "label": "person", "bbox": [44, 4, 62, 45]}]}
        for i in (0, 1)
    ]
    for row in frames:
        assert cv2.imwrite(str(img_root / row["source"]), _image())
    gt.write_text("".join(json.dumps(row) + "\n" for row in frames), encoding="utf-8")
    manifest.write_text(json.dumps({"frame_ids": ["fixture#1", "fixture#0"]}), encoding="utf-8")
    selected, proof = load_selected_image_frames(gt, manifest, expected_frames=2)
    model = tmp_path / "synthetic.onnx"
    model.write_bytes(b"not a real ONNX model: fake provider test only")

    lock = copy.deepcopy(locked_contract())
    lock["r1_evidence"].update({
        "frames": 2,
        "ground_truth_sha256": proof["ground_truth_sha256"],
        "selection_proof_sha256": proof["selection_manifest_sha256"],
        "selected_keys_sha256": proof["selection_keys_sha256"],
        "model_sha256": hashlib.sha256(model.read_bytes()).hexdigest(),
        "true_positive": 2, "false_positive": 0, "false_negative": 0, "scored_gt": 2,
    })
    lock["shared_workload"].update({
        "input_size": 64, "tile_size": 64, "tile_overlap": 0.20,
    })
    monkeypatch.setattr(experiment, "locked_contract", lambda: lock)
    result = run_paired_selected_images(
        phase="r1", ground_truth_path=str(gt), selection_manifest_path=str(manifest),
        source_root=str(img_root), model_path=str(model),
        source_commit="5c880267498a18b51b2e5c5f92f6c5360fafc212",
        detector_factory=_FakeDetector,
    )
    assert result["frame_keys"] == [item.key for item in selected] == ["fixture#1", "fixture#0"]
    assert result["frames"] == 2
    assert result["calls"] == {"control": 6, "candidate": 4}
    assert all(item["control"]["input_sha256"] == item["candidate"]["input_sha256"]
               for item in result["paired_frames"])
    assert result["control"]["quality"]["tp"] == 2
    assert result["control"]["quality"]["fn"] == 0
    assert result["candidate"]["quality"]["gt"] == 2
    assert result["processing_seconds_per_source_second"] is None
