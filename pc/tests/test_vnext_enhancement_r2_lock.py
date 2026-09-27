from __future__ import annotations

import hashlib
import json
from pathlib import Path

LOCK_SHA256 = "62b0beda76d4a0434a0da4d60f92518a884659211df779b07944f62d91855a93"
A7_DISCOVERY_SHA256 = "ad51f9b7c3f4629155298215a52004092ea1b95bd220c4ed6a608e464971afec"


def _lock_paths() -> tuple[Path, Path]:
    pc_root = Path(__file__).resolve().parents[1]
    base = pc_root / "benchmarks" / "vnext" / "enhancement"
    return base / "r2_hypothesis_lock.json", base / "r2_hypothesis_lock.sha256"


def test_r2_hypothesis_lock_bytes_are_immutable() -> None:
    lock_path, digest_path = _lock_paths()
    payload = lock_path.read_bytes()
    actual = hashlib.sha256(payload).hexdigest()

    assert actual == LOCK_SHA256
    assert digest_path.read_text(encoding="utf-8") == f"{LOCK_SHA256}  r2_hypothesis_lock.json\n"


def test_r2_hypothesis_lock_is_no_candidate_and_pre_exposure() -> None:
    lock_path, _ = _lock_paths()
    lock = json.loads(lock_path.read_text(encoding="utf-8"))

    assert lock["schema"] == "spectratrack-vnext-a3-r2-hypothesis-lock-v1"
    assert lock["status"] == "LOCKED_BEFORE_R2_EXPOSURE"
    assert lock["decision"] == "NO_NEW_CANDIDATE"
    assert lock["locked_candidate"] is None
    assert lock["production_enhancement"] == "OFF"
    assert lock["a7_discovery"]["digest_sha256"] == A7_DISCOVERY_SHA256

    policy = lock["r2_execution_policy"]
    assert policy["r2_gt_inspection_allowed"] is False
    assert policy["r2_image_inspection_allowed"] is False
    assert policy["r2_result_inspection_allowed"] is False
    assert policy["r2_scoring_allowed"] is False
    assert policy["full5000_scoring_allowed"] is False
    assert policy["max_added_detector_calls_for_this_lock"] == 0
    assert policy["max_enhanced_rois_per_source_frame_for_this_lock"] == 0
    assert policy["production_enable_allowed"] is False


def test_reserved_future_gate_remains_strict_and_bounded() -> None:
    lock_path, _ = _lock_paths()
    lock = json.loads(lock_path.read_text(encoding="utf-8"))

    admission = lock["future_candidate_admission"]
    assert admission["requires_new_lock_before_any_r2_exposure"] is True
    assert admission["mandatory_raw_corroboration"] is True
    assert admission["max_enhanced_rois_per_source_frame"] == 1
    assert admission["max_added_detector_calls_per_selected_source_frame"] == 2
    assert admission["generic_parameter_sweep_forbidden"] is True

    gate = lock["reserved_r2_reject_promote_conditions"]
    assert gate["minimum_recovered_gt"] >= 2
    assert gate["lost_gt_must_equal"] == 0
    assert gate["net_tp_gain_minimum"] >= 2
    assert gate["minimum_recovered_gt_per_added_detector_call"] >= 0.02
    assert gate["maximum_absolute_precision_drop"] <= 0.005
    assert gate["maximum_fp_increase"] <= 2
    assert gate["all_accepted_enhanced_measurements_must_be_raw_corroborated"] is True
    assert gate["failure_action"] == "REJECT_WITHOUT_FULL5000"
    assert gate["success_action"] == "PROMOTE_TO_FULL_REQUEST_ONLY_NO_AUTOMATIC_FULL5000"
