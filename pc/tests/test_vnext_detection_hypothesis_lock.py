import copy
import json
from pathlib import Path

import pytest

from spectratrack.research.vnext_detection_hypothesis_lock import (
    EXPECTED_CANDIDATE_ID,
    EXPECTED_CLUSTER_IDS,
    EXPECTED_DISCOVERY_DIGEST,
    EXPECTED_LOCK_DIGEST,
    canonical_lock_digest,
    load_lock,
    validate_lock,
)


LOCK_PATH = (
    Path(__file__).parents[1]
    / "benchmarks"
    / "vnext"
    / "detection"
    / "r2_hypothesis_lock.v1.json"
)


def test_committed_r2_hypothesis_lock_is_exact_and_valid():
    lock = load_lock(LOCK_PATH)
    result = validate_lock(lock)

    assert result["valid"] is True
    assert result["candidate_id"] == EXPECTED_CANDIDATE_ID
    assert result["lock_digest"] == EXPECTED_LOCK_DIGEST
    assert canonical_lock_digest(lock) == EXPECTED_LOCK_DIGEST
    assert result["discovery_report_digest"] == EXPECTED_DISCOVERY_DIGEST
    assert result["r2_inspected"] is False
    assert result["r2_scored"] is False


def test_lock_mutation_is_detected_even_if_payload_remains_valid_json():
    lock = load_lock(LOCK_PATH)
    mutated = copy.deepcopy(lock)
    mutated["algorithm"]["microtile"]["source_size_px"] = 513

    assert canonical_lock_digest(mutated) != EXPECTED_LOCK_DIGEST
    with pytest.raises(ValueError, match="payload does not match frozen digest"):
        validate_lock(mutated)


def test_rehashing_a_mutation_cannot_silently_replace_the_frozen_lock():
    lock = load_lock(LOCK_PATH)
    mutated = copy.deepcopy(lock)
    mutated["algorithm"]["grouping"]["pair_relation"]["iou_at_least"] = 0.20
    mutated["lock_digest"] = canonical_lock_digest(mutated)

    with pytest.raises(ValueError, match="committed R2 hypothesis lock digest changed"):
        validate_lock(mutated)


def test_lock_preserves_control_thresholds_and_forbids_global_tile512():
    lock = load_lock(LOCK_PATH)
    control = lock["control_contract"]
    algorithm = lock["algorithm"]

    assert control["conf"] == 0.35
    assert control["decoder_iou"] == 0.45
    assert control["person_conf"] == 0.12
    assert control["tile_size"] == 640
    assert control["fusion_method"] == "hard-nms"
    assert control["enhancement"] == "off"
    assert algorithm["global_smaller_tiles"] is False
    assert algorithm["inference"]["conf"] == control["conf"]
    assert algorithm["inference"]["decoder_iou"] == control["decoder_iou"]
    assert algorithm["inference"]["person_conf"] == control["person_conf"]
    assert algorithm["inference"]["enhancement"] == "off"
    assert algorithm["admission"]["score_floor"] == control["person_conf"]
    assert algorithm["admission"]["new_confidence_threshold"] is False
    assert algorithm["admission"]["final_merge_method"] == "hard-nms"


def test_lock_binds_a7_lineage_and_records_no_r2_or_holdout_exposure():
    lock = load_lock(LOCK_PATH)
    lineage = lock["inspired_by"]
    scope = lock["evidence_scope"]

    assert lineage["discovery_report_digest"] == EXPECTED_DISCOVERY_DIGEST
    assert tuple(lineage["cluster_ids"]) == EXPECTED_CLUSTER_IDS
    assert scope["r2_inspected"] is False
    assert scope["r2_scored"] is False
    assert scope["holdout4200_inspected"] is False
    assert scope["holdout4200_scored"] is False
    assert "nightowls-public-smoke400-r2 ground truth" in scope["forbidden"]
    assert "nightowls-public-holdout4200-r1 ground truth" in scope["forbidden"]


def test_lock_compute_budget_and_stage_gates_are_immutable():
    lock = load_lock(LOCK_PATH)
    budget = lock["compute_budget"]
    stage = lock["next_stage_policy"]
    promote = lock["r2_decision_gates"]["promote_to_holdout4200"]
    reject = lock["r2_decision_gates"]["invalid_or_hard_reject"]

    assert budget["max_extra_onnx_calls_per_frame"] == 1
    assert budget["max_rescue_rois_per_frame"] == 1
    assert budget["no_second_rescue_pass"] is True
    assert stage["r2_is_first_eligible_blinded_evidence"] is True
    assert stage["holdout4200_only_after_r2_promotion"] is True
    assert stage["no_full5000_run_for_this_lock"] is True
    assert promote["min_recall_delta"] == 0.01
    assert promote["min_f1_delta"] == 0.005
    assert reject["precision_delta_at_most"] == -0.01
    assert reject["extra_onnx_calls_per_frame_above"] == 1


def test_lock_json_has_no_accidental_candidate_result_payload():
    lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))

    assert "results" not in lock
    assert "r2_metrics" not in lock
    assert "holdout4200_metrics" not in lock
    assert lock["status"] == "LOCKED_NO_R2_INSPECTION_OR_SCORING"
