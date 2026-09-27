import copy
import json
from pathlib import Path

import pytest

from spectratrack.research.vnext_detection_fusion import FusionCandidate
from spectratrack.research.vnext_detection_r2_validation import (
    A6_POLICY_COMMIT,
    EXPECTED_CANDIDATE_ID,
    EXPECTED_LOCK_DIGEST,
    R2_CORPUS_SHA256,
    R2_FRAME_COUNT,
    R2_GT_SHA256,
    R2_MANIFEST_SHA256,
    R2_PROOF_SHA256,
    R2_REVISION,
    _canonical_digest,
    apply_prelocked_gates,
    microtile_region,
    select_trigger_group,
    validate_result_artifact,
)


def _candidate(
    bbox,
    *,
    source_id,
    score=0.4,
):
    return FusionCandidate(
        bbox,
        score,
        0,
        "person",
        "full" if source_id == "full" else "tile",
        source_id,
    )


def _metrics(
    *,
    tp=10,
    fp=4,
    fn=2,
    precision=0.714,
    recall=0.833,
    f1=0.769,
    bbox=0.72,
    center=0.03,
    duplicates=5,
):
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "bbox_localization_iou": bbox,
        "center_error_gt_diag_ratio": center,
        "duplicate_count_before_fusion": duplicates,
    }


def _clusters(*, small_fn=2, crowd_fn=2):
    return {
        "small_person_height_lt_48": {
            "ground_truth": 4,
            "tp": 4 - small_fn,
            "fn": small_fn,
            "recall": (4 - small_fn) / 4,
        },
        "occlusion_crowd_occluded_true": {
            "ground_truth": 3,
            "tp": 3 - crowd_fn,
            "fn": crowd_fn,
            "recall": (3 - crowd_fn) / 3,
        },
    }


def _compute(max_extra=1):
    return {
        "max_extra_onnx_calls_observed_per_frame": max_extra,
    }


def test_trigger_group_prefers_smallest_median_height_then_source_evidence():
    large = [
        _candidate((10, 10, 50, 110), source_id="full"),
        _candidate((12, 12, 52, 112), source_id="tile:0"),
    ]
    small = [
        _candidate((300, 30, 320, 70), source_id="full"),
        _candidate((302, 31, 322, 71), source_id="tile:4"),
        _candidate((303, 30, 323, 70), source_id="tile:5"),
    ]

    group = select_trigger_group([*large, *small])

    assert group is not None
    assert {item.source_id for item in group} == {"full", "tile:4", "tile:5"}


def test_trigger_requires_two_distinct_sources_and_never_global_tiles():
    same_source = [
        _candidate((20, 20, 40, 60), source_id="tile:1"),
        _candidate((21, 21, 41, 61), source_id="tile:1"),
    ]
    assert select_trigger_group(same_source) is None


def test_microtile_is_exactly_512_square_clamped_to_frame():
    group = (
        _candidate((0, 0, 20, 40), source_id="full"),
        _candidate((2, 2, 22, 42), source_id="tile:0"),
    )
    assert microtile_region(group, frame_width=1024, frame_height=640) == (
        0,
        0,
        512,
        512,
    )
    assert microtile_region(group, frame_width=500, frame_height=640) is None


def test_prelocked_gate_hard_reject_is_mechanical():
    control = _metrics()
    candidate = _metrics(
        tp=10,
        fp=14,
        fn=2,
        precision=0.60,
        recall=0.833,
        f1=0.70,
        duplicates=25,
    )
    result = apply_prelocked_gates(
        control=control,
        candidate=candidate,
        control_clusters=_clusters(),
        candidate_clusters=_clusters(),
        compute=_compute(),
    )
    assert result["decision"] == "REJECT_R2"
    assert "f1_delta_at_most_-0.005" in result["hard_reject_reasons"]
    assert "precision_delta_at_most_-0.01" in result["hard_reject_reasons"]


def test_prelocked_gate_promotes_only_when_every_condition_passes():
    control = _metrics(
        tp=80,
        fp=20,
        fn=20,
        precision=0.80,
        recall=0.80,
        f1=0.80,
        duplicates=20,
    )
    candidate = _metrics(
        tp=82,
        fp=21,
        fn=18,
        precision=0.796,
        recall=0.82,
        f1=0.808,
        bbox=0.715,
        center=0.035,
        duplicates=24,
    )
    result = apply_prelocked_gates(
        control=control,
        candidate=candidate,
        control_clusters=_clusters(small_fn=3, crowd_fn=2),
        candidate_clusters=_clusters(small_fn=3, crowd_fn=1),
        compute=_compute(),
    )
    assert result["decision"] == "PROMOTE_TO_HOLDOUT4200_REQUEST"
    assert all(result["promote_checks"].values())


def test_prelocked_gate_defaults_to_ambiguous():
    control = _metrics()
    candidate = _metrics(
        tp=10,
        fp=4,
        fn=2,
        precision=0.714,
        recall=0.838,
        f1=0.772,
        duplicates=5,
    )
    result = apply_prelocked_gates(
        control=control,
        candidate=candidate,
        control_clusters=_clusters(),
        candidate_clusters=_clusters(),
        compute=_compute(),
    )
    assert result["decision"] == "AMBIGUOUS_R2"
    assert result["hard_reject_reasons"] == []


def _valid_result():
    payload = {
        "schema": "spectratrack-a1-r2-result-v1",
        "candidate_id": EXPECTED_CANDIDATE_ID,
        "source_commit": "1" * 40,
        "lock_digest": EXPECTED_LOCK_DIGEST,
        "candidate_config_digest": "sha256:" + "2" * 64,
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
            "ci": 36317780050,
            "first_allowed_r2_exposure": True,
            "holdout4200_inspected": False,
            "holdout4200_scored": False,
            "full5000_run": False,
            "retuned_after_r2": False,
        },
        "model_sha256": "3" * 64,
        "production_control_changed": False,
        "control": {"metrics": {}, "clusters": {}},
        "candidate": {"metrics": {}, "clusters": {}},
        "compute": {},
        "gates": {},
        "decision": "AMBIGUOUS_R2",
        "provenance_digest": "sha256:" + "4" * 64,
    }
    payload["result_digest"] = _canonical_digest(payload)
    return payload


def test_result_validator_binds_exact_lock_and_r2_hashes():
    result = _valid_result()
    validated = validate_result_artifact(result)
    assert validated["valid"] is True
    assert validated["result_digest"] == result["result_digest"]

    wrong = copy.deepcopy(result)
    wrong["r2"]["ground_truth_sha256"] = "0" * 64
    wrong["result_digest"] = _canonical_digest(
        {key: value for key, value in wrong.items() if key != "result_digest"}
    )
    with pytest.raises(ValueError, match="ground_truth_sha256"):
        validate_result_artifact(wrong)


def test_result_validator_rejects_any_holdout_or_retune_exposure():
    result = _valid_result()
    result["a6_stage_policy"]["holdout4200_inspected"] = True
    result["result_digest"] = _canonical_digest(
        {key: value for key, value in result.items() if key != "result_digest"}
    )
    with pytest.raises(ValueError, match="holdout4200_inspected"):
        validate_result_artifact(result)


def test_committed_r2_result_binds_exact_frozen_lock_and_r2_provenance():
    result_path = (
        Path(__file__).parents[1]
        / "benchmarks"
        / "vnext"
        / "detection"
        / "r2_result.v1.json"
    )
    result = json.loads(result_path.read_text(encoding="utf-8"))
    validated = validate_result_artifact(result)

    assert validated["valid"] is True
    assert result["source_commit"] == "ea1234e153a689811488e439429472c5b2a698ea"
    assert result["lock_digest"] == EXPECTED_LOCK_DIGEST
    assert result["r2"] == {
        "revision": R2_REVISION,
        "frame_count": R2_FRAME_COUNT,
        "corpus_sha256": R2_CORPUS_SHA256,
        "ground_truth_sha256": R2_GT_SHA256,
        "manifest_sha256": R2_MANIFEST_SHA256,
        "selection_proof_sha256": R2_PROOF_SHA256,
    }
    assert result["a6_stage_policy"]["commit"] == A6_POLICY_COMMIT
    assert result["a6_stage_policy"]["holdout4200_inspected"] is False
    assert result["a6_stage_policy"]["holdout4200_scored"] is False
    assert result["a6_stage_policy"]["full5000_run"] is False
    assert result["a6_stage_policy"]["retuned_after_r2"] is False
    assert result["production_control_changed"] is False
    assert result["decision"] == "REJECT_R2"
    assert result["gates"]["hard_reject_reasons"] == [
        "prefusion_duplicate_increase_gate"
    ]
