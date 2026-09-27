from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

LOCK_SCHEMA = "spectratrack-a1-r2-hypothesis-lock-v1"
EXPECTED_CANDIDATE_ID = "a1-r2-conditional-crowd-microtile512-v1"
EXPECTED_LOCK_DIGEST = "sha256:350d92339030c2f1fd781951f25c13f1b96aec5f0a607e3796df3ebf4e6e6cfc"
EXPECTED_DISCOVERY_DIGEST = "sha256:ad51f9b7c3f4629155298215a52004092ea1b95bd220c4ed6a608e464971afec"
EXPECTED_DISCOVERY_COMMIT = "9773947b06edad1c44c20773092f75c212ad517b"
EXPECTED_CLUSTER_IDS = (
    "baseline:low_contrast_night_darkness",
    "baseline:missed_small_person",
    "baseline:occlusion_crowd",
    "a1_r1:prefusion_duplicate_pressure",
)


def canonical_lock_digest(lock: dict[str, Any]) -> str:
    payload = dict(lock)
    payload.pop("lock_digest", None)
    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(canonical).hexdigest()


def load_lock(path: str | Path) -> dict[str, Any]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("hypothesis lock must be a JSON object")
    return data


def validate_lock(lock: dict[str, Any]) -> dict[str, Any]:
    if lock.get("schema") != LOCK_SCHEMA:
        raise ValueError(f"lock schema must be {LOCK_SCHEMA}")
    if lock.get("candidate_id") != EXPECTED_CANDIDATE_ID:
        raise ValueError("unexpected R2 candidate id")
    if lock.get("version") != 1:
        raise ValueError("R2 hypothesis lock version must be 1")
    if lock.get("status") != "LOCKED_NO_R2_INSPECTION_OR_SCORING":
        raise ValueError("R2 hypothesis lock status is not frozen")
    if lock.get("lock_digest") != EXPECTED_LOCK_DIGEST:
        raise ValueError("committed R2 hypothesis lock digest changed")
    if canonical_lock_digest(lock) != EXPECTED_LOCK_DIGEST:
        raise ValueError("R2 hypothesis lock payload does not match frozen digest")

    scope = lock.get("evidence_scope")
    if not isinstance(scope, dict):
        raise ValueError("evidence_scope must be an object")
    for flag in (
        "r2_inspected",
        "r2_scored",
        "holdout4200_inspected",
        "holdout4200_scored",
    ):
        if scope.get(flag) is not False:
            raise ValueError(f"{flag} must remain false in the preregistration lock")

    lineage = lock.get("inspired_by")
    if not isinstance(lineage, dict):
        raise ValueError("inspired_by must be an object")
    if lineage.get("discovery_commit") != EXPECTED_DISCOVERY_COMMIT:
        raise ValueError("A7 discovery commit mismatch")
    if lineage.get("discovery_report_digest") != EXPECTED_DISCOVERY_DIGEST:
        raise ValueError("A7 discovery digest mismatch")
    if tuple(lineage.get("cluster_ids", ())) != EXPECTED_CLUSTER_IDS:
        raise ValueError("A7 discovery cluster lineage changed")

    control = lock.get("control_contract")
    if not isinstance(control, dict):
        raise ValueError("control_contract must be an object")
    expected_control = {
        "input_size": 960,
        "conf": 0.35,
        "decoder_iou": 0.45,
        "person_conf": 0.12,
        "tile_size": 640,
        "tile_overlap": 0.2,
        "fusion_method": "hard-nms",
        "fusion_iou": 0.55,
        "enhancement": "off",
        "production_behavior_changed": False,
    }
    for key, value in expected_control.items():
        if control.get(key) != value:
            raise ValueError(f"control contract changed: {key}")

    algorithm = lock.get("algorithm")
    if not isinstance(algorithm, dict):
        raise ValueError("algorithm must be an object")
    if algorithm.get("global_smaller_tiles") is not False:
        raise ValueError("global smaller-tile policy is forbidden")
    if algorithm.get("max_trigger_groups_per_frame") != 1:
        raise ValueError("only one trigger group per frame is allowed")
    microtile = algorithm.get("microtile")
    if not isinstance(microtile, dict) or microtile.get("source_size_px") != 512:
        raise ValueError("microtile source size must stay locked at 512")
    inference = algorithm.get("inference")
    if not isinstance(inference, dict):
        raise ValueError("algorithm inference settings are missing")
    for key in ("conf", "decoder_iou", "person_conf", "enhancement"):
        if inference.get(key) != control.get(key):
            raise ValueError(f"candidate changed locked detector setting: {key}")
    if inference.get("max_extra_onnx_calls_per_frame") != 1:
        raise ValueError("extra detector-call budget changed")
    if inference.get("repeat_rescue_on_same_frame") is not False:
        raise ValueError("second rescue pass is forbidden")

    admission = algorithm.get("admission")
    if not isinstance(admission, dict):
        raise ValueError("admission contract is missing")
    if admission.get("new_confidence_threshold") is not False:
        raise ValueError("generic confidence-threshold search is forbidden")
    if admission.get("score_floor") != control.get("person_conf"):
        raise ValueError("rescue score floor must equal the frozen control person threshold")
    if admission.get("final_merge_method") != "hard-nms":
        raise ValueError("previously rejected fusion candidates must not be revived")
    if admission.get("final_merge_iou") != control.get("fusion_iou"):
        raise ValueError("final fusion IoU must remain unchanged")

    budget = lock.get("compute_budget")
    if not isinstance(budget, dict):
        raise ValueError("compute_budget must be an object")
    if budget.get("max_extra_onnx_calls_per_frame") != 1:
        raise ValueError("compute budget exceeds one extra detector call per frame")
    if budget.get("max_rescue_rois_per_frame") != 1:
        raise ValueError("compute budget exceeds one rescue ROI per frame")
    if budget.get("no_second_rescue_pass") is not True:
        raise ValueError("second rescue pass must remain disabled")

    stage = lock.get("next_stage_policy")
    if not isinstance(stage, dict):
        raise ValueError("next_stage_policy must be an object")
    required_stage_flags = {
        "r2_is_first_eligible_blinded_evidence": True,
        "holdout4200_only_after_r2_promotion": True,
        "holdout4200_is_final_independent_nightowls_evidence_for_cycle": True,
        "no_r1_rescoring": True,
        "no_full5000_run_for_this_lock": True,
    }
    for key, value in required_stage_flags.items():
        if stage.get(key) is not value:
            raise ValueError(f"stage policy changed: {key}")

    gates = lock.get("r2_decision_gates")
    if not isinstance(gates, dict):
        raise ValueError("R2 decision gates must be frozen")
    if not isinstance(gates.get("invalid_or_hard_reject"), dict):
        raise ValueError("hard-reject gate is missing")
    if not isinstance(gates.get("promote_to_holdout4200"), dict):
        raise ValueError("promotion gate is missing")

    return {
        "valid": True,
        "schema": LOCK_SCHEMA,
        "candidate_id": EXPECTED_CANDIDATE_ID,
        "lock_digest": EXPECTED_LOCK_DIGEST,
        "discovery_report_digest": EXPECTED_DISCOVERY_DIGEST,
        "r2_inspected": False,
        "r2_scored": False,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate the frozen A1 R2 hypothesis lock")
    parser.add_argument("--lock", required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    result = validate_lock(load_lock(args.lock))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
