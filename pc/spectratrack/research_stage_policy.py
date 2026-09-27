from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

POLICY_SCHEMA = "spectratrack-research-stage-policy-v1"
HISTORY_SCHEMA = "spectratrack-research-stage-history-v1"
RESULT_SCHEMA = "spectratrack-research-stage-validation-v1"

INDEPENDENT_STAGES = ("r1", "r2", "holdout4200")
CHARACTERIZATION_STAGE = "full5000_characterization"
EXTERNAL_STAGE = "external_future_corpus"
STOP_STAGE = "STOP"
DECISIONS = {"CLEAR_REJECT", "PROMOTE_TO_NEXT", "AMBIGUOUS"}


def _sha256(value: Any, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(ch not in "0123456789abcdef" for ch in value):
        raise ValueError(f"{name} must be a lowercase 64-character SHA-256 hex digest")
    return value


def _timestamp(value: Any, name: str) -> datetime:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty ISO-8601 timestamp")
    text = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as exc:
        raise ValueError(f"{name} must be a valid ISO-8601 timestamp") from exc
    if parsed.tzinfo is None:
        raise ValueError(f"{name} must include a timezone")
    return parsed


def _stage_map(policy: dict[str, Any]) -> dict[str, dict[str, Any]]:
    if policy.get("schema") != POLICY_SCHEMA:
        raise ValueError(f"policy.schema must be {POLICY_SCHEMA}")
    stages = policy.get("stages")
    if not isinstance(stages, list):
        raise ValueError("policy.stages must be a list")
    result: dict[str, dict[str, Any]] = {}
    for item in stages:
        if not isinstance(item, dict) or not isinstance(item.get("stage"), str):
            raise ValueError("each policy stage must be an object with a stage")
        stage = item["stage"]
        if stage in result:
            raise ValueError(f"duplicate policy stage: {stage}")
        result[stage] = item
    for stage in (*INDEPENDENT_STAGES, CHARACTERIZATION_STAGE, EXTERNAL_STAGE):
        if stage not in result:
            raise ValueError(f"policy is missing stage {stage}")
    for stage in INDEPENDENT_STAGES:
        dataset = result[stage].get("dataset")
        if not isinstance(dataset, dict):
            raise ValueError(f"policy stage {stage} is missing dataset metadata")
        status = dataset.get("status")
        if status not in {"published", "UNPUBLISHED"}:
            raise ValueError(f"policy stage {stage} has unsupported dataset status")
        if status == "published":
            _sha256(dataset.get("identity_sha256"), f"policy stage {stage} identity_sha256")
    return result


def _freeze_provenance(candidate: dict[str, Any], candidate_id: str) -> None:
    provenance = candidate.get("freeze_provenance")
    if not isinstance(provenance, dict):
        raise ValueError(f"{candidate_id}: freeze_provenance is required")
    _sha256(provenance.get("record_sha256"), f"{candidate_id}.freeze_provenance.record_sha256")
    for field in ("source_revision", "source_path"):
        if not isinstance(provenance.get(field), str) or not provenance[field]:
            raise ValueError(f"{candidate_id}.freeze_provenance.{field} must be a non-empty string")


def _candidate_map(candidates: list[Any]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for candidate in candidates:
        if not isinstance(candidate, dict):
            raise ValueError("each candidate must be an object")
        candidate_id = candidate.get("candidate_id")
        if not isinstance(candidate_id, str) or not candidate_id:
            raise ValueError("candidate_id must be a non-empty string")
        if candidate_id in result:
            raise ValueError(f"duplicate candidate_id: {candidate_id}")
        result[candidate_id] = candidate
    return result


def _exposure_log(history: dict[str, Any]) -> dict[str, dict[str, Any]]:
    log = history.get("research_exposure_log")
    if not isinstance(log, list):
        raise ValueError("research_exposure_log must be a list")
    seen: dict[str, dict[str, Any]] = {}
    previous_time: datetime | None = None
    highest_independent = -1
    for index, item in enumerate(log):
        if not isinstance(item, dict):
            raise ValueError("research_exposure_log entries must be objects")
        stage = item.get("stage")
        if stage not in (*INDEPENDENT_STAGES, CHARACTERIZATION_STAGE):
            raise ValueError(f"research_exposure_log[{index}].stage is unsupported")
        if stage in seen:
            raise ValueError(f"repeated peeking: research stage {stage} appears more than once")
        exposed_at = _timestamp(item.get("exposed_at"), f"research_exposure_log[{index}].exposed_at")
        if previous_time is not None and exposed_at <= previous_time:
            raise ValueError("research_exposure_log must be in strictly increasing timestamp order")
        previous_time = exposed_at
        if stage in INDEPENDENT_STAGES:
            position = INDEPENDENT_STAGES.index(stage)
            if position > highest_independent + 1:
                raise ValueError(f"stage skipping in research exposure log: {stage}")
            highest_independent = max(highest_independent, position)
        _sha256(item.get("evidence_sha256"), f"research_exposure_log[{index}].evidence_sha256")
        seen[stage] = {"timestamp": exposed_at, **item}
    return seen


def _required_exposure_at_freeze(
    frozen_at: datetime,
    exposure_log: dict[str, dict[str, Any]],
) -> set[str]:
    return {
        stage
        for stage, item in exposure_log.items()
        if stage in INDEPENDENT_STAGES and item["timestamp"] <= frozen_at
    }


def _next_stage(blocked: set[str]) -> str:
    for stage in INDEPENDENT_STAGES:
        if stage not in blocked:
            return stage
    return EXTERNAL_STAGE


def _performance_exception(
    candidate: dict[str, Any],
    parent: dict[str, Any] | None,
) -> bool:
    if candidate.get("change_kind") != "semantics_preserving_performance":
        return False
    candidate_id = candidate["candidate_id"]
    if parent is None:
        raise ValueError(f"{candidate_id}: performance exception requires a parent candidate")
    if candidate.get("semantic_config_sha256") != parent.get("semantic_config_sha256"):
        raise ValueError(f"{candidate_id}: performance-only change altered semantic_config_sha256")
    if candidate.get("inspired_by_stages") != []:
        raise ValueError(f"{candidate_id}: performance exception requires inspired_by_stages=[]")
    exception = candidate.get("performance_exception")
    if not isinstance(exception, dict) or exception.get("independent_of_quality_results") is not True:
        raise ValueError(f"{candidate_id}: performance exception must attest independent_of_quality_results=true")
    _sha256(
        exception.get("equivalence_evidence_sha256"),
        f"{candidate_id}.performance_exception.equivalence_evidence_sha256",
    )
    if exception.get("optimization_scope") not in {"runtime_only", "memory_only", "serialization_only"}:
        raise ValueError(f"{candidate_id}: unsupported semantics-preserving optimization_scope")
    return True


def validate_history(
    history: dict[str, Any],
    policy: dict[str, Any],
) -> dict[str, Any]:
    stage_map = _stage_map(policy)
    if history.get("schema") != HISTORY_SCHEMA:
        raise ValueError(f"history.schema must be {HISTORY_SCHEMA}")
    if history.get("policy_schema") != POLICY_SCHEMA:
        raise ValueError(f"history.policy_schema must be {POLICY_SCHEMA}")
    if history.get("research_cycle_id") != policy.get("research_cycle_id"):
        raise ValueError("history research_cycle_id does not match policy")
    candidates_raw = history.get("candidates")
    if not isinstance(candidates_raw, list) or not candidates_raw:
        raise ValueError("candidates must be a non-empty list")
    candidates = _candidate_map(candidates_raw)
    exposure_log = _exposure_log(history)

    processed: dict[str, dict[str, Any]] = {}
    summaries: list[dict[str, Any]] = []

    for candidate in candidates_raw:
        candidate_id = candidate["candidate_id"]
        parent_id = candidate.get("parent_candidate_id")
        parent: dict[str, Any] | None = None
        parent_summary: dict[str, Any] | None = None
        if parent_id is not None:
            if not isinstance(parent_id, str) or parent_id not in candidates:
                raise ValueError(f"{candidate_id}: parent_candidate_id does not exist")
            if parent_id not in processed:
                raise ValueError(f"{candidate_id}: parent must appear earlier in candidates")
            parent = candidates[parent_id]
            parent_summary = processed[parent_id]
            if candidate.get("parent_candidate_spec_sha256") != parent.get("candidate_spec_sha256"):
                raise ValueError(f"{candidate_id}: parent candidate spec hash does not match lineage provenance")
        elif candidate.get("parent_candidate_spec_sha256") is not None:
            raise ValueError(f"{candidate_id}: root candidate cannot declare parent_candidate_spec_sha256")

        spec_sha = _sha256(candidate.get("candidate_spec_sha256"), f"{candidate_id}.candidate_spec_sha256")
        _sha256(candidate.get("semantic_config_sha256"), f"{candidate_id}.semantic_config_sha256")
        _freeze_provenance(candidate, candidate_id)
        frozen_at = _timestamp(candidate.get("hypothesis_frozen_at"), f"{candidate_id}.hypothesis_frozen_at")
        if parent is not None:
            parent_frozen = _timestamp(parent.get("hypothesis_frozen_at"), f"{parent_id}.hypothesis_frozen_at")
            if frozen_at <= parent_frozen:
                raise ValueError(f"{candidate_id}: child must be frozen after parent")

        data_exposure = candidate.get("data_exposure")
        if not isinstance(data_exposure, list) or any(stage not in INDEPENDENT_STAGES for stage in data_exposure):
            raise ValueError(f"{candidate_id}: data_exposure must contain only r1/r2/holdout4200")
        if len(data_exposure) != len(set(data_exposure)):
            raise ValueError(f"{candidate_id}: data_exposure contains duplicate stages")
        required_exposure = _required_exposure_at_freeze(frozen_at, exposure_log)
        missing = required_exposure.difference(data_exposure)
        extra = set(data_exposure).difference(required_exposure)
        if missing:
            raise ValueError(f"{candidate_id}: freeze provenance omits exposed data stages {sorted(missing)}")
        if extra:
            raise ValueError(f"{candidate_id}: data_exposure claims unexposed stages {sorted(extra)}")

        inspired = candidate.get("inspired_by_stages")
        if not isinstance(inspired, list) or any(stage not in INDEPENDENT_STAGES for stage in inspired):
            raise ValueError(f"{candidate_id}: inspired_by_stages must contain only independent NightOwls stages")
        if not set(inspired).issubset(set(data_exposure)):
            raise ValueError(f"{candidate_id}: inspired_by_stages must be a subset of data_exposure")
        if candidate.get("change_kind") not in {"research_candidate", "semantics_preserving_performance"}:
            raise ValueError(f"{candidate_id}: unsupported change_kind")

        performance_exception = _performance_exception(candidate, parent)
        inherited = set() if parent_summary is None else set(parent_summary["consumed_independent_stages"])
        effective_freeze_exposure = (
            set(parent_summary["effective_freeze_exposure"])
            if performance_exception and parent_summary is not None
            else set(data_exposure)
        )

        evaluations = candidate.get("evaluations")
        if not isinstance(evaluations, list):
            raise ValueError(f"{candidate_id}: evaluations must be a list")
        consumed = set(inherited)
        seen_stages: set[str] = set()
        latest_decision: str | None = None

        for index, evaluation in enumerate(evaluations):
            if not isinstance(evaluation, dict):
                raise ValueError(f"{candidate_id}: evaluation entries must be objects")
            stage = evaluation.get("stage")
            if stage == CHARACTERIZATION_STAGE:
                if evaluation.get("purpose") != "aggregate_characterization" or evaluation.get("decision") is not None:
                    raise ValueError(f"{candidate_id}: FULL5000 is non-promotional aggregate characterization only")
                if evaluation.get("candidate_spec_sha256") != spec_sha:
                    raise ValueError(f"{candidate_id}: parameter mutation detected in FULL5000 characterization")
                _sha256(evaluation.get("evidence_sha256"), f"{candidate_id}.evaluations[{index}].evidence_sha256")
                continue

            if stage not in INDEPENDENT_STAGES:
                raise ValueError(f"{candidate_id}: unsupported evaluation stage {stage}")
            if stage in seen_stages or stage in consumed:
                raise ValueError(f"{candidate_id}: repeated peeking or stage reuse at {stage}")
            if latest_decision == "CLEAR_REJECT":
                raise ValueError(f"{candidate_id}: evaluation continued after CLEAR_REJECT")
            expected_stage = _next_stage(consumed | effective_freeze_exposure)
            if stage != expected_stage:
                raise ValueError(f"{candidate_id}: stage skipping/reuse; expected {expected_stage}, got {stage}")

            dataset = stage_map[stage]["dataset"]
            if dataset.get("status") != "published":
                raise ValueError(f"{candidate_id}: stage {stage} is not yet hash-bound/published")
            if evaluation.get("dataset_identity_sha256") != dataset.get("identity_sha256"):
                raise ValueError(f"{candidate_id}: {stage} dataset identity does not match frozen policy")
            if evaluation.get("candidate_spec_sha256") != spec_sha:
                raise ValueError(f"{candidate_id}: parameter mutation detected at {stage}")
            decision = evaluation.get("decision")
            if decision not in DECISIONS:
                raise ValueError(f"{candidate_id}: invalid stage decision")
            if "p_value" in evaluation or "significance" in evaluation:
                raise ValueError(f"{candidate_id}: p-values/significance labels are not promotion inputs")
            scored_at = _timestamp(evaluation.get("scored_at"), f"{candidate_id}.evaluations[{index}].scored_at")
            if scored_at <= frozen_at:
                raise ValueError(f"{candidate_id}: {stage} scoring must occur after hypothesis freeze")
            _sha256(evaluation.get("evidence_sha256"), f"{candidate_id}.evaluations[{index}].evidence_sha256")

            seen_stages.add(stage)
            consumed.add(stage)
            latest_decision = decision

        computed_next = (
            STOP_STAGE
            if latest_decision == "CLEAR_REJECT"
            else _next_stage(consumed | effective_freeze_exposure)
        )
        if candidate.get("allowed_next_evaluation_stage") != computed_next:
            raise ValueError(
                f"{candidate_id}: allowed_next_evaluation_stage does not match computed {computed_next}"
            )

        summary = {
            "candidate_id": candidate_id,
            "allowed_next_evaluation_stage": computed_next,
            "consumed_independent_stages": [stage for stage in INDEPENDENT_STAGES if stage in consumed],
            "effective_freeze_exposure": [
                stage for stage in INDEPENDENT_STAGES if stage in effective_freeze_exposure
            ],
            "performance_exception_applied": performance_exception,
            "full5000_independent_heldout": False,
        }
        processed[candidate_id] = summary
        summaries.append(summary)

    return {
        "schema": RESULT_SCHEMA,
        "policy_schema": POLICY_SCHEMA,
        "research_cycle_id": policy["research_cycle_id"],
        "valid": True,
        "candidates": summaries,
        "claims": {
            "holdout4200_is_final_independent_nightowls_stage": True,
            "full5000_is_aggregate_characterization_only_after_component_exposure": True,
            "further_tuning_after_holdout_requires_external_future_corpus": True,
            "p_values_used_for_promotion": False,
        },
    }


def validation_report(
    history: dict[str, Any],
    policy: dict[str, Any],
) -> dict[str, Any]:
    try:
        return validate_history(history, policy)
    except ValueError as exc:
        return {
            "schema": RESULT_SCHEMA,
            "policy_schema": POLICY_SCHEMA,
            "research_cycle_id": history.get("research_cycle_id"),
            "valid": False,
            "errors": [str(exc)],
        }


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate SpectraTrack anti-leakage research-stage history")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)

    history = _load_json(args.input)
    policy = _load_json(args.policy)
    result = validation_report(history, policy)
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(payload, encoding="utf-8")
    else:
        print(payload, end="")
    return 0 if result["valid"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
