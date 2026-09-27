import json
from pathlib import Path

import pytest

from spectratrack.research_stage_policy import (
    CHARACTERIZATION_STAGE,
    EXTERNAL_STAGE,
    HISTORY_SCHEMA,
    POLICY_SCHEMA,
    RESULT_SCHEMA,
    STOP_STAGE,
    main,
    validate_history,
)

ROOT = Path(__file__).parents[1]
POLICY_PATH = ROOT / "benchmarks" / "vnext" / "smoke_stats" / "research_stage_policy.v1.json"
FIXTURE_PATH = ROOT / "benchmarks" / "vnext" / "smoke_stats" / "fixtures" / "research_stage_legal_history.json"


def _policy():
    return json.loads(POLICY_PATH.read_text(encoding="utf-8"))


def _published_policy():
    policy = _policy()
    for stage, token in (("r2", "d"), ("holdout4200", "e")):
        item = next(row for row in policy["stages"] if row["stage"] == stage)
        item["dataset"]["status"] = "published"
        item["dataset"]["identity_sha256"] = token * 64
    return policy


def _identity(policy, stage):
    return next(row for row in policy["stages"] if row["stage"] == stage)["dataset"]["identity_sha256"]


def _exposure(stage, when, token):
    return {"stage": stage, "exposed_at": when, "evidence_sha256": token * 64}


def _evaluation(policy, stage, when, spec="1", decision="PROMOTE_TO_NEXT", token="8"):
    return {
        "stage": stage,
        "scored_at": when,
        "candidate_spec_sha256": spec * 64,
        "dataset_identity_sha256": _identity(policy, stage),
        "decision": decision,
        "evidence_sha256": token * 64,
    }


def _candidate(
    candidate_id,
    frozen_at,
    allowed,
    *,
    spec="1",
    semantic="2",
    exposure=None,
    inspired=None,
    evaluations=None,
    parent=None,
    parent_spec=None,
    change_kind="research_candidate",
    performance_exception=None,
):
    result = {
        "candidate_id": candidate_id,
        "parent_candidate_id": parent,
        "parent_candidate_spec_sha256": parent_spec,
        "candidate_spec_sha256": spec * 64,
        "semantic_config_sha256": semantic * 64,
        "hypothesis_frozen_at": frozen_at,
        "freeze_provenance": {
            "record_sha256": "3" * 64,
            "source_revision": "fixture-revision",
            "source_path": f"fixtures/{candidate_id}.json",
        },
        "data_exposure": exposure or [],
        "inspired_by_stages": inspired or [],
        "change_kind": change_kind,
        "allowed_next_evaluation_stage": allowed,
        "evaluations": evaluations or [],
    }
    if performance_exception is not None:
        result["performance_exception"] = performance_exception
    return result


def _history(candidates, exposures):
    return {
        "schema": HISTORY_SCHEMA,
        "policy_schema": POLICY_SCHEMA,
        "research_cycle_id": "nightowls-r1-r2-holdout-v1",
        "research_exposure_log": exposures,
        "candidates": candidates,
    }


def test_pre_r1_frozen_candidate_can_triage_on_r1_then_advance():
    policy = _policy()
    candidate = _candidate(
        "pre-r1",
        "2026-09-27T09:00:00Z",
        "r2",
        evaluations=[_evaluation(policy, "r1", "2026-09-27T09:30:00Z")],
    )
    result = validate_history(
        _history([candidate], [_exposure("r1", "2026-09-27T10:00:00Z", "4")]),
        policy,
    )
    assert result["schema"] == RESULT_SCHEMA
    assert result["candidates"][0]["allowed_next_evaluation_stage"] == "r2"


def test_clear_reject_stops_same_candidate():
    policy = _policy()
    candidate = _candidate(
        "rejected",
        "2026-09-27T09:00:00Z",
        STOP_STAGE,
        evaluations=[_evaluation(policy, "r1", "2026-09-27T09:30:00Z", decision="CLEAR_REJECT")],
    )
    result = validate_history(
        _history([candidate], [_exposure("r1", "2026-09-27T10:00:00Z", "4")]),
        policy,
    )
    assert result["candidates"][0]["allowed_next_evaluation_stage"] == STOP_STAGE


def test_ambiguous_is_not_positive_claim_but_may_advance_for_more_evidence():
    policy = _policy()
    candidate = _candidate(
        "ambiguous",
        "2026-09-27T09:00:00Z",
        "r2",
        evaluations=[_evaluation(policy, "r1", "2026-09-27T09:30:00Z", decision="AMBIGUOUS")],
    )
    result = validate_history(
        _history([candidate], [_exposure("r1", "2026-09-27T10:00:00Z", "4")]),
        policy,
    )
    assert result["candidates"][0]["allowed_next_evaluation_stage"] == "r2"
    assert result["claims"]["p_values_used_for_promotion"] is False


def test_candidate_frozen_after_r1_must_disclose_r1_exposure():
    policy = _policy()
    candidate = _candidate("late", "2026-09-27T11:00:00Z", "r2")
    history = _history([candidate], [_exposure("r1", "2026-09-27T10:00:00Z", "4")])
    with pytest.raises(ValueError, match="omits exposed data stages"):
        validate_history(history, policy)


def test_r1_inspired_candidate_cannot_reuse_r1():
    policy = _policy()
    candidate = _candidate(
        "r1-tuned",
        "2026-09-27T11:00:00Z",
        "r2",
        exposure=["r1"],
        inspired=["r1"],
        evaluations=[_evaluation(policy, "r1", "2026-09-27T11:30:00Z")],
    )
    history = _history([candidate], [_exposure("r1", "2026-09-27T10:00:00Z", "4")])
    with pytest.raises(ValueError, match="expected r2, got r1"):
        validate_history(history, policy)


def test_r1_inspired_candidate_can_use_published_r2():
    policy = _published_policy()
    candidate = _candidate(
        "r1-tuned",
        "2026-09-27T11:00:00Z",
        "holdout4200",
        exposure=["r1"],
        inspired=["r1"],
        evaluations=[_evaluation(policy, "r2", "2026-09-27T11:30:00Z")],
    )
    history = _history(
        [candidate],
        [
            _exposure("r1", "2026-09-27T10:00:00Z", "4"),
            _exposure("r2", "2026-09-27T12:00:00Z", "5"),
        ],
    )
    result = validate_history(history, policy)
    assert result["candidates"][0]["allowed_next_evaluation_stage"] == "holdout4200"


def test_r2_inspired_candidate_next_eligible_stage_is_holdout():
    policy = _published_policy()
    candidate = _candidate(
        "r2-tuned",
        "2026-09-27T13:00:00Z",
        EXTERNAL_STAGE,
        exposure=["r1", "r2"],
        inspired=["r2"],
        evaluations=[_evaluation(policy, "holdout4200", "2026-09-27T13:30:00Z")],
    )
    history = _history(
        [candidate],
        [
            _exposure("r1", "2026-09-27T10:00:00Z", "4"),
            _exposure("r2", "2026-09-27T12:00:00Z", "5"),
            _exposure("holdout4200", "2026-09-27T14:00:00Z", "6"),
        ],
    )
    result = validate_history(history, policy)
    assert result["candidates"][0]["allowed_next_evaluation_stage"] == EXTERNAL_STAGE


def test_holdout_exposure_forces_new_external_corpus_for_further_tuning():
    policy = _published_policy()
    candidate = _candidate(
        "post-holdout",
        "2026-09-27T15:00:00Z",
        EXTERNAL_STAGE,
        exposure=["r1", "r2", "holdout4200"],
        inspired=["holdout4200"],
    )
    history = _history(
        [candidate],
        [
            _exposure("r1", "2026-09-27T10:00:00Z", "4"),
            _exposure("r2", "2026-09-27T12:00:00Z", "5"),
            _exposure("holdout4200", "2026-09-27T14:00:00Z", "6"),
        ],
    )
    result = validate_history(history, policy)
    assert result["candidates"][0]["allowed_next_evaluation_stage"] == EXTERNAL_STAGE


def test_full5000_is_characterization_only_not_promotion_evidence():
    policy = _policy()
    candidate = _candidate(
        "char",
        "2026-09-27T09:00:00Z",
        "r1",
        evaluations=[
            {
                "stage": CHARACTERIZATION_STAGE,
                "purpose": "aggregate_characterization",
                "decision": None,
                "candidate_spec_sha256": "1" * 64,
                "evidence_sha256": "7" * 64,
            }
        ],
    )
    result = validate_history(_history([candidate], []), policy)
    assert result["candidates"][0]["full5000_independent_heldout"] is False


def test_full5000_cannot_carry_promotion_decision():
    policy = _policy()
    candidate = _candidate(
        "bad-char",
        "2026-09-27T09:00:00Z",
        "r1",
        evaluations=[
            {
                "stage": CHARACTERIZATION_STAGE,
                "purpose": "aggregate_characterization",
                "decision": "PROMOTE_TO_NEXT",
                "candidate_spec_sha256": "1" * 64,
                "evidence_sha256": "7" * 64,
            }
        ],
    )
    with pytest.raises(ValueError, match="non-promotional"):
        validate_history(_history([candidate], []), policy)


def test_repeated_peeking_same_stage_is_rejected():
    policy = _policy()
    first = _evaluation(policy, "r1", "2026-09-27T09:20:00Z")
    second = _evaluation(policy, "r1", "2026-09-27T09:30:00Z", token="9")
    candidate = _candidate("peek", "2026-09-27T09:00:00Z", "r2", evaluations=[first, second])
    with pytest.raises(ValueError, match="repeated peeking"):
        validate_history(
            _history([candidate], [_exposure("r1", "2026-09-27T10:00:00Z", "4")]),
            policy,
        )


def test_parameter_mutation_after_freeze_is_rejected():
    policy = _policy()
    evaluation = _evaluation(policy, "r1", "2026-09-27T09:30:00Z", spec="9")
    candidate = _candidate("mutated", "2026-09-27T09:00:00Z", "r2", evaluations=[evaluation])
    with pytest.raises(ValueError, match="parameter mutation"):
        validate_history(
            _history([candidate], [_exposure("r1", "2026-09-27T10:00:00Z", "4")]),
            policy,
        )


def test_stage_skipping_is_rejected():
    policy = _published_policy()
    candidate = _candidate(
        "skip",
        "2026-09-27T09:00:00Z",
        "holdout4200",
        evaluations=[_evaluation(policy, "r2", "2026-09-27T09:30:00Z")],
    )
    history = _history(
        [candidate],
        [
            _exposure("r1", "2026-09-27T10:00:00Z", "4"),
            _exposure("r2", "2026-09-27T11:00:00Z", "5"),
        ],
    )
    with pytest.raises(ValueError, match="expected r1, got r2"):
        validate_history(history, policy)


def test_lineage_without_freeze_provenance_is_rejected():
    policy = _policy()
    candidate = _candidate("no-freeze", "2026-09-27T09:00:00Z", "r1")
    del candidate["freeze_provenance"]
    with pytest.raises(ValueError, match="freeze_provenance is required"):
        validate_history(_history([candidate], []), policy)


def test_parent_hash_mismatch_is_rejected():
    policy = _policy()
    parent = _candidate("parent", "2026-09-27T08:00:00Z", "r1")
    child = _candidate(
        "child",
        "2026-09-27T09:00:00Z",
        "r1",
        parent="parent",
        parent_spec="9" * 64,
    )
    with pytest.raises(ValueError, match="parent candidate spec hash"):
        validate_history(_history([parent, child], []), policy)


def test_unpublished_r2_cannot_be_scored():
    policy = _policy()
    r2 = next(row for row in policy["stages"] if row["stage"] == "r2")
    r2["dataset"]["status"] = "UNPUBLISHED"
    r2["dataset"]["identity_sha256"] = None
    candidate = _candidate(
        "r1-tuned",
        "2026-09-27T11:00:00Z",
        "holdout4200",
        exposure=["r1"],
        inspired=["r1"],
        evaluations=[
            {
                "stage": "r2",
                "scored_at": "2026-09-27T11:30:00Z",
                "candidate_spec_sha256": "1" * 64,
                "dataset_identity_sha256": "d" * 64,
                "decision": "PROMOTE_TO_NEXT",
                "evidence_sha256": "8" * 64,
            }
        ],
    )
    history = _history([candidate], [_exposure("r1", "2026-09-27T10:00:00Z", "4")])
    with pytest.raises(ValueError, match="not yet hash-bound"):
        validate_history(history, policy)


def test_committed_policy_binds_exact_a5_r1_r2_and_holdout_hashes():
    policy = _policy()
    stages = {row["stage"]: row["dataset"] for row in policy["stages"]}

    assert stages["r1"]["identity_sha256"] == "2a4da42830400b14405de01a82e3b3bee3d17a07db0be067f752545664fc6186"
    assert stages["r1"]["corpus_sha256"] == "70c29ecd91ede9239ebed2949ea46e4b07b63e842aef630c0752ee41b9620162"
    assert stages["r1"]["jsonl_sha256"] == "45ba10895c8a98f433bc7bad8e488b311a31ad8acf8459656d0cd775fe3fc0e8"

    assert stages["r2"]["status"] == "published"
    assert stages["r2"]["ground_truth_sha256"] == "75eba2c9dc3b36d0a2389bfbb080a709621ef690680ae655d85d8732e7bc6097"
    assert stages["r2"]["corpus_sha256"] == "9d145b4dda780052388f3b663203c519ded48f61457adef14654f84fb5549eff"
    assert stages["r2"]["manifest_sha256"] == "9270d46c2776aa531e1b979a1a7ebc16eaed0f6483b2c1095da834af383c4e83"
    assert stages["r2"]["selection_proof_sha256"] == "df11e9dc6ca1f63019cba071ed82420de27c706b41d1d3b46df880bb3ce50faf"
    assert stages["r2"]["identity_sha256"] == "9f2afa165f294a4bce1e7c842b8a135a98876e1a0c4db4b0cbd0b8c583151cce"

    assert stages["holdout4200"]["status"] == "published"
    assert stages["holdout4200"]["ground_truth_sha256"] == "cda56abf7bd48b9849d5b10467be7e192d0fea9abc3ae31fac5a2f180605c621"
    assert stages["holdout4200"]["corpus_sha256"] == "c2e5091fe4e1c146301f6d411c7a9c4d816380f94515e226b25e28e90dfc190a"
    assert stages["holdout4200"]["manifest_sha256"] == "19ea73abf8d97265b3b0a3e46657c09bac370c31835b7414886156994fefc901"
    assert stages["holdout4200"]["selection_proof_sha256"] == "4f1f8e977871ce3c134a4c72029035760f4377d3bddfd7a99cd3cebe568dab3d"
    assert stages["holdout4200"]["identity_sha256"] == "011242dbdc155e28dbae90e48cb2f6162abaf49ec793619936f86b78f41ebc9a"
    assert stages["r2"]["a5_source_commit"] == "a339da324ea0df682179097d7d6810b6fc250240"
    assert stages["holdout4200"]["a5_source_commit"] == "a339da324ea0df682179097d7d6810b6fc250240"


def test_p_values_are_rejected_as_promotion_inputs():
    policy = _policy()
    evaluation = _evaluation(policy, "r1", "2026-09-27T09:30:00Z")
    evaluation["p_value"] = 0.01
    candidate = _candidate("pvalue", "2026-09-27T09:00:00Z", "r2", evaluations=[evaluation])
    with pytest.raises(ValueError, match="p-values"):
        validate_history(
            _history([candidate], [_exposure("r1", "2026-09-27T10:00:00Z", "4")]),
            policy,
        )


def test_semantics_preserving_performance_exception_preserves_parent_freeze_eligibility():
    policy = _policy()
    parent = _candidate("parent", "2026-09-27T08:00:00Z", "r1")
    child = _candidate(
        "perf",
        "2026-09-27T11:00:00Z",
        "r1",
        spec="4",
        exposure=["r1"],
        parent="parent",
        parent_spec="1" * 64,
        change_kind="semantics_preserving_performance",
        performance_exception={
            "independent_of_quality_results": True,
            "equivalence_evidence_sha256": "6" * 64,
            "optimization_scope": "runtime_only",
        },
    )
    result = validate_history(
        _history([parent, child], [_exposure("r1", "2026-09-27T10:00:00Z", "4")]),
        policy,
    )
    assert result["candidates"][1]["performance_exception_applied"] is True
    assert result["candidates"][1]["allowed_next_evaluation_stage"] == "r1"


def test_performance_exception_rejects_semantic_parameter_change():
    policy = _policy()
    parent = _candidate("parent", "2026-09-27T08:00:00Z", "r1")
    child = _candidate(
        "perf",
        "2026-09-27T11:00:00Z",
        "r2",
        spec="4",
        semantic="9",
        exposure=["r1"],
        parent="parent",
        parent_spec="1" * 64,
        change_kind="semantics_preserving_performance",
        performance_exception={
            "independent_of_quality_results": True,
            "equivalence_evidence_sha256": "6" * 64,
            "optimization_scope": "runtime_only",
        },
    )
    with pytest.raises(ValueError, match="altered semantic_config_sha256"):
        validate_history(
            _history([parent, child], [_exposure("r1", "2026-09-27T10:00:00Z", "4")]),
            policy,
        )


def test_performance_exception_never_reuses_stage_parent_already_consumed():
    policy = _policy()
    parent = _candidate(
        "parent",
        "2026-09-27T08:00:00Z",
        "r2",
        evaluations=[_evaluation(policy, "r1", "2026-09-27T09:00:00Z")],
    )
    child = _candidate(
        "perf",
        "2026-09-27T11:00:00Z",
        "r2",
        spec="4",
        exposure=["r1"],
        parent="parent",
        parent_spec="1" * 64,
        change_kind="semantics_preserving_performance",
        performance_exception={
            "independent_of_quality_results": True,
            "equivalence_evidence_sha256": "6" * 64,
            "optimization_scope": "runtime_only",
        },
    )
    result = validate_history(
        _history([parent, child], [_exposure("r1", "2026-09-27T10:00:00Z", "4")]),
        policy,
    )
    assert result["candidates"][1]["allowed_next_evaluation_stage"] == "r2"


def test_committed_legal_history_fixture_validates():
    result = validate_history(
        json.loads(FIXTURE_PATH.read_text(encoding="utf-8")),
        _policy(),
    )
    assert result["valid"] is True


def test_cli_emits_deterministic_validation_report(tmp_path):
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    args = ["--input", str(FIXTURE_PATH), "--policy", str(POLICY_PATH)]
    assert main([*args, "--output", str(first)]) == 0
    assert main([*args, "--output", str(second)]) == 0
    assert first.read_bytes() == second.read_bytes()
