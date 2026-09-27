import json
from pathlib import Path

import pytest

from spectratrack.smoke_stats import (
    INPUT_SCHEMA,
    RESULT_SCHEMA,
    analyze,
    compute_budget,
    coverage_summary,
    main,
    paired_binary_benefit_interval,
    poisson_rate_interval,
    promotion_decision,
    rare_event_sensitivity,
    wilson_interval,
    zero_event_upper_probability,
)


def _selection(sample_frames=400, population_frames=5000):
    return {
        "corpus_revision": "fixture-corpus-r1",
        "corpus_sha256": "a" * 64,
        "selection_manifest_sha256": "b" * 64,
        "selection_identity_sha256": "c" * 64,
        "selection_method": "deterministic_outcome_independent_without_replacement",
        "inference_design": "simple_random_without_replacement",
        "population_frames": population_frames,
        "sample_frames": sample_frames,
        "frozen_before_candidate_scoring": True,
    }


def _primary(**overrides):
    metric = {
        "name": "person_recall",
        "role": "primary",
        "kind": "paired_binary_delta",
        "opportunities": 400,
        "improvements": 40,
        "regressions": 0,
        "minimum_material_improvement": 0.05,
        "maximum_tolerable_harm": 0.02,
    }
    metric.update(overrides)
    return metric


def _spec(metrics=None):
    return {
        "schema": INPUT_SCHEMA,
        "confidence": 0.95,
        "selection": _selection(),
        "population_composition": {"frames": 5000, "positive_frames": 600, "objects": 1000},
        "sample_composition": {"frames": 400, "positive_frames": 50, "objects": 82},
        "metrics": metrics or [_primary()],
    }


def test_wilson_and_poisson_handle_zero_denominators_without_division():
    assert wilson_interval(0, 0) == {"estimate": None, "lower": None, "upper": None}
    assert poisson_rate_interval(0, 0) == {"estimate": None, "lower": None, "upper": None}
    assert zero_event_upper_probability(0) is None


def test_zero_event_bound_is_conservative_and_shrinks_with_more_evidence():
    small = zero_event_upper_probability(40)
    smoke = zero_event_upper_probability(400)
    full = zero_event_upper_probability(5000)
    assert small is not None and smoke is not None and full is not None
    assert small > smoke > full > 0.0
    assert smoke == pytest.approx(0.007461, rel=0.01)


def test_paired_interval_zero_changes_is_not_false_certainty():
    result = paired_binary_benefit_interval(0, 0, 400)
    assert result["estimate"] == 0.0
    assert result["lower"] < 0.0
    assert result["upper"] > 0.0


def test_coverage_reports_expected_and_observed_without_claiming_independent_objects():
    result = coverage_summary(
        {"frames": 5000, "positive_frames": 600, "objects": 1000},
        {"frames": 400, "positive_frames": 50, "objects": 82},
    )
    assert result["positive_frames"]["expected"] == pytest.approx(48.0)
    assert result["objects"]["expected"] == pytest.approx(80.0)
    assert result["positive_frames"]["observed"] == 50
    assert result["objects"]["observed"] == 82
    assert "descriptive" in result["objects"]["poisson_style_note"]


def test_rare_event_capture_probability_is_monotone():
    result = rare_event_sensitivity(5000, 400)
    probabilities = result["event_frame_capture_probability"]
    ordered = [probabilities[str(count)] for count in (1, 2, 5, 10, 20, 50, 100)]
    assert ordered == sorted(ordered)
    assert probabilities["1"] == pytest.approx(0.08)
    thresholds = result["minimum_full_corpus_event_frames_for_capture_probability"]
    assert thresholds["50pct"] <= thresholds["80pct"] <= thresholds["90pct"] <= thresholds["95pct"]


def test_budget_model_has_expected_8_percent_smoke_and_break_even():
    result = compute_budget(5000, 400)
    assert result["smoke_fraction_of_full"] == pytest.approx(0.08)
    assert result["full_to_smoke_frame_ratio"] == pytest.approx(12.5)
    assert result["break_even_survival_rate"] == pytest.approx(0.92)
    scenarios = {row["candidate_survival_rate"]: row for row in result["scenarios"]}
    assert scenarios[0.0]["expected_savings_fraction"] == pytest.approx(0.92)
    assert scenarios[0.25]["expected_savings_fraction"] == pytest.approx(0.67)
    assert scenarios[0.5]["expected_savings_fraction"] == pytest.approx(0.42)
    assert scenarios[1.0]["expected_savings_fraction"] == pytest.approx(-0.08)


def test_clear_reject_when_primary_effect_is_confidently_too_small():
    result = promotion_decision([_primary(improvements=0, regressions=0)])
    assert result["category"] == "CLEAR_REJECT"
    assert result["metrics"][0]["state"] == "TOO_SMALL"
    assert result["final_benchmark_claim"] is False


def test_clear_reject_when_harm_exceeds_tolerance():
    result = promotion_decision([_primary(improvements=0, regressions=80)])
    assert result["category"] == "CLEAR_REJECT"
    assert result["metrics"][0]["state"] == "HARMFUL"


def test_promote_requires_material_effect_and_is_never_final_claim():
    result = promotion_decision([_primary(improvements=80, regressions=0)])
    assert result["category"] == "PROMOTE_TO_FULL"
    assert result["metrics"][0]["state"] == "PROMISING"
    assert result["full_corpus_required_for_acceptance"] is True
    assert result["final_benchmark_claim"] is False


def test_zero_rare_guardrail_counts_do_not_prove_safety():
    guardrail = {
        "name": "new_false_positive_rate",
        "role": "guardrail",
        "kind": "poisson_rate_delta",
        "baseline_count": 0,
        "candidate_count": 0,
        "exposure": 400,
        "direction": "lower_is_better",
        "maximum_tolerable_harm": 0.0,
    }
    result = promotion_decision([_primary(improvements=80), guardrail])
    states = {item["name"]: item["state"] for item in result["metrics"]}
    assert states["new_false_positive_rate"] == "AMBIGUOUS"
    assert result["category"] == "AMBIGUOUS"


def test_more_same_direction_evidence_can_resolve_ambiguity_to_promotion():
    small = promotion_decision(
        [_primary(opportunities=100, improvements=10, regressions=0, minimum_material_improvement=0.05)]
    )
    large = promotion_decision(
        [_primary(opportunities=1000, improvements=100, regressions=0, minimum_material_improvement=0.05)]
    )
    assert small["category"] == "AMBIGUOUS"
    assert large["category"] == "PROMOTE_TO_FULL"


def test_more_same_direction_harm_does_not_increase_confidence_in_candidate():
    small = promotion_decision(
        [_primary(opportunities=100, improvements=0, regressions=10, maximum_tolerable_harm=0.05)]
    )
    large = promotion_decision(
        [_primary(opportunities=1000, improvements=0, regressions=100, maximum_tolerable_harm=0.05)]
    )
    assert small["category"] == "AMBIGUOUS"
    assert large["category"] == "CLEAR_REJECT"


@pytest.mark.parametrize("opportunities", [1, 2, 5, 10, 50, 100, 400, 1000])
def test_interval_width_nonincreasing_for_replicated_zero_delta(opportunities):
    current = paired_binary_benefit_interval(0, 0, opportunities)
    doubled = paired_binary_benefit_interval(0, 0, opportunities * 2)
    current_width = current["upper"] - current["lower"]
    doubled_width = doubled["upper"] - doubled["lower"]
    assert doubled_width <= current_width


def test_analyze_requires_selection_frozen_before_candidate_scoring():
    spec = _spec()
    spec["selection"]["frozen_before_candidate_scoring"] = False
    with pytest.raises(ValueError, match="frozen before candidate scoring"):
        analyze(spec)


def test_analyze_rejects_composition_frame_mismatch():
    spec = _spec()
    spec["sample_composition"]["frames"] = 399
    with pytest.raises(ValueError, match="composition frame counts"):
        analyze(spec)


def test_analyze_emits_nonfinal_triage_contract():
    result = analyze(_spec())
    assert result["schema"] == RESULT_SCHEMA
    assert result["claims"] == {
        "smoke_is_triage_only": True,
        "smoke_pass_is_final_benchmark_claim": False,
        "authoritative_acceptance_requires_full_frozen_corpus": True,
        "simple_random_inference_supported": True,
    }


def test_cli_is_deterministic(tmp_path: Path):
    input_path = tmp_path / "input.json"
    first_path = tmp_path / "first.json"
    second_path = tmp_path / "second.json"
    input_path.write_text(json.dumps(_spec()), encoding="utf-8")
    assert main(["--input", str(input_path), "--output", str(first_path)]) == 0
    assert main(["--input", str(input_path), "--output", str(second_path)]) == 0
    assert first_path.read_bytes() == second_path.read_bytes()


def test_stratified_selection_marks_simple_random_calculations_as_reference_only():
    spec = _spec()
    spec["selection"]["inference_design"] = "deterministic_sequence_coverage_plus_hash_fill"
    result = analyze(spec)
    assert result["coverage"]["simple_random_reference_only"] is True
    assert result["rare_event_sensitivity"]["reference_only"] is True
    assert result["claims"]["simple_random_inference_supported"] is False


def test_composition_only_analysis_does_not_require_candidate_metrics():
    spec = _spec(metrics=[])
    spec["metrics"] = []
    result = analyze(spec)
    assert result["triage"] is None
    assert result["coverage"] is not None
    assert result["claims"]["smoke_is_triage_only"] is True


def test_committed_example_result_matches_module_output():
    root = Path(__file__).parents[1]
    input_path = root / "benchmarks" / "vnext" / "smoke_stats" / "fixtures" / "example_input.json"
    result_path = root / "benchmarks" / "vnext" / "smoke_stats" / "fixtures" / "example_result.json"
    expected = analyze(json.loads(input_path.read_text(encoding="utf-8")))
    actual = json.loads(result_path.read_text(encoding="utf-8"))
    assert actual == expected
