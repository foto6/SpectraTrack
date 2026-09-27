from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import NormalDist
from typing import Any

RESULT_SCHEMA = "spectratrack-smoke-promotion-v1"
INPUT_SCHEMA = "spectratrack-smoke-promotion-input-v1"
DEFAULT_CONFIDENCE = 0.95


def _validate_confidence(confidence: float) -> float:
    value = float(confidence)
    if not 0.0 < value < 1.0:
        raise ValueError("confidence must be in (0, 1)")
    return value


def _non_negative_int(value: Any, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _positive_int(value: Any, name: str) -> int:
    result = _non_negative_int(value, name)
    if result == 0:
        raise ValueError(f"{name} must be greater than zero")
    return result


def _finite_non_negative(value: Any, name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return result


def _sha256(value: Any, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(ch not in "0123456789abcdef" for ch in value):
        raise ValueError(f"{name} must be a lowercase 64-character SHA-256 hex digest")
    return value


def _interval(estimate: float | None, lower: float | None, upper: float | None) -> dict[str, float | None]:
    return {"estimate": estimate, "lower": lower, "upper": upper}


def wilson_interval(successes: int, trials: int, confidence: float = DEFAULT_CONFIDENCE) -> dict[str, float | None]:
    successes = _non_negative_int(successes, "successes")
    trials = _non_negative_int(trials, "trials")
    confidence = _validate_confidence(confidence)
    if successes > trials:
        raise ValueError("successes cannot exceed trials")
    if trials == 0:
        return _interval(None, None, None)

    z = NormalDist().inv_cdf(0.5 + confidence / 2.0)
    p = successes / trials
    z2_over_n = (z * z) / trials
    denominator = 1.0 + z2_over_n
    center = (p + z2_over_n / 2.0) / denominator
    half = z * math.sqrt((p * (1.0 - p) / trials) + (z * z) / (4.0 * trials * trials)) / denominator
    return _interval(p, max(0.0, center - half), min(1.0, center + half))


def poisson_rate_interval(count: int, exposure: int, confidence: float = DEFAULT_CONFIDENCE) -> dict[str, float | None]:
    count = _non_negative_int(count, "count")
    exposure = _non_negative_int(exposure, "exposure")
    confidence = _validate_confidence(confidence)
    if exposure == 0:
        return _interval(None, None, None)

    z = NormalDist().inv_cdf(0.5 + confidence / 2.0)
    if count == 0:
        lower_count = 0.0
    else:
        base = max(0.0, 1.0 - 1.0 / (9.0 * count) - z / (3.0 * math.sqrt(count)))
        lower_count = count * base**3

    upper_k = count + 1
    upper_base = 1.0 - 1.0 / (9.0 * upper_k) + z / (3.0 * math.sqrt(upper_k))
    upper_count = upper_k * upper_base**3
    return _interval(count / exposure, lower_count / exposure, upper_count / exposure)


def zero_event_upper_probability(opportunities: int, confidence: float = DEFAULT_CONFIDENCE) -> float | None:
    opportunities = _non_negative_int(opportunities, "opportunities")
    confidence = _validate_confidence(confidence)
    if opportunities == 0:
        return None
    return 1.0 - (1.0 - confidence) ** (1.0 / opportunities)


def paired_binary_benefit_interval(
    improvements: int,
    regressions: int,
    opportunities: int,
    confidence: float = DEFAULT_CONFIDENCE,
) -> dict[str, Any]:
    improvements = _non_negative_int(improvements, "improvements")
    regressions = _non_negative_int(regressions, "regressions")
    opportunities = _non_negative_int(opportunities, "opportunities")
    if improvements + regressions > opportunities:
        raise ValueError("improvements + regressions cannot exceed opportunities")
    if opportunities == 0:
        return {**_interval(None, None, None), "improvements": improvements, "regressions": regressions}

    gain = wilson_interval(improvements, opportunities, confidence)
    loss = wilson_interval(regressions, opportunities, confidence)
    return {
        "estimate": (improvements - regressions) / opportunities,
        "lower": max(-1.0, float(gain["lower"]) - float(loss["upper"])),
        "upper": min(1.0, float(gain["upper"]) - float(loss["lower"])),
        "improvements": improvements,
        "regressions": regressions,
    }


def _oriented_delta(
    baseline: dict[str, float | None],
    candidate: dict[str, float | None],
    direction: str,
) -> dict[str, float | None]:
    if any(item is None for item in (*baseline.values(), *candidate.values())):
        return _interval(None, None, None)
    raw_estimate = float(candidate["estimate"]) - float(baseline["estimate"])
    raw_lower = float(candidate["lower"]) - float(baseline["upper"])
    raw_upper = float(candidate["upper"]) - float(baseline["lower"])
    if direction == "higher_is_better":
        return _interval(raw_estimate, raw_lower, raw_upper)
    if direction == "lower_is_better":
        return _interval(-raw_estimate, -raw_upper, -raw_lower)
    raise ValueError("direction must be higher_is_better or lower_is_better")


def binomial_benefit_interval(
    baseline_successes: int,
    candidate_successes: int,
    opportunities: int,
    direction: str,
    confidence: float = DEFAULT_CONFIDENCE,
) -> dict[str, float | None]:
    baseline = wilson_interval(baseline_successes, opportunities, confidence)
    candidate = wilson_interval(candidate_successes, opportunities, confidence)
    return _oriented_delta(baseline, candidate, direction)


def poisson_benefit_interval(
    baseline_count: int,
    candidate_count: int,
    exposure: int,
    direction: str,
    confidence: float = DEFAULT_CONFIDENCE,
) -> dict[str, float | None]:
    baseline = poisson_rate_interval(baseline_count, exposure, confidence)
    candidate = poisson_rate_interval(candidate_count, exposure, confidence)
    return _oriented_delta(baseline, candidate, direction)


def _hypergeometric_at_least_one(population: int, sample: int, event_frames: int) -> float:
    population = _positive_int(population, "population")
    sample = _non_negative_int(sample, "sample")
    event_frames = _non_negative_int(event_frames, "event_frames")
    if sample > population:
        raise ValueError("sample cannot exceed population")
    if event_frames > population:
        raise ValueError("event_frames cannot exceed population")
    if sample == 0 or event_frames == 0:
        return 0.0
    if population - event_frames < sample:
        return 1.0

    miss_probability = 1.0
    for index in range(sample):
        miss_probability *= (population - event_frames - index) / (population - index)
    return 1.0 - miss_probability


def rare_event_sensitivity(
    population_frames: int,
    sample_frames: int,
    inference_design: str = "simple_random_without_replacement",
) -> dict[str, Any]:
    checkpoints = (1, 2, 5, 10, 20, 50, 100)
    probabilities = {
        str(count): _hypergeometric_at_least_one(population_frames, sample_frames, count)
        for count in checkpoints
        if count <= population_frames
    }
    thresholds: dict[str, int | None] = {}
    for target in (0.5, 0.8, 0.9, 0.95):
        first = next(
            (
                event_frames
                for event_frames in range(1, population_frames + 1)
                if _hypergeometric_at_least_one(population_frames, sample_frames, event_frames) >= target
            ),
            None,
        )
        thresholds[f"{int(target * 100)}pct"] = first
    srs = inference_design == "simple_random_without_replacement"
    return {
        "event_frame_capture_probability": probabilities,
        "minimum_full_corpus_event_frames_for_capture_probability": thresholds,
        "reference_model": "simple_random_without_replacement",
        "reference_only": not srs,
        "selection_inference_design": inference_design,
        "assumption": (
            "exact under a simple random sample without replacement; otherwise a reference sensitivity model only"
        ),
    }


def coverage_summary(
    population: dict[str, Any],
    sample: dict[str, Any],
    inference_design: str = "simple_random_without_replacement",
) -> dict[str, Any]:
    population_frames = _positive_int(population["frames"], "population.frames")
    sample_frames = _positive_int(sample["frames"], "sample.frames")
    if sample_frames > population_frames:
        raise ValueError("sample.frames cannot exceed population.frames")

    population_positive = _non_negative_int(population["positive_frames"], "population.positive_frames")
    population_objects = _non_negative_int(population["objects"], "population.objects")
    sample_positive = _non_negative_int(sample["positive_frames"], "sample.positive_frames")
    sample_objects = _non_negative_int(sample["objects"], "sample.objects")
    if population_positive > population_frames or sample_positive > sample_frames:
        raise ValueError("positive frame count cannot exceed frame count")

    p = population_positive / population_frames
    expected_positive = sample_frames * p
    if population_frames > 1:
        fpc = (population_frames - sample_frames) / (population_frames - 1)
        positive_sd = math.sqrt(sample_frames * p * (1.0 - p) * fpc)
    else:
        positive_sd = 0.0

    expected_objects = sample_frames * (population_objects / population_frames)
    positive_interval = wilson_interval(sample_positive, sample_frames)
    object_rate_interval = poisson_rate_interval(sample_objects, sample_frames)

    return {
        "selection_inference_design": inference_design,
        "simple_random_reference_only": inference_design != "simple_random_without_replacement",
        "population": {
            "frames": population_frames,
            "positive_frames": population_positive,
            "objects": population_objects,
        },
        "sample": {"frames": sample_frames, "positive_frames": sample_positive, "objects": sample_objects},
        "positive_frames": {
            "expected": expected_positive,
            "observed": sample_positive,
            "hypergeometric_sd": positive_sd,
            "standardized_deviation": (sample_positive - expected_positive) / positive_sd if positive_sd > 0.0 else None,
            "observed_fraction_interval": positive_interval,
        },
        "objects": {
            "expected": expected_objects,
            "observed": sample_objects,
            "observed_rate_per_frame_interval": object_rate_interval,
            "poisson_style_note": "object counts can cluster within frames; this interval is descriptive, not an independence claim",
        },
    }


def _metric_interval(metric: dict[str, Any], confidence: float) -> dict[str, Any]:
    kind = metric["kind"]
    if kind == "paired_binary_delta":
        return paired_binary_benefit_interval(
            metric["improvements"],
            metric["regressions"],
            metric["opportunities"],
            confidence,
        )
    if kind == "binomial_delta":
        return binomial_benefit_interval(
            metric["baseline_successes"],
            metric["candidate_successes"],
            metric["opportunities"],
            metric["direction"],
            confidence,
        )
    if kind == "poisson_rate_delta":
        return poisson_benefit_interval(
            metric["baseline_count"],
            metric["candidate_count"],
            metric["exposure"],
            metric["direction"],
            confidence,
        )
    raise ValueError(f"unsupported metric kind: {kind!r}")


def triage_metric(metric: dict[str, Any], confidence: float = DEFAULT_CONFIDENCE) -> dict[str, Any]:
    role = metric.get("role", "primary")
    if role not in {"primary", "guardrail"}:
        raise ValueError("metric role must be primary or guardrail")
    minimum_improvement = _finite_non_negative(metric.get("minimum_material_improvement", 0.0), "minimum_material_improvement")
    maximum_harm = _finite_non_negative(metric.get("maximum_tolerable_harm", 0.0), "maximum_tolerable_harm")
    benefit = _metric_interval(metric, confidence)
    lower = benefit["lower"]
    upper = benefit["upper"]

    if lower is None or upper is None:
        state = "INSUFFICIENT_EVIDENCE"
    elif upper < -maximum_harm:
        state = "HARMFUL"
    elif role == "guardrail":
        state = "SAFE" if lower >= -maximum_harm else "AMBIGUOUS"
    elif lower >= minimum_improvement:
        state = "PROMISING"
    elif lower >= -maximum_harm and upper < minimum_improvement:
        state = "TOO_SMALL"
    else:
        state = "AMBIGUOUS"

    return {
        "name": metric["name"],
        "role": role,
        "kind": metric["kind"],
        "benefit_interval": benefit,
        "minimum_material_improvement": minimum_improvement,
        "maximum_tolerable_harm": maximum_harm,
        "state": state,
    }


def promotion_decision(metrics: list[dict[str, Any]], confidence: float = DEFAULT_CONFIDENCE) -> dict[str, Any]:
    confidence = _validate_confidence(confidence)
    results = [triage_metric(metric, confidence) for metric in metrics]
    primary = [item for item in results if item["role"] == "primary"]
    guardrails = [item for item in results if item["role"] == "guardrail"]
    if not primary:
        raise ValueError("at least one primary metric is required")

    harmful = [item["name"] for item in results if item["state"] == "HARMFUL"]
    if harmful:
        category = "CLEAR_REJECT"
        reason = "at least one metric excludes the allowed harm region"
    elif all(item["state"] == "TOO_SMALL" for item in primary):
        category = "CLEAR_REJECT"
        reason = "all primary metrics exclude the declared minimum material improvement"
    elif all(item["state"] == "PROMISING" for item in primary) and all(
        item["state"] == "SAFE" for item in guardrails
    ):
        category = "PROMOTE_TO_FULL"
        reason = "all primary metrics support material improvement and guardrails exclude unacceptable harm"
    else:
        category = "AMBIGUOUS"
        reason = "smoke evidence overlaps a material boundary or is insufficient"

    return {
        "category": category,
        "confidence": confidence,
        "reason": reason,
        "metrics": results,
        "final_benchmark_claim": False,
        "full_corpus_required_for_acceptance": True,
    }


def compute_budget(
    full_frames: int,
    smoke_frames: int,
    survival_rates: tuple[float, ...] = (0.0, 0.1, 0.25, 0.5, 0.75, 1.0),
) -> dict[str, Any]:
    full_frames = _positive_int(full_frames, "full_frames")
    smoke_frames = _positive_int(smoke_frames, "smoke_frames")
    if smoke_frames > full_frames:
        raise ValueError("smoke_frames cannot exceed full_frames")

    smoke_fraction = smoke_frames / full_frames
    scenarios = []
    for rate in survival_rates:
        rate = float(rate)
        if not 0.0 <= rate <= 1.0:
            raise ValueError("survival rates must be in [0, 1]")
        cost_ratio = smoke_fraction + rate
        scenarios.append(
            {
                "candidate_survival_rate": rate,
                "cost_vs_all_full": cost_ratio,
                "expected_savings_fraction": 1.0 - cost_ratio,
            }
        )
    return {
        "smoke_fraction_of_full": smoke_fraction,
        "full_to_smoke_frame_ratio": full_frames / smoke_frames,
        "break_even_survival_rate": 1.0 - smoke_fraction,
        "model": "all candidates pay smoke cost; survivors additionally pay one full-corpus run",
        "scenarios": scenarios,
    }


def resolution_limits(sample: dict[str, Any], confidence: float = DEFAULT_CONFIDENCE) -> dict[str, Any]:
    frames = _positive_int(sample["frames"], "sample.frames")
    positives = _non_negative_int(sample["positive_frames"], "sample.positive_frames")
    objects = _non_negative_int(sample["objects"], "sample.objects")
    frame_zero = paired_binary_benefit_interval(0, 0, frames, confidence)
    positive_zero = paired_binary_benefit_interval(0, 0, positives, confidence)
    object_zero = paired_binary_benefit_interval(0, 0, objects, confidence)
    zero_count_rate = poisson_rate_interval(0, frames, confidence)
    return {
        "zero_observed_frame_event_upper_probability": zero_event_upper_probability(frames, confidence),
        "zero_observed_positive_frame_event_upper_probability": zero_event_upper_probability(positives, confidence),
        "zero_observed_object_event_upper_probability": zero_event_upper_probability(objects, confidence),
        "paired_zero_change_half_width": {
            "all_frames": frame_zero["upper"],
            "positive_frames": positive_zero["upper"],
            "objects": object_zero["upper"],
        },
        "zero_count_poisson_upper_rate_per_frame": zero_count_rate["upper"],
        "interpretation": (
            "effects smaller than these zero-event/zero-change ceilings cannot be ruled out merely because the smoke sample "
            "observes none; rare subgroup and clustered-object effects are less identifiable still"
        ),
    }


def analyze(spec: dict[str, Any]) -> dict[str, Any]:
    if spec.get("schema") != INPUT_SCHEMA:
        raise ValueError(f"input schema must be {INPUT_SCHEMA!r}")
    selection = spec["selection"]
    if selection.get("frozen_before_candidate_scoring") is not True:
        raise ValueError("smoke selection must be frozen before candidate scoring")
    if not isinstance(selection.get("corpus_revision"), str) or not selection["corpus_revision"]:
        raise ValueError("selection.corpus_revision must be a non-empty string")
    _sha256(selection.get("corpus_sha256"), "selection.corpus_sha256")
    _sha256(selection.get("selection_manifest_sha256"), "selection.selection_manifest_sha256")
    _sha256(selection.get("selection_identity_sha256"), "selection.selection_identity_sha256")

    population_frames = _positive_int(selection["population_frames"], "selection.population_frames")
    sample_frames = _positive_int(selection["sample_frames"], "selection.sample_frames")
    if sample_frames > population_frames:
        raise ValueError("sample_frames cannot exceed population_frames")

    confidence = _validate_confidence(spec.get("confidence", DEFAULT_CONFIDENCE))
    inference_design = selection.get("inference_design", "unspecified")
    if not isinstance(inference_design, str) or not inference_design:
        raise ValueError("selection.inference_design must be a non-empty string")
    sample = spec.get("sample_composition")
    population = spec.get("population_composition")
    coverage = None
    resolution = None
    if population is not None or sample is not None:
        if population is None or sample is None:
            raise ValueError("population_composition and sample_composition must be supplied together")
        if population["frames"] != population_frames or sample["frames"] != sample_frames:
            raise ValueError("composition frame counts must match selection provenance")
        coverage = coverage_summary(population, sample, inference_design)
        resolution = resolution_limits(sample, confidence)

    metrics = spec.get("metrics", [])
    if not isinstance(metrics, list):
        raise ValueError("metrics must be a list")
    decision = promotion_decision(metrics, confidence) if metrics else None
    return {
        "schema": RESULT_SCHEMA,
        "selection": selection,
        "confidence": confidence,
        "coverage": coverage,
        "rare_event_sensitivity": rare_event_sensitivity(
            population_frames,
            sample_frames,
            inference_design,
        ),
        "resolution_limits": resolution,
        "triage": decision,
        "compute_budget": compute_budget(population_frames, sample_frames),
        "claims": {
            "smoke_is_triage_only": True,
            "smoke_pass_is_final_benchmark_claim": False,
            "authoritative_acceptance_requires_full_frozen_corpus": True,
            "simple_random_inference_supported": inference_design == "simple_random_without_replacement",
        },
    }


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Conservative smoke-sample statistics and full-benchmark promotion gate")
    parser.add_argument("--input", required=True, help="spectratrack-smoke-promotion-input-v1 JSON")
    parser.add_argument("--output", help="write deterministic JSON result to this path")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    spec = json.loads(Path(args.input).read_text(encoding="utf-8"))
    result = analyze(spec)
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        Path(args.output).write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
