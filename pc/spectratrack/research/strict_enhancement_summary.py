from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from spectratrack.integrity import sha256_file

SCHEMA = "spectratrack-vnext-enhancement-postfusion-summary-v1"
PROFILE_SCHEMA = "spectratrack-vnext-enhancement-profile-v1"
STRICT_OPERATIONS = ("bilateral", "current_adaptive_cached", "sharpen")
STABILITY_KEYS = (
    "normalized_center_jitter_mean",
    "width_log_jitter_mean",
    "height_log_jitter_mean",
    "area_log_jitter_mean",
    "temporal_iou_mean",
)


def _load_json(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    with source.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"{source}: top-level JSON must be an object")
    return data


def _finite_number(value: Any) -> float | None:
    if value is None or isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _model_sha(result: dict[str, Any]) -> str | None:
    model = result.get("model")
    return model.get("sha256") if isinstance(model, dict) else None


def _validate_postfusion_pair(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
) -> None:
    baseline_gt = baseline.get("ground_truth_sha256")
    candidate_gt = candidate.get("ground_truth_sha256")
    if not baseline_gt or baseline_gt != candidate_gt:
        raise ValueError("baseline/candidate ground-truth SHA-256 mismatch")

    baseline_eval = baseline.get("evaluation", {})
    candidate_eval = candidate.get("evaluation", {})
    for key in ("label", "match_iou"):
        if baseline_eval.get(key) != candidate_eval.get(key):
            raise ValueError(f"baseline/candidate evaluation mismatch: {key}")

    baseline_model = _model_sha(baseline)
    candidate_model = _model_sha(candidate)
    if baseline_model and candidate_model and baseline_model != candidate_model:
        raise ValueError("baseline/candidate model SHA-256 mismatch")


def _validate_strict_profile(
    profile: dict[str, Any],
    operation: str,
    baseline: dict[str, Any],
) -> dict[str, Any]:
    if profile.get("schema") != PROFILE_SCHEMA:
        raise ValueError(f"unsupported A3 profile schema: {profile.get('schema')!r}")
    if operation not in STRICT_OPERATIONS:
        raise ValueError(f"operation must be one of {STRICT_OPERATIONS}")

    settings = profile.get("settings", {})
    if settings.get("selective_gate") != "weak-person":
        raise ValueError("strict A3 comparison requires selective_gate=weak-person")
    if int(settings.get("max_enhanced_rois_per_frame", 0)) != 1:
        raise ValueError("strict A3 comparison requires max_enhanced_rois_per_frame=1")
    if settings.get("raw_corroboration_required") is not True:
        raise ValueError("strict A3 comparison requires raw corroboration")

    operations = profile.get("operations", {})
    if operation not in operations:
        raise ValueError(f"A3 profile does not contain operation {operation!r}")

    profile_gt = profile.get("ground_truth", {})
    profile_gt_sha = profile_gt.get("sha256") if isinstance(profile_gt, dict) else None
    baseline_gt_sha = baseline.get("ground_truth_sha256")
    if profile_gt_sha and baseline_gt_sha and profile_gt_sha != baseline_gt_sha:
        raise ValueError("A3 profile/post-fusion ground-truth SHA-256 mismatch")

    profile_model = profile.get("model", {})
    profile_model_sha = profile_model.get("sha256") if isinstance(profile_model, dict) else None
    baseline_model_sha = _model_sha(baseline)
    if profile_model_sha and baseline_model_sha and profile_model_sha != baseline_model_sha:
        raise ValueError("A3 profile/post-fusion model SHA-256 mismatch")

    return operations[operation]


def _metric_set(result: dict[str, Any]) -> set[str]:
    metrics = result.get("metrics", {})
    values = metrics.get("matched_ground_truth", [])
    if not isinstance(values, list):
        raise ValueError("metrics.matched_ground_truth must be a list")
    return {str(item) for item in values}


def _stability(result: dict[str, Any]) -> dict[str, Any]:
    metrics = result.get("metrics", {})
    bbox = metrics.get("bbox_stability", {})
    detection = bbox.get("detection", {}) if isinstance(bbox, dict) else {}
    return detection if isinstance(detection, dict) else {}


def _stability_summary(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
) -> dict[str, Any]:
    base = _stability(baseline)
    cand = _stability(candidate)
    delta: dict[str, float | None] = {}
    for key in STABILITY_KEYS:
        base_value = _finite_number(base.get(key))
        cand_value = _finite_number(cand.get(key))
        delta[key] = (
            cand_value - base_value
            if base_value is not None and cand_value is not None
            else None
        )
    return {"baseline": base, "candidate": cand, "delta": delta}


def _subgroup_summary(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    section: str,
) -> dict[str, Any]:
    base_groups = baseline.get("metrics", {}).get(section, {})
    cand_groups = candidate.get("metrics", {}).get(section, {})
    if not isinstance(base_groups, dict) or not isinstance(cand_groups, dict):
        return {}

    output: dict[str, Any] = {}
    for name in sorted(set(base_groups) | set(cand_groups)):
        base = base_groups.get(name, {})
        cand = cand_groups.get(name, {})
        if not isinstance(base, dict) or not isinstance(cand, dict):
            continue
        base_recall = _finite_number(base.get("recall"))
        cand_recall = _finite_number(cand.get("recall"))
        base_precision = _finite_number(base.get("precision"))
        cand_precision = _finite_number(cand.get("precision"))
        output[name] = {
            "baseline_recall": base_recall,
            "candidate_recall": cand_recall,
            "recall_delta": (
                cand_recall - base_recall
                if base_recall is not None and cand_recall is not None
                else None
            ),
            "baseline_precision": base_precision,
            "candidate_precision": cand_precision,
            "precision_delta": (
                cand_precision - base_precision
                if base_precision is not None and cand_precision is not None
                else None
            ),
            "baseline_gt": base.get("gt"),
            "candidate_gt": cand.get("gt"),
            "baseline_tp": base.get("tp"),
            "candidate_tp": cand.get("tp"),
            "baseline_fn": base.get("fn"),
            "candidate_fn": cand.get("fn"),
        }
    return output


def _performance_summary(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
) -> dict[str, Any]:
    base = baseline.get("performance", {})
    cand = candidate.get("performance", {})
    base_wall = _finite_number(base.get("wall_seconds"))
    cand_wall = _finite_number(cand.get("wall_seconds"))
    base_calls = _finite_number(base.get("onnx_inference_calls"))
    cand_calls = _finite_number(cand.get("onnx_inference_calls"))
    return {
        "baseline_wall_seconds": base_wall,
        "candidate_wall_seconds": cand_wall,
        "wall_seconds_delta": (
            cand_wall - base_wall
            if base_wall is not None and cand_wall is not None
            else None
        ),
        "baseline_onnx_calls": base_calls,
        "candidate_onnx_calls": cand_calls,
        "postfusion_run_onnx_call_delta": (
            cand_calls - base_calls
            if base_calls is not None and cand_calls is not None
            else None
        ),
        "baseline_processing_seconds_per_source_second": _finite_number(
            base.get("processing_seconds_per_source_second")
        ),
        "candidate_processing_seconds_per_source_second": _finite_number(
            cand.get("processing_seconds_per_source_second")
        ),
    }


def build_summary(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    profile: dict[str, Any],
    *,
    operation: str,
    baseline_path: str | Path | None = None,
    candidate_path: str | Path | None = None,
    profile_path: str | Path | None = None,
) -> dict[str, Any]:
    _validate_postfusion_pair(baseline, candidate)
    operation_profile = _validate_strict_profile(profile, operation, baseline)

    baseline_metrics = baseline["metrics"]
    candidate_metrics = candidate["metrics"]
    baseline_matches = _metric_set(baseline)
    candidate_matches = _metric_set(candidate)
    recovered = sorted(candidate_matches - baseline_matches)
    lost = sorted(baseline_matches - candidate_matches)

    baseline_fp = int(baseline_metrics["false_positives"])
    candidate_fp = int(candidate_metrics["false_positives"])
    fp_delta = candidate_fp - baseline_fp

    cost = operation_profile.get("cost", {})
    raw_calls = int(cost.get("raw_probe_calls", 0))
    extra_calls = int(cost.get("enhanced_inference_calls", 0))
    recovered_count = len(recovered)

    recovered_per_extra_call = (
        recovered_count / extra_calls if extra_calls > 0 else None
    )
    fp_cost_per_recovered = (
        max(fp_delta, 0) / recovered_count if recovered_count > 0 else None
    )
    net_fp_delta_per_recovered = (
        fp_delta / recovered_count if recovered_count > 0 else None
    )

    provenance: dict[str, Any] = {
        "ground_truth_sha256": baseline.get("ground_truth_sha256"),
        "model_sha256": _model_sha(baseline),
        "evaluation": baseline.get("evaluation"),
        "a3_profile_source_commit": profile.get("source_commit"),
        "a3_profile_corpus_revision": profile.get("ground_truth", {}).get(
            "corpus_revision"
        ),
    }
    for name, path in (
        ("baseline_result", baseline_path),
        ("candidate_result", candidate_path),
        ("a3_profile", profile_path),
    ):
        if path is not None:
            provenance[name] = {
                "path": str(path),
                "sha256": sha256_file(path),
            }

    return {
        "schema": SCHEMA,
        "operation": operation,
        "strict_configuration": {
            "selective_gate": "weak-person",
            "max_enhanced_rois_per_frame": 1,
            "raw_corroboration_required": True,
        },
        "provenance": provenance,
        "postfusion_quality": {
            "baseline_recall": float(baseline_metrics["recall"]),
            "candidate_recall": float(candidate_metrics["recall"]),
            "recall_delta": (
                float(candidate_metrics["recall"]) - float(baseline_metrics["recall"])
            ),
            "baseline_precision": float(baseline_metrics["precision"]),
            "candidate_precision": float(candidate_metrics["precision"]),
            "precision_delta": (
                float(candidate_metrics["precision"])
                - float(baseline_metrics["precision"])
            ),
            "baseline_false_positives": baseline_fp,
            "candidate_false_positives": candidate_fp,
            "fp_delta": fp_delta,
            "baseline_false_negatives": int(baseline_metrics["false_negatives"]),
            "candidate_false_negatives": int(candidate_metrics["false_negatives"]),
            "recovered_gt": recovered,
            "recovered_gt_count": recovered_count,
            "lost_gt": lost,
            "lost_gt_count": len(lost),
        },
        "bbox_stability": _stability_summary(baseline, candidate),
        "subgroups": {
            "by_tag": _subgroup_summary(baseline, candidate, "by_tag"),
            "by_attribute": _subgroup_summary(baseline, candidate, "by_attribute"),
            "by_size": _subgroup_summary(baseline, candidate, "by_size"),
        },
        "compute": {
            "raw_onnx_calls": raw_calls,
            "extra_enhancement_onnx_calls": extra_calls,
            "recovered_gt_per_extra_call": recovered_per_extra_call,
            "fp_cost_per_recovered_gt": fp_cost_per_recovered,
            "net_fp_delta_per_recovered_gt": net_fp_delta_per_recovered,
            "operation_preprocessing_ms": _finite_number(
                cost.get("operation_preprocessing_ms")
            ),
            "enhanced_inference_ms": _finite_number(
                cost.get("enhanced_inference_ms")
            ),
            "a3_attributed_total_ms": _finite_number(cost.get("attributed_total_ms")),
        },
        "postfusion_run_performance": _performance_summary(baseline, candidate),
        "notes": [
            "Fusion is not implemented by A3; baseline/candidate inputs must already be canonical post-fusion results.",
            "A3 cost fields come from the strict weak-person profile and are validated against its GT/model provenance when available.",
            "A positive temporal_iou_mean delta is better; lower center/width/height/area jitter deltas are better.",
            "NightOwls low-light/blur/low-contrast/occlusion slices are reported only when A5 ground truth exposes those official tags/attributes; A3 does not invent labels.",
        ],
    }


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Summarize strict A3 enhancement quality/cost from canonical post-fusion results"
    )
    parser.add_argument("--baseline-result", required=True)
    parser.add_argument("--candidate-result", required=True)
    parser.add_argument("--a3-profile", required=True)
    parser.add_argument("--operation", required=True, choices=STRICT_OPERATIONS)
    parser.add_argument("--output", required=True)
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    baseline = _load_json(args.baseline_result)
    candidate = _load_json(args.candidate_result)
    profile = _load_json(args.a3_profile)
    summary = build_summary(
        baseline,
        candidate,
        profile,
        operation=args.operation,
        baseline_path=args.baseline_result,
        candidate_path=args.candidate_result,
        profile_path=args.a3_profile,
    )
    _write_json(args.output, summary)
    quality = summary["postfusion_quality"]
    compute = summary["compute"]
    print(
        f"operation={args.operation} recovered={quality['recovered_gt_count']} "
        f"lost={quality['lost_gt_count']} fp_delta={quality['fp_delta']} "
        f"recall={quality['candidate_recall']:.6f} "
        f"precision={quality['candidate_precision']:.6f} "
        f"extra_calls={compute['extra_enhancement_onnx_calls']} "
        f"recovered_per_extra_call={compute['recovered_gt_per_extra_call']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
