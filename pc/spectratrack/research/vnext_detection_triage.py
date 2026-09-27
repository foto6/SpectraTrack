from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

from ..integrity import sha256_file
from .vnext_detection_corpus import _artifact_record, _write_completion_marker, run_corpus_benchmark
from .vnext_detection_selector import load_frozen_frame_selector, selector_record

MATRIX_SCHEMA = "spectratrack-a1-triage-candidate-matrix-v1"
TRIAGE_SCHEMA = "spectratrack-a1-triage-result-v1"
_ALLOWED_VARIATION_FIELDS = {"tile_size"}


def load_candidate_matrix(path: str | Path) -> tuple[dict[str, Any], str]:
    source = Path(path)
    data = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("schema") != MATRIX_SCHEMA:
        raise ValueError("unsupported candidate matrix schema")
    if data.get("expected_frame_count") != 400:
        raise ValueError("triage matrix must bind exactly 400 frames")
    experiments = data.get("experiments")
    if not isinstance(experiments, list) or len(experiments) < 2:
        raise ValueError("matrix requires one control and at least one candidate")
    enabled = [item for item in experiments if isinstance(item, dict) and item.get("enabled") is True]
    ids = [item.get("id") for item in experiments if isinstance(item, dict)]
    if any(not isinstance(item, str) or not item for item in ids) or len(set(ids)) != len(ids):
        raise ValueError("experiment ids must be unique non-empty strings")
    controls = [item for item in enabled if item.get("role") == "control"]
    if len(controls) != 1 or controls[0].get("id") != data.get("control_id"):
        raise ValueError("matrix must have one enabled control matching control_id")
    candidates = [item for item in enabled if item.get("role") == "candidate"]
    if not candidates:
        raise ValueError("matrix must have at least one enabled candidate")
    control_config = controls[0].get("config")
    if not isinstance(control_config, dict):
        raise ValueError("control config must be an object")
    for item in enabled:
        config = item.get("config")
        if not isinstance(config, dict):
            raise ValueError(f"{item.get('id')}: config must be an object")
        if config.get("enhancement") != "off":
            raise ValueError(f"{item['id']}: enhancement must remain off")
        if config.get("method") != "hard-nms":
            raise ValueError(f"{item['id']}: rejected fusion challengers are not eligible")
        if item.get("role") not in {"control", "candidate"}:
            raise ValueError(f"{item['id']}: role must be control or candidate")
        if item.get("role") == "candidate":
            changed = {key for key in set(control_config) | set(config) if control_config.get(key) != config.get(key)}
            forbidden = changed - _ALLOWED_VARIATION_FIELDS
            if forbidden:
                raise ValueError(f"{item['id']}: changes locked fields: {sorted(forbidden)}")
            if not changed:
                raise ValueError(f"{item['id']}: candidate must differ from control")
    if not isinstance(data.get("decision_rules"), dict):
        raise ValueError("decision_rules must be an object")
    return data, sha256_file(source)


def _metric(report: Mapping[str, Any], experiment: Mapping[str, Any]) -> Mapping[str, Any]:
    metrics = report.get("methods", {}).get(experiment["config"]["method"])
    if not isinstance(metrics, dict):
        raise ValueError("result is missing expected method metrics")
    return metrics


def _delta(candidate: Any, control: Any) -> float | None:
    if candidate is None or control is None:
        return None
    return float(candidate) - float(control)


def score_candidate(
    control_report: Mapping[str, Any],
    candidate_report: Mapping[str, Any],
    *,
    control: Mapping[str, Any],
    candidate: Mapping[str, Any],
    rules: Mapping[str, Any],
) -> dict[str, Any]:
    base = _metric(control_report, control)
    current = _metric(candidate_report, candidate)
    deltas = {
        "precision": float(current["precision"]) - float(base["precision"]),
        "recall": float(current["recall"]) - float(base["recall"]),
        "f1": float(current["f1"]) - float(base["f1"]),
        "tp": int(current["tp"]) - int(base["tp"]),
        "fp": int(current["fp"]) - int(base["fp"]),
        "fn": int(current["fn"]) - int(base["fn"]),
        "bbox_localization_iou": _delta(current.get("bbox_localization_iou"), base.get("bbox_localization_iou")),
        "center_error_gt_diag_ratio": _delta(
            current.get("center_error_gt_diag_ratio"), base.get("center_error_gt_diag_ratio")
        ),
    }
    promote = rules["promote_to_full"]
    reject = rules["reject"]
    bbox_delta = deltas["bbox_localization_iou"]
    center_delta = deltas["center_error_gt_diag_ratio"]
    material = (
        deltas["f1"] >= float(promote["min_f1_delta"])
        and deltas["recall"] >= float(promote["min_recall_delta"])
        and deltas["precision"] >= -float(promote["max_precision_drop"])
        and (bbox_delta is None or bbox_delta >= -float(promote["max_bbox_iou_drop"]))
        and (center_delta is None or center_delta <= float(promote["max_center_error_ratio_increase"]))
    )
    inferior = (
        deltas["f1"] <= float(reject["max_f1_delta"])
        or deltas["recall"] <= float(reject["max_recall_delta"])
        or deltas["precision"] <= float(reject["max_precision_delta"])
    )
    return {
        "candidate_id": candidate["id"],
        "recommendation": "PROMOTE_TO_FULL" if material else ("REJECT" if inferior else "AMBIGUOUS"),
        "deltas_vs_control": deltas,
        "control_metrics": dict(base),
        "candidate_metrics": dict(current),
        "full5000_not_run": True,
    }


def _namespace(cli: argparse.Namespace, experiment: Mapping[str, Any], output_dir: Path) -> SimpleNamespace:
    c = experiment["config"]
    return SimpleNamespace(
        model=cli.model, ground_truth=cli.ground_truth, video_root=cli.video_root,
        output=str(output_dir / f"{experiment['id']}.json"),
        source_commit=cli.source_commit, corpus_revision=cli.corpus_revision,
        include_video=[], frame_manifest=cli.frame_manifest,
        frame_manifest_sha256=cli.frame_manifest_sha256, frame_manifest_count=400,
        frame_manifest_revision=cli.frame_manifest_revision,
        prefusion_dir=str(output_dir / "prefusion" / experiment["id"]),
        replay_dir=str(output_dir / "replays" / experiment["id"]),
        resume=cli.resume, per_video=True, frame_limit_per_video=0,
        input_size=int(c["input_size"]), conf=float(c["conf"]),
        decoder_iou=float(c["decoder_iou"]), person_conf=float(c["person_conf"]),
        tile_size=int(c["tile_size"]), tile_overlap=float(c["tile_overlap"]),
        match_iou=float(c["match_iou"]), fusion_iou=float(c["fusion_iou"]),
        center_ratio=float(c["center_ratio"]), size_ratio=float(c["size_ratio"]),
        score_power=float(c["score_power"]), full_weight=float(c["full_weight"]),
        tile_weight=float(c["tile_weight"]), evidence_weak_score=float(c["evidence_weak_score"]),
        evidence_solo_score=float(c["evidence_solo_score"]),
        evidence_strong_score=float(c["evidence_strong_score"]),
        evidence_min_sources=int(c["evidence_min_sources"]), method=[c["method"]], cpu=cli.cpu,
    )


def run_triage(cli: argparse.Namespace) -> dict[str, Any]:
    matrix, matrix_sha = load_candidate_matrix(cli.matrix)
    selector = load_frozen_frame_selector(
        cli.frame_manifest, expected_sha256=cli.frame_manifest_sha256,
        expected_count=400, expected_revision=cli.frame_manifest_revision,
    )
    output_dir = Path(cli.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    enabled = [item for item in matrix["experiments"] if item["enabled"]]
    reports: dict[str, dict[str, Any]] = {}
    artifacts: dict[str, dict[str, Any]] = {}
    for experiment in enabled:
        args = _namespace(cli, experiment, output_dir)
        report = run_corpus_benchmark(args)
        target = Path(args.output)
        target.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        marker = _write_completion_marker(target, report)
        reports[experiment["id"]] = report
        artifacts[experiment["id"]] = {
            "result": _artifact_record(target), "completion_marker": _artifact_record(marker)
        }
    control = next(item for item in enabled if item["id"] == matrix["control_id"])
    decisions = [
        score_candidate(reports[control["id"]], reports[item["id"]],
                        control=control, candidate=item, rules=matrix["decision_rules"])
        for item in enabled if item["role"] == "candidate"
    ]
    return {
        "schema": TRIAGE_SCHEMA, "source_commit": cli.source_commit,
        "corpus_revision": cli.corpus_revision,
        "candidate_matrix": {"path": str(Path(cli.matrix)), "sha256": matrix_sha, "name": matrix["name"]},
        "frame_selector": selector_record(selector), "control_id": control["id"],
        "experiments": enabled,
        "artifacts": artifacts, "decisions": decisions,
        "note": "Smoke400 is triage only; this harness never launches FULL5000.",
    }


def _write_summary(path: str | Path, report: Mapping[str, Any]) -> dict[str, Any]:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    marker = target.with_name(target.name + ".complete.json")
    marker.write_text(json.dumps({
        "schema": "spectratrack-a1-triage-completion-v1",
        "summary": _artifact_record(target),
        "candidate_matrix": report["candidate_matrix"],
        "frame_selector": report["frame_selector"],
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"summary": _artifact_record(target), "completion_marker": _artifact_record(marker)}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SpectraTrack A1 deterministic smoke400 triage")
    sub = parser.add_subparsers(dest="command", required=True)
    validate = sub.add_parser("validate-matrix")
    validate.add_argument("--matrix", required=True)
    run = sub.add_parser("run")
    for name in ("matrix", "frame-manifest", "frame-manifest-sha256", "frame-manifest-revision",
                 "model", "ground-truth", "video-root", "output-dir", "summary",
                 "source-commit", "corpus-revision"):
        run.add_argument(f"--{name}", required=True)
    run.add_argument("--resume", action="store_true")
    run.add_argument("--cpu", action="store_true")
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    if args.command == "validate-matrix":
        matrix, digest = load_candidate_matrix(args.matrix)
        print(json.dumps({
            "schema": matrix["schema"], "name": matrix["name"], "sha256": digest,
            "expected_frame_count": matrix["expected_frame_count"],
            "enabled": [item["id"] for item in matrix["experiments"] if item["enabled"]],
        }, sort_keys=True))
        return 0
    report = run_triage(args)
    print(json.dumps({"decisions": report["decisions"], **_write_summary(args.summary, report)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
