from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from .error_analysis import TAXONOMY, _classify_false_negative, load_frame_manifest
from .qa_benchmark import GroundTruthFrame, load_ground_truth

DISCOVERY_SCHEMA = "spectratrack-error-discovery-taxonomy-v1"
LINEAGE_SCHEMA = "spectratrack-error-hypothesis-lineage-v1"
ALLOWED_LINEAGE_ROLES = ("A1", "A3")
DEFAULT_TOP_K = 5
MAX_TOP_K = 25


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8-sig") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected JSON object")
    return payload


def _parse_miss_id(event_id: str) -> tuple[str, int, int]:
    prefix, sep, raw_index = event_id.rpartition("#index:")
    if not sep:
        raise ValueError(f"invalid A1 miss ID: {event_id}")
    video, sep, raw_frame = prefix.rpartition("#")
    if not sep or not video:
        raise ValueError(f"invalid A1 miss ID: {event_id}")
    try:
        return video, int(raw_frame), int(raw_index)
    except ValueError as exc:
        raise ValueError(f"invalid A1 miss ID: {event_id}") from exc


def _ground_truth_index(frames: list[GroundTruthFrame]) -> dict[tuple[str, int], GroundTruthFrame]:
    indexed: dict[tuple[str, int], GroundTruthFrame] = {}
    for frame in frames:
        key = (frame.video, frame.frame)
        if key in indexed:
            raise ValueError(f"duplicate ground-truth frame: {frame.video}#{frame.frame}")
        indexed[key] = frame
    return indexed


def _classify_miss(
    event_id: str,
    gt_by_frame: dict[tuple[str, int], GroundTruthFrame],
    *,
    match_iou: float,
) -> tuple[tuple[str, ...], float]:
    video, frame_number, object_index = _parse_miss_id(event_id)
    frame = gt_by_frame.get((video, frame_number))
    if frame is None:
        raise ValueError(f"A1 miss ID absent from frozen GT: {event_id}")
    if object_index < 0 or object_index >= len(frame.objects):
        raise ValueError(f"A1 miss ID object index out of range: {event_id}")
    obj = frame.objects[object_index]
    if obj.ignore or obj.label != "person":
        raise ValueError(f"A1 miss ID does not reference scored person GT: {event_id}")
    categories, _, severity = _classify_false_negative(
        frame, obj, None, match_iou
    )
    return categories, severity


def _method_metrics(artifact: dict[str, Any], method: str = "hard-nms") -> dict[str, Any]:
    if artifact.get("schema") != "spectratrack-vnext-fusion-corpus-v1":
        raise ValueError("unsupported A1 corpus artifact schema")
    per_video = artifact.get("per_video")
    if not isinstance(per_video, dict):
        raise ValueError("A1 corpus artifact missing per_video")
    totals = {
        "tp": 0,
        "fp": 0,
        "fn": 0,
        "duplicate_count_before_fusion": 0,
    }
    miss_ids: set[str] = set()
    for video in sorted(per_video):
        methods = per_video[video]
        if not isinstance(methods, dict) or method not in methods:
            raise ValueError(f"A1 artifact missing {method} metrics for {video}")
        metrics = methods[method]
        for key in totals:
            value = metrics.get(key, 0)
            if not isinstance(value, int) or isinstance(value, bool):
                raise ValueError(f"A1 {key} must be integer")
            totals[key] += value
        taxonomy = metrics.get("error_taxonomy")
        if not isinstance(taxonomy, dict):
            raise ValueError(f"A1 artifact missing error_taxonomy for {video}")
        reps = taxonomy.get("representative_false_negatives", [])
        if not isinstance(reps, list) or not all(isinstance(item, str) for item in reps):
            raise ValueError("A1 representative_false_negatives must be strings")
        miss_ids.update(reps)
    if len(miss_ids) != totals["fn"]:
        raise ValueError(
            "A1 representative false-negative IDs do not cover every miss deterministically"
        )
    totals["miss_ids"] = sorted(miss_ids)
    return totals


def _ranked_ids(
    event_ids: set[str] | list[str],
    gt_by_frame: dict[tuple[str, int], GroundTruthFrame],
    *,
    match_iou: float,
    top_k: int,
    category: str | None = None,
) -> list[str]:
    ranked: list[tuple[float, str]] = []
    for event_id in event_ids:
        categories, severity = _classify_miss(
            event_id, gt_by_frame, match_iou=match_iou
        )
        if category is None or category in categories:
            ranked.append((-severity, event_id))
    ranked.sort()
    return [event_id for _, event_id in ranked[:top_k]]


def _taxonomy_counts(
    miss_ids: set[str],
    gt_by_frame: dict[tuple[str, int], GroundTruthFrame],
    *,
    match_iou: float,
) -> dict[str, int]:
    counts = {category: 0 for category in TAXONOMY}
    for event_id in sorted(miss_ids):
        categories, _ = _classify_miss(
            event_id, gt_by_frame, match_iou=match_iou
        )
        for category in categories:
            counts[category] += 1
    return counts


def _cluster(
    *,
    cluster_id: str,
    category: str,
    count: int,
    denominator: int | None,
    representative_ids: list[str],
    evidence_scope: str,
    note: str,
) -> dict[str, Any]:
    if category not in TAXONOMY:
        raise ValueError(f"unknown taxonomy category: {category}")
    return {
        "cluster_id": cluster_id,
        "taxonomy_category": category,
        "count": count,
        "rate": (count / denominator) if denominator else None,
        "rate_denominator": denominator,
        "representative_ids": representative_ids,
        "evidence_scope": evidence_scope,
        "note": note,
    }


def _a5_binding(
    manifest_path: str | Path,
    ground_truth_path: str | Path,
) -> tuple[dict[str, Any], list[GroundTruthFrame]]:
    manifest_raw = _load_json(manifest_path)
    selection = load_frame_manifest(
        manifest_path, ground_truth_path=ground_truth_path
    )
    if selection.schema != "spectratrack-smoke-slice-v1":
        raise ValueError("discovery report requires A5 spectratrack-smoke-slice-v1")
    artifacts = manifest_raw.get("artifacts")
    raw_selection = manifest_raw.get("selection")
    if not isinstance(artifacts, dict) or not isinstance(raw_selection, dict):
        raise ValueError("invalid A5 smoke manifest")
    proof_sha = artifacts.get("selection_proof_sha256")
    if not isinstance(proof_sha, str) or len(proof_sha) != 64:
        raise ValueError("A5 smoke manifest missing selection_proof_sha256")
    return (
        {
            "revision": selection.revision,
            "manifest_sha256": selection.file_sha256,
            "corpus_sha256": selection.corpus_sha256,
            "ground_truth_sha256": _sha256_file(ground_truth_path),
            "selection_proof_sha256": proof_sha,
            "frame_count": len(selection.frames),
            "selected_frame_ids_sha256": selection.frame_ids_sha256,
        },
        load_ground_truth(ground_truth_path),
    )


def build_discovery_report(
    *,
    manifest_path: str | Path,
    ground_truth_path: str | Path,
    a1_summary_path: str | Path,
    a1_control_path: str | Path,
    a1_candidate_path: str | Path,
    a3_final_path: str | Path,
    a3_score_path: str | Path,
    a3_evidence_path: str | Path | None = None,
    top_k: int = DEFAULT_TOP_K,
) -> dict[str, Any]:
    if top_k < 1 or top_k > MAX_TOP_K:
        raise ValueError(f"top_k must be in [1, {MAX_TOP_K}]")

    a5, gt_frames = _a5_binding(manifest_path, ground_truth_path)
    gt_by_frame = _ground_truth_index(gt_frames)

    a1_summary = _load_json(a1_summary_path)
    a1_control = _load_json(a1_control_path)
    a1_candidate = _load_json(a1_candidate_path)
    if a1_summary.get("schema") != "spectratrack-a1-triage-result-v1":
        raise ValueError("unsupported A1 triage summary schema")
    if a1_summary.get("corpus_revision") != a5["revision"]:
        raise ValueError("A1 summary corpus revision does not match A5")
    selector = a1_summary.get("frame_selector")
    if not isinstance(selector, dict):
        raise ValueError("A1 summary missing frame_selector")
    if selector.get("revision") != a5["revision"]:
        raise ValueError("A1 selector revision does not match A5")
    if selector.get("manifest_sha256") != a5["selection_proof_sha256"]:
        raise ValueError("A1 selection proof does not match A5")
    if selector.get("frame_count") != a5["frame_count"]:
        raise ValueError("A1 frame count does not match A5")

    artifacts = a1_summary.get("artifacts")
    control_id = a1_summary.get("control_id")
    decisions = a1_summary.get("decisions")
    if (
        not isinstance(artifacts, dict)
        or not isinstance(control_id, str)
        or not isinstance(decisions, list)
        or len(decisions) != 1
    ):
        raise ValueError("A1 summary structure is not the locked r1 shape")
    decision = decisions[0]
    candidate_id = decision.get("candidate_id")
    if decision.get("recommendation") != "REJECT":
        raise ValueError("A1 r1 source disposition is not REJECT")
    for artifact_id, path in (
        (control_id, a1_control_path),
        (candidate_id, a1_candidate_path),
    ):
        expected = artifacts.get(artifact_id, {}).get("result", {}).get("sha256")
        if expected != _sha256_file(path):
            raise ValueError(f"A1 artifact hash mismatch: {artifact_id}")

    for artifact in (a1_control, a1_candidate):
        if artifact.get("corpus_revision") != a5["revision"]:
            raise ValueError("A1 corpus artifact revision does not match A5")
        if artifact.get("ground_truth_sha256") != a5["ground_truth_sha256"]:
            raise ValueError("A1 corpus artifact GT hash does not match A5")
        if artifact.get("frame_count") != a5["frame_count"]:
            raise ValueError("A1 corpus artifact frame count does not match A5")

    control = _method_metrics(a1_control)
    candidate = _method_metrics(a1_candidate)
    match_iou = float(a1_control.get("settings", {}).get("match_iou", 0.5))
    control_misses = set(control["miss_ids"])
    candidate_misses = set(candidate["miss_ids"])
    recovered = control_misses - candidate_misses
    new_misses = candidate_misses - control_misses

    baseline_counts = _taxonomy_counts(
        control_misses, gt_by_frame, match_iou=match_iou
    )
    baseline_clusters: list[dict[str, Any]] = []
    for category in TAXONOMY:
        count = baseline_counts[category]
        if count:
            baseline_clusters.append(
                _cluster(
                    cluster_id=f"baseline:{category}",
                    category=category,
                    count=count,
                    denominator=control["fn"],
                    representative_ids=_ranked_ids(
                        control_misses,
                        gt_by_frame,
                        match_iou=match_iou,
                        top_k=top_k,
                        category=category,
                    ),
                    evidence_scope="control_false_negative_gt_geometry_metadata",
                    note="Candidate-independent control miss cluster from frozen GT and A1 control miss IDs.",
                )
            )
    if control["fp"]:
        baseline_clusters.append(
            _cluster(
                cluster_id="baseline:unknown_false_positive",
                category="unknown",
                count=control["fp"],
                denominator=control["fp"],
                representative_ids=[],
                evidence_scope="aggregate_false_positive_count_only",
                note="A1 control exposes aggregate FP count without post-fusion per-detection geometry; cause remains unknown.",
            )
        )

    new_miss_counts = _taxonomy_counts(
        new_misses, gt_by_frame, match_iou=match_iou
    )
    a1_clusters: list[dict[str, Any]] = []
    for category in TAXONOMY:
        count = new_miss_counts[category]
        if count:
            a1_clusters.append(
                _cluster(
                    cluster_id=f"a1_r1:new_miss:{category}",
                    category=category,
                    count=count,
                    denominator=len(new_misses),
                    representative_ids=_ranked_ids(
                        new_misses,
                        gt_by_frame,
                        match_iou=match_iou,
                        top_k=top_k,
                        category=category,
                    ),
                    evidence_scope="candidate_new_false_negatives_vs_control",
                    note="New tile512 miss relative to identical smoke400 control frames; source candidate remains REJECTED.",
                )
            )
    fp_delta = candidate["fp"] - control["fp"]
    if fp_delta:
        a1_clusters.append(
            _cluster(
                cluster_id="a1_r1:aggregate_fp_delta",
                category="unknown",
                count=fp_delta,
                denominator=control["fp"],
                representative_ids=[],
                evidence_scope="aggregate_postfusion_false_positive_delta",
                note="No post-fusion prediction geometry is available to assign the FP delta to background/duplicate/localization categories.",
            )
        )
    duplicate_delta = (
        candidate["duplicate_count_before_fusion"]
        - control["duplicate_count_before_fusion"]
    )
    if duplicate_delta:
        a1_clusters.append(
            _cluster(
                cluster_id="a1_r1:prefusion_duplicate_pressure",
                category="duplicate_fragmented_detection",
                count=duplicate_delta,
                denominator=control["duplicate_count_before_fusion"],
                representative_ids=[],
                evidence_scope="prefusion_duplicate_count_diagnostic",
                note="Diagnostic pre-fusion duplicate pressure only; it is not a post-fusion false-positive attribution.",
            )
        )

    a3_final = _load_json(a3_final_path)
    a3_score = _load_json(a3_score_path)
    if a3_final.get("schema") != "spectratrack-vnext-a3-smoke400-final-v1":
        raise ValueError("unsupported A3 final schema")
    if a3_final.get("decision") != "REJECT":
        raise ValueError("A3 r1 source disposition is not REJECT")
    a3_smoke = a3_final.get("a5_smoke")
    if not isinstance(a3_smoke, dict):
        raise ValueError("A3 final missing A5 smoke binding")
    expected_a5 = {
        "revision": a5["revision"],
        "manifest_sha256": a5["manifest_sha256"],
        "corpus_sha256": a5["corpus_sha256"],
        "ground_truth_sha256": a5["ground_truth_sha256"],
        "selection_proof_sha256": a5["selection_proof_sha256"],
        "frames": a5["frame_count"],
    }
    for key, value in expected_a5.items():
        if a3_smoke.get(key) != value:
            raise ValueError(f"A3 final A5 binding mismatch: {key}")
    a3_artifacts = a3_final.get("artifacts")
    if not isinstance(a3_artifacts, dict):
        raise ValueError("A3 final missing artifact hashes")
    if a3_artifacts.get("candidate_score_sha256") != _sha256_file(a3_score_path):
        raise ValueError("A3 candidate score hash mismatch")
    if a3_score.get("schema") != "spectratrack-vnext-a3-smoke400-score-v1":
        raise ValueError("unsupported A3 score schema")

    a3_candidate = a3_final.get("candidate_result")
    a3_compute = a3_final.get("compute")
    a3_off = a3_final.get("off")
    if not all(isinstance(item, dict) for item in (a3_candidate, a3_compute, a3_off)):
        raise ValueError("A3 final missing result/compute blocks")
    if (
        a3_off.get("tp") != control["tp"]
        or a3_off.get("fp") != control["fp"]
        or a3_off.get("fn") != control["fn"]
    ):
        raise ValueError("A3 OFF baseline does not match A1 control totals")

    a3_findings = [
        {
            "finding_id": "a3_r1:no_recall_gain",
            "related_taxonomy_categories": [
                "low_contrast_night_darkness",
                "missed_small_person",
                "occlusion_crowd",
            ],
            "recovered_gt": a3_candidate.get("recovered_gt"),
            "lost_gt": a3_candidate.get("lost_gt"),
            "false_negative_delta": a3_candidate.get("fn", 0) - a3_off.get("fn", 0),
            "recall_delta": a3_candidate.get("recall", 0.0) - a3_off.get("recall", 0.0),
            "representative_ids": _ranked_ids(
                control_misses,
                gt_by_frame,
                match_iou=match_iou,
                top_k=top_k,
                category="low_contrast_night_darkness",
            ),
            "note": "LIME recovered zero frozen GT; baseline discovery clusters therefore remain unresolved.",
        },
        {
            "finding_id": "a3_r1:compute_cost",
            "related_taxonomy_categories": [],
            "additional_raw_onnx_calls": a3_compute.get("additional_raw_onnx_calls"),
            "extra_enhancement_onnx_calls": a3_compute.get("extra_enhancement_onnx_calls"),
            "extra_onnx_calls_total": (
                int(a3_compute.get("additional_raw_onnx_calls", 0))
                + int(a3_compute.get("extra_enhancement_onnx_calls", 0))
            ),
            "activation_rate_vs_smoke_frames": a3_compute.get(
                "activation_rate_vs_smoke_frames"
            ),
            "recovered_gt_per_extra_enhancement_call": a3_compute.get(
                "recovered_gt_per_extra_enhancement_call"
            ),
            "representative_ids": [],
            "note": "Measured r1 compute cost only; this discovery report does not score or retune the rejected candidate.",
        },
        {
            "finding_id": "a3_r1:unlocalized_fp_reduction",
            "related_taxonomy_categories": ["unknown"],
            "false_positive_delta": a3_candidate.get("fp_delta"),
            "representative_ids": [],
            "note": "One fewer aggregate FP is observed, but absent per-detection geometry it is not labeled background false positive.",
        },
    ]

    source_artifacts: dict[str, Any] = {
        "a1_summary": {
            "schema": a1_summary.get("schema"),
            "sha256": _sha256_file(a1_summary_path),
            "source_commit": a1_summary.get("source_commit"),
            "source_disposition": "REJECT",
        },
        "a1_control": {
            "schema": a1_control.get("schema"),
            "sha256": _sha256_file(a1_control_path),
            "source_commit": a1_control.get("source_commit"),
        },
        "a1_tile512": {
            "schema": a1_candidate.get("schema"),
            "sha256": _sha256_file(a1_candidate_path),
            "source_commit": a1_candidate.get("source_commit"),
            "source_disposition": "REJECT",
        },
        "a3_final": {
            "schema": a3_final.get("schema"),
            "sha256": _sha256_file(a3_final_path),
            "source_commit": a3_final.get("candidate_lock", {}).get("commit"),
            "source_disposition": "REJECT",
        },
        "a3_score": {
            "schema": a3_score.get("schema"),
            "sha256": _sha256_file(a3_score_path),
            "a1_fusion_source_commit": a3_score.get("a1_fusion_source_commit"),
        },
    }
    if a3_evidence_path is not None:
        evidence_sha = _sha256_file(a3_evidence_path)
        if a3_artifacts.get("candidate_evidence_sha256") != evidence_sha:
            raise ValueError("A3 evidence hash mismatch")
        source_artifacts["a3_evidence"] = {
            "schema": "spectratrack-vnext-enhancement-alternates-v1",
            "sha256": evidence_sha,
        }

    report: dict[str, Any] = {
        "schema": DISCOVERY_SCHEMA,
        "discovery_only": True,
        "candidate_evaluation": "forbidden_on_r1_discovery_report",
        "report_revision": "nightowls-smoke400-discovery-r1",
        "a5_smoke": a5,
        "taxonomy_categories": list(TAXONOMY),
        "source_artifacts": source_artifacts,
        "baseline_control_totals": {
            "tp": control["tp"],
            "fp": control["fp"],
            "fn": control["fn"],
        },
        "baseline_failure_clusters": baseline_clusters,
        "a1_specific": {
            "source_disposition": "REJECT",
            "control_totals": {
                "tp": control["tp"],
                "fp": control["fp"],
                "fn": control["fn"],
                "duplicate_count_before_fusion": control[
                    "duplicate_count_before_fusion"
                ],
            },
            "candidate_totals": {
                "tp": candidate["tp"],
                "fp": candidate["fp"],
                "fn": candidate["fn"],
                "duplicate_count_before_fusion": candidate[
                    "duplicate_count_before_fusion"
                ],
            },
            "recovered_gt_count": len(recovered),
            "recovered_gt_representatives": _ranked_ids(
                recovered,
                gt_by_frame,
                match_iou=match_iou,
                top_k=top_k,
            ),
            "new_missed_gt_count": len(new_misses),
            "new_missed_gt_representatives": _ranked_ids(
                new_misses,
                gt_by_frame,
                match_iou=match_iou,
                top_k=top_k,
            ),
            "regression_clusters": a1_clusters,
        },
        "a3_specific": {
            "source_disposition": "REJECT",
            "operation": a3_final.get("candidate"),
            "findings": a3_findings,
        },
        "lineage_interface": {
            "schema": LINEAGE_SCHEMA,
            "allowed_roles": list(ALLOWED_LINEAGE_ROLES),
            "inspired_by": {
                "required_report_digest_field": "discovery_report_digest",
                "required_cluster_ids_field": "cluster_ids",
            },
        },
        "limits": {
            "top_k_representative_ids": top_k,
            "all_frame_rendering": False,
            "threshold_proposals": False,
            "threshold_scoring": False,
        },
    }
    report["report_digest"] = f"sha256:{_canonical_sha256(report)}"
    return report


def cluster_ids(report: dict[str, Any]) -> set[str]:
    ids: set[str] = set()
    for cluster in report.get("baseline_failure_clusters", []):
        if isinstance(cluster, dict) and isinstance(cluster.get("cluster_id"), str):
            ids.add(cluster["cluster_id"])
    a1 = report.get("a1_specific")
    if isinstance(a1, dict):
        for cluster in a1.get("regression_clusters", []):
            if isinstance(cluster, dict) and isinstance(cluster.get("cluster_id"), str):
                ids.add(cluster["cluster_id"])
    a3 = report.get("a3_specific")
    if isinstance(a3, dict):
        for finding in a3.get("findings", []):
            if isinstance(finding, dict) and isinstance(finding.get("finding_id"), str):
                ids.add(finding["finding_id"])
    return ids


def validate_hypothesis_lineage(
    record: dict[str, Any],
    discovery_report: dict[str, Any],
) -> dict[str, Any]:
    if record.get("schema") != LINEAGE_SCHEMA:
        raise ValueError(f"lineage schema must be {LINEAGE_SCHEMA}")
    role = record.get("role")
    if role not in ALLOWED_LINEAGE_ROLES:
        raise ValueError(f"lineage role must be one of {ALLOWED_LINEAGE_ROLES}")
    hypothesis_id = record.get("hypothesis_id")
    if not isinstance(hypothesis_id, str) or not hypothesis_id.strip():
        raise ValueError("lineage hypothesis_id must be non-empty")
    inspired_by = record.get("inspired_by")
    if not isinstance(inspired_by, dict):
        raise ValueError("lineage inspired_by must be an object")
    if inspired_by.get("discovery_report_digest") != discovery_report.get(
        "report_digest"
    ):
        raise ValueError("lineage discovery report digest mismatch")
    referenced = inspired_by.get("cluster_ids")
    if (
        not isinstance(referenced, list)
        or not referenced
        or not all(isinstance(item, str) and item for item in referenced)
        or len(set(referenced)) != len(referenced)
    ):
        raise ValueError("lineage cluster_ids must be a non-empty unique string list")
    missing = sorted(set(referenced) - cluster_ids(discovery_report))
    if missing:
        raise ValueError(f"lineage references unknown discovery clusters: {missing}")
    for field in ("hypothesis", "mechanism"):
        if not isinstance(record.get(field), str) or not record[field].strip():
            raise ValueError(f"lineage {field} must be non-empty")
    effects = record.get("expected_taxonomy_effects")
    if not isinstance(effects, list) or not effects:
        raise ValueError("lineage expected_taxonomy_effects must be non-empty")
    for index, effect in enumerate(effects):
        if not isinstance(effect, dict):
            raise ValueError(f"lineage effect {index} must be an object")
        if effect.get("category") not in TAXONOMY:
            raise ValueError(f"lineage effect {index} has unknown taxonomy category")
        if effect.get("direction") not in {"decrease", "unchanged", "increase"}:
            raise ValueError(f"lineage effect {index} has invalid direction")
    return {
        "valid": True,
        "schema": LINEAGE_SCHEMA,
        "hypothesis_id": hypothesis_id,
        "role": role,
        "discovery_report_digest": discovery_report["report_digest"],
        "cluster_ids": referenced,
        "record_digest": f"sha256:{_canonical_sha256(record)}",
    }


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Freeze discovery-only error taxonomy and validate hypothesis lineage"
    )
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build")
    build.add_argument("--manifest", required=True)
    build.add_argument("--ground-truth", required=True)
    build.add_argument("--a1-summary", required=True)
    build.add_argument("--a1-control", required=True)
    build.add_argument("--a1-candidate", required=True)
    build.add_argument("--a3-final", required=True)
    build.add_argument("--a3-score", required=True)
    build.add_argument("--a3-evidence")
    build.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    build.add_argument("--output", required=True)

    lineage = sub.add_parser("validate-lineage")
    lineage.add_argument("--report", required=True)
    lineage.add_argument("--record", required=True)
    lineage.add_argument("--output")
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.command == "build":
        report = build_discovery_report(
            manifest_path=args.manifest,
            ground_truth_path=args.ground_truth,
            a1_summary_path=args.a1_summary,
            a1_control_path=args.a1_control,
            a1_candidate_path=args.a1_candidate,
            a3_final_path=args.a3_final,
            a3_score_path=args.a3_score,
            a3_evidence_path=args.a3_evidence,
            top_k=args.top_k,
        )
        _write_json(args.output, report)
        print(
            json.dumps(
                {
                    "discovery_only": True,
                    "report_digest": report["report_digest"],
                    "baseline_clusters": len(report["baseline_failure_clusters"]),
                    "a1_regression_clusters": len(
                        report["a1_specific"]["regression_clusters"]
                    ),
                    "a3_findings": len(report["a3_specific"]["findings"]),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    report = _load_json(args.report)
    record = _load_json(args.record)
    result = validate_hypothesis_lineage(record, report)
    if args.output:
        _write_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
