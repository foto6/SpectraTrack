import hashlib
import json
from pathlib import Path

import pytest

from spectratrack.discovery_taxonomy import (
    LINEAGE_SCHEMA,
    build_discovery_report,
    validate_hypothesis_lineage,
)


def _write_json(path: Path, payload) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _write_jsonl(path: Path, rows) -> Path:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    return path


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _a1_artifact(
    gt_sha: str,
    *,
    source_commit: str,
    misses_by_video: dict[str, list[str]],
    fp_by_video: dict[str, int],
    duplicate_by_video: dict[str, int],
) -> dict:
    per_video = {}
    tp = fp = fn = 0
    for video in sorted(misses_by_video):
        misses = misses_by_video[video]
        frame_tp = 1 - len(misses)
        frame_fp = fp_by_video[video]
        frame_fn = len(misses)
        tp += frame_tp
        fp += frame_fp
        fn += frame_fn
        per_video[video] = {
            "hard-nms": {
                "tp": frame_tp,
                "fp": frame_fp,
                "fn": frame_fn,
                "duplicate_count_before_fusion": duplicate_by_video[video],
                "error_taxonomy": {
                    "representative_false_negatives": misses,
                },
            }
        }
    return {
        "schema": "spectratrack-vnext-fusion-corpus-v1",
        "source_commit": source_commit,
        "corpus_revision": "smoke-r1",
        "ground_truth_sha256": gt_sha,
        "frame_count": 2,
        "settings": {"match_iou": 0.5},
        "per_video": per_video,
    }


def _fixture(tmp_path: Path):
    gt = _write_jsonl(
        tmp_path / "smoke.jsonl",
        [
            {
                "video": "a.mp4",
                "frame": 0,
                "tags": ["night_dark"],
                "objects": [
                    {
                        "id": "p0",
                        "label": "person",
                        "bbox": [10, 10, 20, 40],
                        "attributes": ["small"],
                    }
                ],
            },
            {
                "video": "b.mp4",
                "frame": 0,
                "tags": ["night_dark"],
                "objects": [
                    {
                        "id": "p1",
                        "label": "person",
                        "bbox": [20, 20, 50, 100],
                        "attributes": ["nightowls_occluded_true"],
                    }
                ],
            },
        ],
    )
    gt_sha = _sha(gt)
    manifest = _write_json(
        tmp_path / "manifest.json",
        {
            "schema": "spectratrack-smoke-slice-v1",
            "revision": "smoke-r1",
            "corpus_sha256": "c" * 64,
            "artifacts": {
                "ground_truth": "smoke.jsonl",
                "ground_truth_sha256": gt_sha,
                "selection_proof": "proof.json",
                "selection_proof_sha256": "d" * 64,
            },
            "selection": {"frame_count": 2},
        },
    )
    source_commit = "1" * 40
    control = _write_json(
        tmp_path / "control.json",
        _a1_artifact(
            gt_sha,
            source_commit=source_commit,
            misses_by_video={
                "a.mp4": ["a.mp4#0#index:0"],
                "b.mp4": [],
            },
            fp_by_video={"a.mp4": 1, "b.mp4": 1},
            duplicate_by_video={"a.mp4": 1, "b.mp4": 0},
        ),
    )
    candidate = _write_json(
        tmp_path / "candidate.json",
        _a1_artifact(
            gt_sha,
            source_commit=source_commit,
            misses_by_video={
                "a.mp4": [],
                "b.mp4": ["b.mp4#0#index:0"],
            },
            fp_by_video={"a.mp4": 2, "b.mp4": 1},
            duplicate_by_video={"a.mp4": 2, "b.mp4": 2},
        ),
    )
    summary = _write_json(
        tmp_path / "summary.json",
        {
            "schema": "spectratrack-a1-triage-result-v1",
            "source_commit": source_commit,
            "corpus_revision": "smoke-r1",
            "control_id": "control",
            "frame_selector": {
                "revision": "smoke-r1",
                "frame_count": 2,
                "manifest_sha256": "d" * 64,
            },
            "artifacts": {
                "control": {"result": {"sha256": _sha(control)}},
                "candidate": {"result": {"sha256": _sha(candidate)}},
            },
            "decisions": [
                {
                    "candidate_id": "candidate",
                    "recommendation": "REJECT",
                }
            ],
        },
    )
    a3_score = _write_json(
        tmp_path / "a3-score.json",
        {
            "schema": "spectratrack-vnext-a3-smoke400-score-v1",
            "a1_fusion_source_commit": source_commit,
        },
    )
    manifest_sha = _sha(manifest)
    a3_final = _write_json(
        tmp_path / "a3-final.json",
        {
            "schema": "spectratrack-vnext-a3-smoke400-final-v1",
            "decision": "REJECT",
            "candidate": "lime_maxrgb_bounded",
            "candidate_lock": {"commit": "3" * 40},
            "a5_smoke": {
                "revision": "smoke-r1",
                "manifest_sha256": manifest_sha,
                "corpus_sha256": "c" * 64,
                "ground_truth_sha256": gt_sha,
                "selection_proof_sha256": "d" * 64,
                "frames": 2,
            },
            "off": {
                "tp": 1,
                "fp": 2,
                "fn": 1,
                "recall": 0.5,
            },
            "candidate_result": {
                "tp": 1,
                "fp": 1,
                "fn": 1,
                "recall": 0.5,
                "recovered_gt": 0,
                "lost_gt": 0,
                "fp_delta": -1,
            },
            "compute": {
                "additional_raw_onnx_calls": 2,
                "extra_enhancement_onnx_calls": 3,
                "activation_rate_vs_smoke_frames": 0.5,
                "recovered_gt_per_extra_enhancement_call": 0.0,
            },
            "artifacts": {
                "candidate_score_sha256": _sha(a3_score),
            },
        },
    )
    return {
        "manifest": manifest,
        "gt": gt,
        "a1_summary": summary,
        "a1_control": control,
        "a1_candidate": candidate,
        "a3_final": a3_final,
        "a3_score": a3_score,
    }


def _build(paths):
    return build_discovery_report(
        manifest_path=paths["manifest"],
        ground_truth_path=paths["gt"],
        a1_summary_path=paths["a1_summary"],
        a1_control_path=paths["a1_control"],
        a1_candidate_path=paths["a1_candidate"],
        a3_final_path=paths["a3_final"],
        a3_score_path=paths["a3_score"],
        top_k=2,
    )


def test_discovery_report_is_deterministic_and_separates_sources(tmp_path: Path):
    paths = _fixture(tmp_path)
    first = _build(paths)
    second = _build(paths)
    assert first == second
    assert first["discovery_only"] is True
    assert first["report_digest"].startswith("sha256:")
    assert first["candidate_evaluation"] == "forbidden_on_r1_discovery_report"

    baseline = {item["cluster_id"]: item for item in first["baseline_failure_clusters"]}
    assert baseline["baseline:missed_small_person"]["count"] == 1
    assert baseline["baseline:low_contrast_night_darkness"]["count"] == 1
    assert baseline["baseline:unknown_false_positive"]["count"] == 2

    a1 = {item["cluster_id"]: item for item in first["a1_specific"]["regression_clusters"]}
    assert a1["a1_r1:new_miss:occlusion_crowd"]["count"] == 1
    assert a1["a1_r1:aggregate_fp_delta"]["count"] == 1
    assert a1["a1_r1:prefusion_duplicate_pressure"]["count"] == 3

    findings = {item["finding_id"]: item for item in first["a3_specific"]["findings"]}
    assert findings["a3_r1:no_recall_gain"]["recovered_gt"] == 0
    assert findings["a3_r1:compute_cost"]["extra_onnx_calls_total"] == 5
    assert findings["a3_r1:unlocalized_fp_reduction"]["false_positive_delta"] == -1


def test_lineage_must_bind_digest_and_known_cluster(tmp_path: Path):
    report = _build(_fixture(tmp_path))
    record = {
        "schema": LINEAGE_SCHEMA,
        "hypothesis_id": "a1-next-hypothesis",
        "role": "A1",
        "inspired_by": {
            "discovery_report_digest": report["report_digest"],
            "cluster_ids": [
                "baseline:missed_small_person",
                "a1_r1:aggregate_fp_delta",
            ],
        },
        "hypothesis": "Test a bounded mechanism without changing this discovery report.",
        "mechanism": "Future candidate-specific implementation.",
        "expected_taxonomy_effects": [
            {"category": "missed_small_person", "direction": "decrease"},
            {"category": "unknown", "direction": "unchanged"},
        ],
    }
    result = validate_hypothesis_lineage(record, report)
    assert result["valid"] is True
    assert result["discovery_report_digest"] == report["report_digest"]
    assert result["record_digest"].startswith("sha256:")

    bad_digest = json.loads(json.dumps(record))
    bad_digest["inspired_by"]["discovery_report_digest"] = "sha256:" + "0" * 64
    with pytest.raises(ValueError, match="digest mismatch"):
        validate_hypothesis_lineage(bad_digest, report)

    bad_cluster = json.loads(json.dumps(record))
    bad_cluster["inspired_by"]["cluster_ids"] = ["baseline:not-real"]
    with pytest.raises(ValueError, match="unknown discovery clusters"):
        validate_hypothesis_lineage(bad_cluster, report)
