from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from spectratrack.research import r2_blind_smoke400 as r2


def _lock() -> dict:
    path = Path(__file__).resolve().parents[1] / "benchmarks/vnext/enhancement/r2_hypothesis_lock.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _candidate(*, tp: int, positive_fp: int, negative_fp: int, fn: int, matches: list[str],
               raw_calls: int, enhanced_calls: int, accepted: int, wall: float = 105.0) -> dict:
    provenance = {
        "corpus_sha256": r2.FREEZE["corpus_sha256"],
        "ground_truth_sha256": "same-frozen-gt",
        "model_sha256": "same-model",
        "provider": "same-provider",
        "frame_selection_sha256": "same-selection",
        "detector_policy_sha256": "same-detector",
        "fusion_policy_sha256": "same-fusion",
        "scoring_policy_sha256": "same-A5-scoring",
        "gt_rule_version": "A5-pedestrian-ignore-v1",
    }
    return {
        **provenance,
        "tp": tp, "fp": positive_fp + negative_fp, "fn": fn,
        "matched_gt_ids": matches,
        "positive": {"frames": 100, "tp": tp, "fp": positive_fp, "fn": fn},
        "negative": {"frames": 300, "fp": negative_fp},
        "raw_onnx_calls": raw_calls, "enhancement_onnx_calls": enhanced_calls,
        "selected_roi_count": 10, "max_rois_per_frame": 1,
        "accepted_alternates": accepted, "raw_corroborated_alternates": accepted,
        "wall_seconds": wall,
    }


def test_pinned_r2_digests_are_full_sha256_values() -> None:
    values = [
        r2.LOCK_SHA256,
        r2.FREEZE["bundle_file_sha256"],
        r2.FREEZE["bundle_declared_sha256"],
        r2.FREEZE["corpus_sha256"],
        r2.FREEZE["parent_corpus_sha256"],
        *(item[1] for item in r2.FREEZE["files"].values()),
    ]
    assert all(len(value) == 64 and all(c in "0123456789abcdef" for c in value) for value in values)


def test_existing_r2_lock_is_canonically_pinned_and_no_candidate() -> None:
    lock = _lock()
    assert r2.canonical_sha(lock) == r2.LOCK_SHA256
    assert lock["decision"] == "NO_NEW_CANDIDATE"
    assert lock["locked_candidate"] is None
    assert lock["r2_execution_policy"]["r2_scoring_allowed"] is False
    assert lock["r2_execution_policy"]["max_added_detector_calls_for_this_lock"] == 0
    assert lock["production_enhancement"] == "OFF"


def test_freeze_preflight_checks_sha_without_parsing_gt_images_or_selection(tmp_path, monkeypatch) -> None:
    entries: dict[str, tuple[str, str]] = {}
    metadata: dict = {
        "schema": "spectratrack-nightowls-blind-hashes-v1",
        "revision": r2.FREEZE["revision"],
        "corpus_sha256": r2.FREEZE["corpus_sha256"],
        "parent_corpus_sha256": r2.FREEZE["parent_corpus_sha256"],
        "hash_bundle_sha256": "synthetic-metadata-only",
        "files": {},
    }
    for label, (name, _old_sha) in r2.FREEZE["files"].items():
        # Intentionally invalid JSON for GT/manifest/proof: preflight may only byte-hash these.
        payload = b"not json; do not parse held-out data" + label.encode()
        path = tmp_path / name
        path.write_bytes(payload)
        sha = hashlib.sha256(payload).hexdigest()
        entries[label] = (name, sha)
        metadata["files"][label] = {"name": name, "sha256": sha, "bytes": len(payload)}
    bundle = tmp_path / "nightowls-public-smoke400-r2.hashes.json"
    bundle.write_text(json.dumps(metadata, sort_keys=True), encoding="utf-8")
    modified = copy.deepcopy(r2.FREEZE)
    modified["files"] = entries
    modified["bundle_declared_sha256"] = "synthetic-metadata-only"
    modified["bundle_file_sha256"] = r2.digest(bundle)
    monkeypatch.setattr(r2, "FREEZE", modified)
    checked = r2.freeze_preflight(tmp_path)
    assert checked["revision"] == "nightowls-public-smoke400-r2"
    assert checked["semantic_gt_or_image_or_result_inspection"] is False
    assert checked["files"]["ground_truth"]["sha256"] == entries["ground_truth"][1]


def test_freeze_preflight_refuses_missing_or_changed_gt_sha(tmp_path, monkeypatch) -> None:
    modified = copy.deepcopy(r2.FREEZE)
    metadata = {
        "schema": "spectratrack-nightowls-blind-hashes-v1",
        "revision": modified["revision"],
        "corpus_sha256": modified["corpus_sha256"],
        "parent_corpus_sha256": modified["parent_corpus_sha256"],
        "hash_bundle_sha256": modified["bundle_declared_sha256"],
        "files": {
            kind: {"name": name, "sha256": digest, "bytes": 12}
            for kind, (name, digest) in modified["files"].items()
        },
    }
    bundle = tmp_path / "nightowls-public-smoke400-r2.hashes.json"
    bundle.write_text(json.dumps(metadata), encoding="utf-8")
    modified["bundle_file_sha256"] = r2.digest(bundle)
    monkeypatch.setattr(r2, "FREEZE", modified)
    with pytest.raises(ValueError, match="missing R2 ground_truth file and SHA"):
        r2.freeze_preflight(tmp_path)
    (tmp_path / modified["files"]["ground_truth"][0]).write_bytes(b"corrupt bytes")
    with pytest.raises(ValueError, match="R2 ground_truth bytes/SHA mismatch"):
        r2.freeze_preflight(tmp_path)


def test_preflight_blocks_before_any_scoring_or_gt_content_access(tmp_path, monkeypatch) -> None:
    lock = _lock()
    monkeypatch.setattr(r2, "lock_preflight", lambda _repo: (lock, {"git_ancestry_verified": True}))
    monkeypatch.setattr(r2, "freeze_preflight", lambda _frozen: {"revision": r2.FREEZE["revision"], "exact_sha_verified": True})
    result = r2.preflight(tmp_path, tmp_path)
    assert result["status"] == "BLOCKED_NO_AUTHORIZED_R2_CANDIDATE"
    assert result["missing_sha256"] == []
    assert result["r2_detector_calls_executed"] == 0
    assert result["r2_scoring_executed"] is False
    assert result["r2_gt_image_result_content_read"] is False
    assert result["metrics"]["off"] is None
    assert result["metrics"]["candidate"] is None


def test_blind_pair_math_reject_gate_and_positive_negative_deltas() -> None:
    baseline = _candidate(tp=8, positive_fp=2, negative_fp=5, fn=2,
                          matches=[f"gt-{i}" for i in range(8)],
                          raw_calls=1200, enhanced_calls=0, accepted=0, wall=100)
    rejected = _candidate(tp=8, positive_fp=2, negative_fp=4, fn=2,
                          matches=[f"gt-{i}" for i in range(8)],
                          raw_calls=1210, enhanced_calls=10, accepted=1)
    gate = _lock()["reserved_r2_reject_promote_conditions"]
    result = r2.calculate_pair_delta(baseline, rejected, gate)
    assert result["decision"] == "REJECT_WITHOUT_FULL5000"
    assert result["positive_delta"] == {"tp": 0, "fp": 0, "fn": 0}
    assert result["negative_delta"] == {"fp": -1}
    assert result["net_fp_delta"] == -1
    assert result["recovered_gt"] == 0
    assert result["added_raw_onnx_calls"] == 10
    assert result["extra_enhancement_onnx_calls"] == 10
    assert result["total_extra_onnx_calls"] == 20
    assert result["recovered_gt_per_extra_call"] == 0
    assert result["fp_cost_per_recovered_gt"] is None
    assert result["gate_results"]["recovered_gt"] is False
    assert result["incremental_wall_seconds"] == 5.0


def test_predeclared_gate_synthetic_positive_requires_all_conditions() -> None:
    baseline = _candidate(tp=8, positive_fp=2, negative_fp=5, fn=2,
                          matches=[f"gt-{i}" for i in range(8)],
                          raw_calls=1200, enhanced_calls=0, accepted=0, wall=100)
    successful = _candidate(tp=10, positive_fp=2, negative_fp=5, fn=0,
                            matches=[f"gt-{i}" for i in range(10)],
                            raw_calls=1210, enhanced_calls=10, accepted=2)
    result = r2.calculate_pair_delta(
        baseline, successful, _lock()["reserved_r2_reject_promote_conditions"]
    )
    assert result["recovered_gt"] == 2 and result["lost_gt"] == 0
    assert result["positive_delta"] == {"tp": 2, "fp": 0, "fn": -2}
    assert result["negative_delta"] == {"fp": 0}
    assert result["recovered_gt_per_extra_call"] == pytest.approx(0.1)
    assert all(result["gate_results"].values())
    assert result["decision"] == "PROMOTE_TO_FULL_REQUEST_ONLY_NO_AUTOMATIC_FULL5000"


def test_pair_refuses_changed_policy_bad_gt_population_and_missing_raw_support() -> None:
    baseline = _candidate(tp=8, positive_fp=2, negative_fp=5, fn=2,
                          matches=[f"gt-{i}" for i in range(8)],
                          raw_calls=1200, enhanced_calls=0, accepted=0)
    candidate = _candidate(tp=8, positive_fp=2, negative_fp=4, fn=2,
                           matches=[f"gt-{i}" for i in range(8)],
                           raw_calls=1210, enhanced_calls=10, accepted=1)
    gates = _lock()["reserved_r2_reject_promote_conditions"]
    changed = copy.deepcopy(candidate)
    changed["fusion_policy_sha256"] = "different-fusion"
    with pytest.raises(ValueError, match="fusion_policy_sha256"):
        r2.calculate_pair_delta(baseline, changed, gates)
    changed = copy.deepcopy(candidate)
    changed["negative"]["fp"] = 5
    with pytest.raises(ValueError, match="FP must account"):
        r2.calculate_pair_delta(baseline, changed, gates)
    changed = copy.deepcopy(candidate)
    changed["raw_corroborated_alternates"] = 0
    with pytest.raises(ValueError, match="enhanced-only"):
        r2.calculate_pair_delta(baseline, changed, gates)
    changed = copy.deepcopy(candidate)
    changed["max_rois_per_frame"] = 2
    with pytest.raises(ValueError, match="at most one"):
        r2.calculate_pair_delta(baseline, changed, gates)
