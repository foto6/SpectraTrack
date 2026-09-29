"""Fail-closed A3 R2 smoke400 preflight. Never reads R2 GT/image/result content."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

LOCK_SHA256 = "e042ffe79a5e8e45d770fee38b33f3b496dd7b04022a50470aa701e24f6b52ba"
LOCK_FIRST_COMMIT = "526be08432dc7a9316ae1f46c37f281d4cb27acc"
LOCK_HASH_COMMIT = "f09b4f2f6aed32a76904fddf9db3c9c65aff152e"
PRIOR_R1_SCORE_COMMIT = "993c4a6574d0a6298774a7937ba3daeca8bc2e30"
A5_R2_FREEZE_COMMIT = "2effaa3b9dc0cb5911cb3057b13e6976fbfb9381"
FIRST_OBSERVED_A1_R2_RESULT_COMMIT = "58047dbc336d9a9e3468d4afb1b17dc5683a69f4"

FREEZE = {
    "revision": "nightowls-public-smoke400-r2",
    "bundle_file_sha256": "185c79d00480c05600a7181e0bc514b4a560e4ab27f27af076bf33081b2e8c3a",
    "bundle_declared_sha256": "9f2afa165f294a4bce1e7c842b8a135a98876e1a0c4db4b0cbd0b8c583151cce",
    "corpus_sha256": "9d145b4dda780052388f3b663203c519ded48f61457adef14654f84fb5549eff",
    "parent_corpus_sha256": "1ba30ef5adad0f6bedba4319d1c3b5f246d4c576c92b26f8b98b68b4b94a0ae8",
    "files": {
        "ground_truth": ("nightowls-public-smoke400-r2.jsonl", "75eba2c9dc3b36d0a2389bfbb080a709621ef690680ae655d85d8732e7bc6097"),
        "manifest": ("nightowls-public-smoke400-r2.manifest.json", "9270d46c2776aa531e1b979a1a7ebc16eaed0f6483b2c1095da834af383c4e83"),
        "selection_proof": ("nightowls-public-smoke400-r2.selection-proof.json", "df11e9dc6ca1f63019cba071ed82420de27c706b41d1d3b46df880bb3ce50faf"),
    },
}

GT_RULES = {
    "scorer": "canonical A5 NightOwls detection-only post-fusion scorer",
    "scored_label": "person / official scored pedestrian only",
    "ignore": "official ignore objects excluded under A5 scorer; never relabel as GT",
    "riders": "not promoted to scored pedestrian",
    "positives": "TP and FN from A5 scored pedestrian instances",
    "negatives": "negative-frame unmatched predictions contribute to FP, never TP",
    "tracking": "no temporal tracking/jitter claims",
    "runtime_hints": "GT tags/objects/ignore annotations forbidden for ROI routing",
    "content_read_at_preflight": False,
}


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for part in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def canonical_sha(lock: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(lock, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def git(repo: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repo), *args],
        text=True, capture_output=True, check=True,
    )
    return completed.stdout.strip()


def ancestor(repo: Path, older: str, younger: str) -> bool:
    result = subprocess.run(
        ["git", "-C", str(repo), "merge-base", "--is-ancestor", older, younger],
        capture_output=True, check=False,
    )
    if result.returncode not in (0, 1):
        raise ValueError(f"cannot verify ancestry: {older} -> {younger}")
    return result.returncode == 0


def lock_preflight(repo: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    base = repo / "pc/benchmarks/vnext/enhancement"
    lock_path = base / "r2_hypothesis_lock.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    actual = canonical_sha(lock)
    recorded = (base / "r2_hypothesis_lock.sha256").read_text(encoding="utf-8")
    if actual != LOCK_SHA256 or recorded != f"{LOCK_SHA256}  r2_hypothesis_lock.json\n":
        raise ValueError("immutable A3 R2 lock canonical SHA mismatch")
    if lock["immutability"]["artifact_is_read_only_for_this_r2_cycle"] is not True:
        raise ValueError("lock is not marked immutable")
    if lock["schema"] != "spectratrack-vnext-a3-r2-hypothesis-lock-v1":
        raise ValueError("unexpected R2 hypothesis schema")
    if not ancestor(repo, LOCK_FIRST_COMMIT, LOCK_HASH_COMMIT):
        raise ValueError("lock first commit not ancestor of canonical hash commit")
    head = git(repo, "rev-parse", "HEAD")
    if not ancestor(repo, LOCK_HASH_COMMIT, head):
        raise ValueError("canonical lock is not an ancestor of current HEAD")
    if not ancestor(repo, PRIOR_R1_SCORE_COMMIT, LOCK_FIRST_COMMIT):
        raise ValueError("prior smoke result not established as R1 antecedent")
    if not ancestor(repo, FIRST_OBSERVED_A1_R2_RESULT_COMMIT, "origin/agent/vnext-detection"):
        raise ValueError("cannot independently verify A1 R2 result ancestry")
    if not ancestor(repo, A5_R2_FREEZE_COMMIT, "origin/agent/vnext-qa"):
        raise ValueError("cannot independently verify A5 R2 freeze ancestry")
    times = {
        "prior_r1_smoke_score": git(repo, "show", "-s", "--format=%cI", PRIOR_R1_SCORE_COMMIT),
        "a5_r2_freeze": git(repo, "show", "-s", "--format=%cI", A5_R2_FREEZE_COMMIT),
        "a3_lock_first": git(repo, "show", "-s", "--format=%cI", LOCK_FIRST_COMMIT),
        "a3_lock_hash_final": git(repo, "show", "-s", "--format=%cI", LOCK_HASH_COMMIT),
        "first_observed_a1_r2_result": git(repo, "show", "-s", "--format=%cI", FIRST_OBSERVED_A1_R2_RESULT_COMMIT),
    }
    utc = {k: datetime.fromisoformat(v) for k, v in times.items()}
    if not (
        utc["prior_r1_smoke_score"] < utc["a3_lock_first"]
        and utc["a5_r2_freeze"] < utc["a3_lock_first"]
        and utc["a3_lock_first"] <= utc["a3_lock_hash_final"]
        and utc["a3_lock_hash_final"] < utc["first_observed_a1_r2_result"]
    ):
        raise ValueError("observed R2 lock/scoring chronology is inconsistent")
    paths = git(repo, "ls-tree", "-r", "--name-only", head, "pc/benchmarks/vnext/enhancement").splitlines()
    preexisting_r2_score_files = [
        p for p in paths if Path(p).name.startswith("r2_")
        and ("score" in Path(p).name or "result" in Path(p).name)
    ]
    if preexisting_r2_score_files:
        raise ValueError(f"unexpected A3 R2 score artifact in HEAD: {preexisting_r2_score_files}")
    return lock, {
        "verified_head_before_this_run": head,
        "lock_canonical_sha256": actual,
        "first_lock_commit": LOCK_FIRST_COMMIT,
        "canonical_hash_commit": LOCK_HASH_COMMIT,
        "r1_previous_score_commit": PRIOR_R1_SCORE_COMMIT,
        "a5_r2_freeze_commit": A5_R2_FREEZE_COMMIT,
        "first_observed_a1_r2_result_commit": FIRST_OBSERVED_A1_R2_RESULT_COMMIT,
        "commit_times_iso8601": times,
        "git_ancestry_verified": True,
        "a3_r2_scoring_artifacts_at_head": [],
        "scope": "committed A3 history and first published A1 R2 result; not an assertion about unknown uncommitted work",
    }


def freeze_preflight(directory: Path) -> dict[str, Any]:
    bundle_path = directory / "nightowls-public-smoke400-r2.hashes.json"
    if not bundle_path.is_file():
        raise ValueError(f"missing R2 frozen SHA bundle: {bundle_path}")
    actual_bundle_sha = digest(bundle_path)
    if actual_bundle_sha != FREEZE["bundle_file_sha256"]:
        raise ValueError("R2 SHA bundle bytes mismatch")
    metadata = json.loads(bundle_path.read_text(encoding="utf-8"))
    for name in ("revision", "corpus_sha256", "parent_corpus_sha256"):
        if metadata.get(name) != FREEZE[name]:
            raise ValueError(f"R2 freeze {name} mismatch")
    if metadata.get("schema") != "spectratrack-nightowls-blind-hashes-v1":
        raise ValueError("unsupported R2 hash-bundle schema")
    if metadata.get("hash_bundle_sha256") != FREEZE["bundle_declared_sha256"]:
        raise ValueError("published R2 bundle-internal SHA mismatch")
    files = {}
    for kind, (name, expected) in FREEZE["files"].items():
        location = directory / name
        if not location.is_file():
            raise ValueError(f"missing R2 {kind} file and SHA: {name} / {expected}")
        declared = metadata.get("files", {}).get(kind, {})
        if declared.get("name") != name or declared.get("sha256") != expected:
            raise ValueError(f"A5 metadata mismatch: {kind}")
        actual = digest(location)  # Byte hashing only: no R2 GT/manifest/selection rows parsed.
        if actual != expected or location.stat().st_size != declared.get("bytes"):
            raise ValueError(f"R2 {kind} bytes/SHA mismatch: {expected} vs {actual}")
        files[kind] = {"name": name, "sha256": actual, "bytes": location.stat().st_size}
    return {
        "revision": metadata["revision"],
        "corpus_sha256": metadata["corpus_sha256"],
        "parent_corpus_sha256": metadata["parent_corpus_sha256"],
        "bundle_file_sha256": actual_bundle_sha,
        "bundle_declared_sha256": metadata["hash_bundle_sha256"],
        "files": files,
        "disjointness": "A5 published freeze/selection-proof SHA verified; frame identities not opened under NO_NEW_CANDIDATE lock",
        "semantic_gt_or_image_or_result_inspection": False,
    }


def calculate_pair_delta(off: dict[str, Any], candidate: dict[str, Any], gates: dict[str, Any]) -> dict[str, Any]:
    """Pure post-fusion comparison for synthetic unit tests or a separately authorized future lock."""
    required_policy = (
        "corpus_sha256", "ground_truth_sha256", "model_sha256", "provider",
        "frame_selection_sha256", "detector_policy_sha256", "fusion_policy_sha256",
        "scoring_policy_sha256", "gt_rule_version",
    )
    for field in required_policy:
        if not off.get(field) or off[field] != candidate.get(field):
            raise ValueError(f"baseline/candidate provenance mismatch: {field}")
    if off["corpus_sha256"] != FREEZE["corpus_sha256"]:
        raise ValueError("not the exact disjoint R2 smoke400 corpus")
    for result in (off, candidate):
        positive, negative = result["positive"], result["negative"]
        if positive["frames"] + negative["frames"] != 400:
            raise ValueError("R2 smoke must contain exactly 400 positive/negative frames")
        if result["tp"] != positive["tp"] or result["fn"] != positive["fn"]:
            raise ValueError("GT TP/FN must originate only from positive scored pedestrian instances")
        if result["fp"] != positive["fp"] + negative["fp"]:
            raise ValueError("FP must account for both positive and true-negative frames")
        if len(set(result["matched_gt_ids"])) != result["tp"]:
            raise ValueError("non-unique or incomplete matched scored GT IDs")
    if off["positive"]["frames"] != candidate["positive"]["frames"]:
        raise ValueError("different positive-frame population")
    if off["negative"]["frames"] != candidate["negative"]["frames"]:
        raise ValueError("different true-negative-frame population")
    if off["tp"] + off["fn"] != candidate["tp"] + candidate["fn"]:
        raise ValueError("different scored GT population")
    if off["enhancement_onnx_calls"] != 0 or candidate["raw_onnx_calls"] < off["raw_onnx_calls"]:
        raise ValueError("baseline enhancement or raw ONNX accounting invalid")
    added_raw = candidate["raw_onnx_calls"] - off["raw_onnx_calls"]
    added_enhanced = candidate["enhancement_onnx_calls"]
    if added_raw < 0 or added_enhanced < 0 or added_enhanced > 400 or added_raw > 400:
        raise ValueError("locked cap / additional inference accounting violated")
    if candidate["selected_roi_count"] > 400 or candidate["max_rois_per_frame"] > 1:
        raise ValueError("at most one enhancement ROI per frame")
    if candidate["accepted_alternates"] != candidate["raw_corroborated_alternates"]:
        raise ValueError("enhanced-only or unsupported alternate")
    recovered_ids = set(candidate["matched_gt_ids"]) - set(off["matched_gt_ids"])
    lost_ids = set(off["matched_gt_ids"]) - set(candidate["matched_gt_ids"])
    added_calls = added_raw + added_enhanced
    def pr(r: dict[str, Any]) -> tuple[float, float, float]:
        p = r["tp"] / (r["tp"] + r["fp"]) if r["tp"] + r["fp"] else 0.0
        rec = r["tp"] / (r["tp"] + r["fn"]) if r["tp"] + r["fn"] else 0.0
        f1 = 2 * p * rec / (p + rec) if p + rec else 0.0
        return p, rec, f1
    p0, r0, f0 = pr(off)
    p1, r1, f1 = pr(candidate)
    checks = {
        "recovered_gt": len(recovered_ids) >= gates["minimum_recovered_gt"],
        "lost_gt": len(lost_ids) == gates["lost_gt_must_equal"],
        "net_tp": candidate["tp"] - off["tp"] >= gates["net_tp_gain_minimum"],
        "recall": r1 > r0,
        "f1": f1 > f0,
        "precision": p1 >= p0 - gates["maximum_absolute_precision_drop"],
        "fp": candidate["fp"] - off["fp"] <= gates["maximum_fp_increase"],
        "recovery_per_added_call": bool(added_calls) and len(recovered_ids) / added_calls >= gates["minimum_recovered_gt_per_added_detector_call"],
        "raw_corroboration": candidate["accepted_alternates"] == candidate["raw_corroborated_alternates"],
    }
    return {
        "decision": "PROMOTE_TO_FULL_REQUEST_ONLY_NO_AUTOMATIC_FULL5000"
        if all(checks.values()) else "REJECT_WITHOUT_FULL5000",
        "gate_results": checks,
        "off": {"tp": off["tp"], "fp": off["fp"], "fn": off["fn"], "precision": p0, "recall": r0, "f1": f0},
        "candidate": {"tp": candidate["tp"], "fp": candidate["fp"], "fn": candidate["fn"], "precision": p1, "recall": r1, "f1": f1},
        "positive_delta": {k: candidate["positive"][k] - off["positive"][k] for k in ("tp", "fp", "fn")},
        "negative_delta": {"fp": candidate["negative"]["fp"] - off["negative"]["fp"]},
        "recovered_gt": len(recovered_ids), "lost_gt": len(lost_ids),
        "net_fp_delta": candidate["fp"] - off["fp"], "net_fn_delta": candidate["fn"] - off["fn"],
        "added_raw_onnx_calls": added_raw, "extra_enhancement_onnx_calls": added_enhanced,
        "total_extra_onnx_calls": added_calls,
        "recovered_gt_per_extra_call": len(recovered_ids) / added_calls if added_calls else None,
        "fp_cost_per_recovered_gt": (candidate["fp"] - off["fp"]) / len(recovered_ids) if recovered_ids else None,
        "incremental_wall_seconds": candidate["wall_seconds"] - off["wall_seconds"],
    }


def preflight(repo: Path, frozen: Path) -> dict[str, Any]:
    evidence: dict[str, Any] = {
        "schema": "spectratrack-a3-r2-blind-smoke400-preflight-v1",
        "production_enhancement": "OFF",
        "expected_r2_revision": FREEZE["revision"],
        "expected_sha256": FREEZE,
        "gt_rules": GT_RULES,
        "r2_gt_image_result_content_read": False,
        "r2_detector_calls_executed": 0,
        "r2_scoring_executed": False,
        "metrics": {
            "off": None, "candidate": None, "positive_delta": None, "negative_delta": None,
            "precision": None, "recall": None, "tp": None, "fp": None, "fn": None,
            "extra_onnx_calls": None, "wall_seconds": None,
        },
    }
    try:
        lock, chronology = lock_preflight(repo)
        evidence["lock"] = {
            "canonical_sha256": canonical_sha(lock),
            "decision": lock["decision"],
            "locked_candidate": lock["locked_candidate"],
            "execution_policy": lock["r2_execution_policy"],
            "reserved_reject_gates": lock["reserved_r2_reject_promote_conditions"],
        }
        evidence["chronology"] = chronology
        evidence["freeze"] = freeze_preflight(frozen)
    except (OSError, ValueError, KeyError, json.JSONDecodeError, subprocess.CalledProcessError) as exc:
        evidence.update(status="BLOCKED_PREFLIGHT_INTEGRITY", blocker=str(exc))
        return evidence
    if (
        lock["decision"] == "NO_NEW_CANDIDATE"
        or lock["locked_candidate"] is None
        or lock["r2_execution_policy"]["r2_scoring_allowed"] is not True
        or lock["r2_execution_policy"]["max_added_detector_calls_for_this_lock"] == 0
    ):
        evidence.update(
            status="BLOCKED_NO_AUTHORIZED_R2_CANDIDATE",
            blocker="Immutable pre-data R2 lock has locked_candidate=null and explicitly forbids R2 scoring/exposure and extra detector calls. No superseding pre-exposure lock exists on this A3 branch.",
            missing_sha256=[],
        )
        return evidence
    # A future lock cannot be substituted into this script: exact lock SHA is pinned above.
    evidence.update(status="BLOCKED_UNEXPECTED_LOCK_STATE", blocker="Refuse any implicit unlock or post-exposure candidate admission")
    return evidence


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--frozen-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = preflight(args.repo_root, args.frozen_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, sort_keys=True, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "output": str(args.output), "artifact_sha256": digest(args.output)}, sort_keys=True))
    return 2  # BLOCKED is deliberate: never pretend a held-out evaluation ran.


if __name__ == "__main__":
    raise SystemExit(main())
