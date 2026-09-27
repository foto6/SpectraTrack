from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from pathlib import Path
from typing import Any

from .integrity import sha256_file
from .vnext_qa import load_frozen_manifest

SMOKE_SCHEMA = "spectratrack-smoke-slice-v1"
PROOF_SCHEMA = "spectratrack-smoke-selection-proof-v1"
HASHES_SCHEMA = "spectratrack-smoke-hashes-v1"
SMOKE_REVISION = "nightowls-public-smoke400-r1"
SMOKE_SEED = "spectratrack-round2-nightowls-smoke400-v1"
PARENT_REVISION = "nightowls-public-slice5000-r1"
PARENT_CORPUS_SHA256 = "1ba30ef5adad0f6bedba4319d1c3b5f246d4c576c92b26f8b98b68b4b94a0ae8"
PARENT_FRAMES = 5000
SMOKE_FRAMES = 400

IDENTITY_FIELDS = ("video", "frame", "source", "source_frame", "source_sequence")

PROMOTION_POLICY = {
    "stage1": {
        "name": "SMOKE400",
        "purpose": "cheap deterministic candidate triage only",
        "obvious_reject": {
            "invalid_or_incomplete_run": True,
            "provenance_or_corpus_mismatch": True,
            "false_negative_rule": {
                "relative_multiplier_at_least": 2.0,
                "absolute_increase_at_least": 10,
                "require_both": True,
            },
            "false_positive_rule": {
                "relative_multiplier_at_least": 2.0,
                "absolute_increase_at_least": 10,
                "require_both": True,
            },
        },
        "pass_meaning": "survives triage; not a final quality claim",
    },
    "stage2": {
        "name": "FULL5000",
        "required_for": [
            "every smoke survivor",
            "every promotion candidate",
            "every genuinely ambiguous smoke result",
        ],
        "final_claims_require_full5000": True,
    },
}


def _canonical_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _canonical_sha256(payload: Any) -> str:
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _json_dump(path: str | Path, payload: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _read_jsonl(path: str | Path) -> tuple[list[str], list[dict[str, Any]]]:
    source = Path(path)
    lines = source.read_text(encoding="utf-8").splitlines()
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            raise ValueError(f"{source}:{line_number}: blank JSONL row")
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{source}:{line_number}: JSONL row must be an object")
        rows.append(value)
    return lines, rows


def _identity(row: dict[str, Any]) -> dict[str, Any]:
    identity = {field: row.get(field) for field in IDENTITY_FIELDS}
    if not isinstance(identity["video"], str) or not identity["video"]:
        raise ValueError("smoke selection requires non-empty video identity")
    if not isinstance(identity["frame"], int) or isinstance(identity["frame"], bool) or identity["frame"] < 0:
        raise ValueError("smoke selection requires non-negative integer frame identity")
    if not isinstance(identity["source"], str) or not identity["source"]:
        raise ValueError("smoke selection requires non-empty source identity")
    if (
        not isinstance(identity["source_frame"], int)
        or isinstance(identity["source_frame"], bool)
        or identity["source_frame"] < 0
    ):
        raise ValueError("smoke selection requires non-negative integer source_frame identity")
    if not isinstance(identity["source_sequence"], str) or not identity["source_sequence"]:
        raise ValueError("smoke selection requires non-empty source_sequence identity")
    return identity


def _rank(seed: str, stage: str, identity: dict[str, Any]) -> str:
    material = f"{seed}\0{stage}\0{_canonical_json(identity)}".encode("utf-8")
    return hashlib.sha256(material).hexdigest()


def select_parent_indexes(
    rows: list[dict[str, Any]],
    *,
    count: int = SMOKE_FRAMES,
    seed: str = SMOKE_SEED,
) -> dict[int, dict[str, str]]:
    if count <= 0:
        raise ValueError("smoke frame count must be > 0")
    if count > len(rows):
        raise ValueError("smoke frame count exceeds parent universe")
    identities = [_identity(row) for row in rows]
    identity_tokens = [_canonical_json(value) for value in identities]
    if len(set(identity_tokens)) != len(identity_tokens):
        raise ValueError("parent frame identities must be unique")

    groups: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        groups[str(row["video"])].append(index)
    if len(groups) > count:
        raise ValueError(
            f"smoke frame count {count} cannot cover all {len(groups)} parent logical sequences"
        )

    selected: dict[int, dict[str, str]] = {}
    for sequence in sorted(groups):
        chosen = min(
            groups[sequence],
            key=lambda index: (
                _rank(seed, "sequence-coverage", identities[index]),
                identity_tokens[index],
            ),
        )
        selected[chosen] = {
            "selected_via": "sequence_coverage",
            "rank_sha256": _rank(seed, "sequence-coverage", identities[chosen]),
        }

    remaining = [index for index in range(len(rows)) if index not in selected]
    remaining.sort(
        key=lambda index: (
            _rank(seed, "hash-fill", identities[index]),
            identity_tokens[index],
        )
    )
    for index in remaining[: count - len(selected)]:
        selected[index] = {
            "selected_via": "hash_fill",
            "rank_sha256": _rank(seed, "hash-fill", identities[index]),
        }
    if len(selected) != count:
        raise RuntimeError(f"smoke selection produced {len(selected)} rows instead of {count}")
    return selected


def _derive_smoke(
    parent_lines: list[str],
    parent_rows: list[dict[str, Any]],
    *,
    count: int,
    seed: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
    selected = select_parent_indexes(parent_rows, count=count, seed=seed)
    ordered = sorted(
        selected,
        key=lambda index: (
            str(parent_rows[index]["video"]),
            int(parent_rows[index]["frame"]),
            int(parent_rows[index]["source_frame"]),
            index,
        ),
    )
    next_frame: dict[str, int] = defaultdict(int)
    smoke_rows: list[dict[str, Any]] = []
    proof_entries: list[dict[str, Any]] = []
    for smoke_line_number, parent_index in enumerate(ordered, start=1):
        parent_row = parent_rows[parent_index]
        video = str(parent_row["video"])
        smoke_frame = next_frame[video]
        next_frame[video] += 1
        smoke_row = dict(parent_row)
        smoke_row["frame"] = smoke_frame
        smoke_rows.append(smoke_row)

        parent_identity = _identity(parent_row)
        smoke_identity = _identity(smoke_row)
        proof_entries.append(
            {
                "smoke_line_number": smoke_line_number,
                "smoke_frame": smoke_frame,
                "parent_line_number": parent_index + 1,
                "parent_frame": int(parent_row["frame"]),
                "video": video,
                "source": parent_row["source"],
                "source_frame": parent_row["source_frame"],
                "source_sequence": parent_row["source_sequence"],
                "parent_identity_sha256": _canonical_sha256(parent_identity),
                "smoke_identity_sha256": _canonical_sha256(smoke_identity),
                "parent_row_sha256": hashlib.sha256(
                    parent_lines[parent_index].encode("utf-8")
                ).hexdigest(),
                "selected_via": selected[parent_index]["selected_via"],
                "rank_sha256": selected[parent_index]["rank_sha256"],
            }
        )
    return smoke_rows, proof_entries, {
        "parent_sequences": len({str(row["video"]) for row in parent_rows}),
        "smoke_sequences": len({str(row["video"]) for row in smoke_rows}),
        "sequence_coverage_frames": sum(
            1 for entry in proof_entries if entry["selected_via"] == "sequence_coverage"
        ),
        "hash_fill_frames": sum(
            1 for entry in proof_entries if entry["selected_via"] == "hash_fill"
        ),
    }


def _serialize_rows(rows: list[dict[str, Any]]) -> str:
    return "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)


def _composition(rows: list[dict[str, Any]]) -> dict[str, int]:
    positive_frames = 0
    scored_person_objects = 0
    ignored_person_objects = 0
    total_objects = 0
    for row in rows:
        objects = row.get("objects", [])
        if not isinstance(objects, list):
            raise ValueError("objects must be a list")
        scored_here = 0
        for obj in objects:
            if not isinstance(obj, dict):
                raise ValueError("object rows must be objects")
            total_objects += 1
            if obj.get("label") != "person":
                continue
            if obj.get("ignore", False):
                ignored_person_objects += 1
            else:
                scored_person_objects += 1
                scored_here += 1
        if scored_here:
            positive_frames += 1
    return {
        "positive_frames": positive_frames,
        "true_negative_frames": len(rows) - positive_frames,
        "scored_person_objects": scored_person_objects,
        "ignored_person_objects": ignored_person_objects,
        "total_objects": total_objects,
    }


def _load_locked_parent(
    parent_manifest_path: str | Path,
    parent_ground_truth_path: str | Path,
    *,
    expected_parent_revision: str,
    expected_parent_corpus_sha256: str,
    expected_parent_frames: int,
) -> tuple[dict[str, Any], list[str], list[dict[str, Any]]]:
    manifest = load_frozen_manifest(parent_manifest_path)
    if manifest.get("revision") != expected_parent_revision:
        raise ValueError(
            f"parent revision mismatch: {manifest.get('revision')!r} != {expected_parent_revision!r}"
        )
    if manifest.get("corpus_sha256") != expected_parent_corpus_sha256:
        raise ValueError("parent corpus SHA-256 mismatch")
    golden = manifest.get("splits", {}).get("golden")
    if not isinstance(golden, dict):
        raise ValueError("parent manifest has no golden split")
    actual_gt_sha = sha256_file(parent_ground_truth_path)
    if golden.get("ground_truth_sha256") != actual_gt_sha:
        raise ValueError("parent ground-truth SHA-256 does not match frozen manifest")
    lines, rows = _read_jsonl(parent_ground_truth_path)
    if len(rows) != expected_parent_frames:
        raise ValueError(
            f"parent frame count mismatch: {len(rows)} != {expected_parent_frames}"
        )
    imports = manifest.get("dataset_imports", [])
    if imports and any(item.get("tracking_supported") is not False for item in imports):
        raise ValueError("locked parent must remain tracking_supported=false")
    return manifest, lines, rows


def build_smoke_slice(
    *,
    parent_manifest_path: str | Path,
    parent_ground_truth_path: str | Path,
    output_ground_truth_path: str | Path,
    output_manifest_path: str | Path,
    output_proof_path: str | Path,
    output_hashes_path: str | Path,
    revision: str = SMOKE_REVISION,
    seed: str = SMOKE_SEED,
    count: int = SMOKE_FRAMES,
    expected_parent_revision: str = PARENT_REVISION,
    expected_parent_corpus_sha256: str = PARENT_CORPUS_SHA256,
    expected_parent_frames: int = PARENT_FRAMES,
) -> dict[str, Any]:
    parent_manifest, parent_lines, parent_rows = _load_locked_parent(
        parent_manifest_path,
        parent_ground_truth_path,
        expected_parent_revision=expected_parent_revision,
        expected_parent_corpus_sha256=expected_parent_corpus_sha256,
        expected_parent_frames=expected_parent_frames,
    )
    smoke_rows, proof_entries, coverage = _derive_smoke(
        parent_lines, parent_rows, count=count, seed=seed
    )

    gt_path = Path(output_ground_truth_path)
    gt_path.parent.mkdir(parents=True, exist_ok=True)
    gt_path.write_text(_serialize_rows(smoke_rows), encoding="utf-8")
    gt_sha = sha256_file(gt_path)

    proof = {
        "schema": PROOF_SCHEMA,
        "revision": revision,
        "parent": {
            "revision": parent_manifest["revision"],
            "corpus_sha256": parent_manifest["corpus_sha256"],
            "ground_truth_sha256": sha256_file(parent_ground_truth_path),
            "frame_count": len(parent_rows),
        },
        "selection": {
            "seed": seed,
            "frame_count": count,
            "identity_fields": list(IDENTITY_FIELDS),
            "sequence_key": "video",
            "rule": (
                "select the lowest SHA-256 rank per logical video using metadata identity only, "
                "then fill remaining slots by global SHA-256 rank; never inspect objects, tags, "
                "detections, scores, or candidate outcomes"
            ),
            "selection_identity_sha256": _canonical_sha256(
                [
                    {
                        "video": entry["video"],
                        "parent_frame": entry["parent_frame"],
                        "source": entry["source"],
                        "source_frame": entry["source_frame"],
                        "source_sequence": entry["source_sequence"],
                    }
                    for entry in proof_entries
                ]
            ),
            **coverage,
        },
        "entries": proof_entries,
    }
    _json_dump(output_proof_path, proof)
    proof_sha = sha256_file(output_proof_path)

    composition = _composition(smoke_rows)
    compute = {
        "parent_frames": len(parent_rows),
        "smoke_frames": len(smoke_rows),
        "frame_fraction": len(smoke_rows) / len(parent_rows),
        "approx_variable_frame_compute_reduction": 1.0 - (len(smoke_rows) / len(parent_rows)),
        "frame_count_speedup_factor": len(parent_rows) / len(smoke_rows),
        "note": "Approximation excludes fixed startup/model-load and I/O overhead.",
    }
    manifest = {
        "schema": SMOKE_SCHEMA,
        "revision": revision,
        "parent": {
            "revision": parent_manifest["revision"],
            "corpus_sha256": parent_manifest["corpus_sha256"],
            "ground_truth_sha256": sha256_file(parent_ground_truth_path),
            "frame_count": len(parent_rows),
        },
        "selection": {
            "seed": seed,
            "frame_count": len(smoke_rows),
            "identity_fields": list(IDENTITY_FIELDS),
            "sequence_key": "video",
            "candidate_result_inputs": [],
            "ground_truth_labels_used_for_selection": False,
            "sequence_coverage_before_hash_fill": True,
            **coverage,
        },
        "tracking_supported": False,
        "artifacts": {
            "ground_truth": gt_path.name,
            "ground_truth_sha256": gt_sha,
            "selection_proof": Path(output_proof_path).name,
            "selection_proof_sha256": proof_sha,
        },
        "composition": composition,
        "promotion_policy": PROMOTION_POLICY,
        "compute_savings": compute,
    }
    manifest["corpus_sha256"] = _canonical_sha256(manifest)
    _json_dump(output_manifest_path, manifest)
    manifest_file_sha = sha256_file(output_manifest_path)

    hashes = {
        "schema": HASHES_SCHEMA,
        "revision": revision,
        "parent_corpus_sha256": parent_manifest["corpus_sha256"],
        "smoke_corpus_sha256": manifest["corpus_sha256"],
        "files": {
            "ground_truth": {
                "name": gt_path.name,
                "bytes": gt_path.stat().st_size,
                "sha256": gt_sha,
            },
            "manifest": {
                "name": Path(output_manifest_path).name,
                "bytes": Path(output_manifest_path).stat().st_size,
                "sha256": manifest_file_sha,
            },
            "selection_proof": {
                "name": Path(output_proof_path).name,
                "bytes": Path(output_proof_path).stat().st_size,
                "sha256": proof_sha,
            },
        },
    }
    hashes["hash_bundle_sha256"] = _canonical_sha256(hashes)
    _json_dump(output_hashes_path, hashes)
    return {
        "manifest": manifest,
        "proof": proof,
        "hashes": hashes,
    }


def validate_smoke_slice(
    *,
    parent_manifest_path: str | Path,
    parent_ground_truth_path: str | Path,
    smoke_ground_truth_path: str | Path,
    smoke_manifest_path: str | Path,
    selection_proof_path: str | Path,
    hashes_path: str | Path,
    expected_parent_revision: str = PARENT_REVISION,
    expected_parent_corpus_sha256: str = PARENT_CORPUS_SHA256,
    expected_parent_frames: int = PARENT_FRAMES,
) -> dict[str, Any]:
    parent_manifest, parent_lines, parent_rows = _load_locked_parent(
        parent_manifest_path,
        parent_ground_truth_path,
        expected_parent_revision=expected_parent_revision,
        expected_parent_corpus_sha256=expected_parent_corpus_sha256,
        expected_parent_frames=expected_parent_frames,
    )
    manifest = json.loads(Path(smoke_manifest_path).read_text(encoding="utf-8"))
    proof = json.loads(Path(selection_proof_path).read_text(encoding="utf-8"))
    hashes = json.loads(Path(hashes_path).read_text(encoding="utf-8"))

    manifest_unsigned = dict(manifest)
    recorded_corpus_sha = manifest_unsigned.pop("corpus_sha256", None)
    if _canonical_sha256(manifest_unsigned) != recorded_corpus_sha:
        raise ValueError("smoke manifest corpus SHA-256 mismatch")
    if manifest.get("schema") != SMOKE_SCHEMA:
        raise ValueError("unsupported smoke manifest schema")
    if manifest.get("tracking_supported") is not False:
        raise ValueError("smoke tracking_supported must remain false")
    if manifest.get("parent", {}).get("corpus_sha256") != parent_manifest["corpus_sha256"]:
        raise ValueError("smoke manifest is not bound to the locked parent corpus")

    count = int(manifest.get("selection", {}).get("frame_count", 0))
    seed = manifest.get("selection", {}).get("seed")
    if not isinstance(seed, str) or not seed:
        raise ValueError("smoke manifest seed is missing")
    smoke_lines, smoke_rows = _read_jsonl(smoke_ground_truth_path)
    if len(smoke_rows) != count:
        raise ValueError("smoke JSONL frame count does not match manifest")
    if count != SMOKE_FRAMES and expected_parent_frames == PARENT_FRAMES:
        raise ValueError(f"production Smoke400 must contain exactly {SMOKE_FRAMES} frames")

    expected_rows, expected_entries, coverage = _derive_smoke(
        parent_lines, parent_rows, count=count, seed=seed
    )
    expected_lines = _serialize_rows(expected_rows).splitlines()
    if smoke_lines != expected_lines:
        raise ValueError("smoke JSONL is not the deterministic regeneration of the locked parent")
    if proof.get("schema") != PROOF_SCHEMA or proof.get("entries") != expected_entries:
        raise ValueError("selection proof does not exactly map regenerated smoke frames to parent")
    if proof.get("selection", {}).get("frame_count") != count:
        raise ValueError("selection proof frame count mismatch")

    frame_tokens = [
        _canonical_json(
            {
                "video": row["video"],
                "frame": row["frame"],
                "source": row["source"],
                "source_frame": row["source_frame"],
                "source_sequence": row["source_sequence"],
            }
        )
        for row in smoke_rows
    ]
    if len(set(frame_tokens)) != count:
        raise ValueError("smoke JSONL contains duplicate canonical frame identities")

    expected_hashes = {
        "ground_truth": sha256_file(smoke_ground_truth_path),
        "manifest": sha256_file(smoke_manifest_path),
        "selection_proof": sha256_file(selection_proof_path),
    }
    for key, actual in expected_hashes.items():
        if hashes.get("files", {}).get(key, {}).get("sha256") != actual:
            raise ValueError(f"hash bundle mismatch for {key}")
    unsigned_hashes = dict(hashes)
    recorded_hash_bundle_sha = unsigned_hashes.pop("hash_bundle_sha256", None)
    if _canonical_sha256(unsigned_hashes) != recorded_hash_bundle_sha:
        raise ValueError("hash bundle canonical SHA-256 mismatch")
    if hashes.get("smoke_corpus_sha256") != recorded_corpus_sha:
        raise ValueError("hash bundle smoke corpus SHA-256 mismatch")

    return {
        "valid": True,
        "revision": manifest["revision"],
        "smoke_corpus_sha256": recorded_corpus_sha,
        "frame_count": count,
        "unique_frame_count": len(set(frame_tokens)),
        "foreign_frames": 0,
        "deterministic_regeneration": True,
        "candidate_result_inputs": manifest["selection"]["candidate_result_inputs"],
        "ground_truth_labels_used_for_selection": manifest["selection"][
            "ground_truth_labels_used_for_selection"
        ],
        "tracking_supported": manifest["tracking_supported"],
        "composition": _composition(smoke_rows),
        "sequence_coverage": coverage,
        "compute_savings": manifest["compute_savings"],
    }
