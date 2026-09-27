from __future__ import annotations

from collections import defaultdict
import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from .integrity import sha256_file
from .smoke_gate import (
    IDENTITY_FIELDS,
    PARENT_CORPUS_SHA256,
    PARENT_FRAMES,
    PARENT_REVISION,
    PROOF_SCHEMA,
    SMOKE_FRAMES,
    _canonical_json,
    _canonical_sha256,
    _composition,
    _identity,
    _json_dump,
    _load_locked_parent,
    _serialize_rows,
)

BLIND_SCHEMA = "spectratrack-nightowls-blind-partition-v1"
BLIND_HASHES_SCHEMA = "spectratrack-nightowls-blind-hashes-v1"
R1_REVISION = "nightowls-public-smoke400-r1"
R1_CORPUS_SHA256 = "70c29ecd91ede9239ebed2949ea46e4b07b63e842aef630c0752ee41b9620162"
R1_GT_SHA256 = "45ba10895c8a98f433bc7bad8e488b311a31ad8acf8459656d0cd775fe3fc0e8"
R1_MANIFEST_FILE_SHA256 = "3f65339609de78e7ef255e94875727a4bb26c4769231763a2660e1bb8f15905e"
R1_PROOF_SHA256 = "0838ecfd1341eb5f0193ad369516098be5a7d8b53d77302f8e2e4a4089c34940"
R2_REVISION = "nightowls-public-smoke400-r2"
R2_SEED = "spectratrack-round2-nightowls-smoke400-r2-v1"
HOLDOUT_REVISION = "nightowls-public-holdout4200-r1"
HOLDOUT_FRAMES = 4200

STAGE_SEMANTICS = {
    "r1": {
        "revision": R1_REVISION,
        "role": "discovery/triage; now observed and not blind for r1-inspired hypotheses",
    },
    "r2": {
        "revision": R2_REVISION,
        "role": "blind validation for hypotheses inspired by r1",
        "pass_meaning": "blind subset evidence only; not the final NightOwls heldout claim",
    },
    "holdout": {
        "revision": HOLDOUT_REVISION,
        "role": "final disjoint NightOwls heldout after r1/r2 decisions are locked",
    },
    "full5000": {
        "revision": PARENT_REVISION,
        "role": (
            "post-decision characterization/aggregate only; not statistically independent "
            "after any constituent subsets have been observed"
        ),
    },
}


def _rank(seed: str, stage: str, identity: dict[str, Any]) -> str:
    material = f"{seed}\0{stage}\0{_canonical_json(identity)}".encode("utf-8")
    return hashlib.sha256(material).hexdigest()


def _validate_r1_frozen_inputs(
    *,
    r1_manifest_path: str | Path,
    r1_ground_truth_path: str | Path,
    r1_proof_path: str | Path,
) -> dict[str, Any]:
    if sha256_file(r1_manifest_path) != R1_MANIFEST_FILE_SHA256:
        raise ValueError("r1 frozen manifest file hash mismatch")
    if sha256_file(r1_ground_truth_path) != R1_GT_SHA256:
        raise ValueError("r1 frozen ground-truth file hash mismatch")
    if sha256_file(r1_proof_path) != R1_PROOF_SHA256:
        raise ValueError("r1 frozen selection-proof file hash mismatch")
    manifest = json.loads(Path(r1_manifest_path).read_text(encoding="utf-8"))
    if manifest.get("revision") != R1_REVISION:
        raise ValueError("r1 frozen manifest revision mismatch")
    if manifest.get("corpus_sha256") != R1_CORPUS_SHA256:
        raise ValueError("r1 frozen corpus SHA-256 mismatch")
    if manifest.get("tracking_supported") is not False:
        raise ValueError("r1 tracking_supported must remain false")
    if manifest.get("selection", {}).get("frame_count") != SMOKE_FRAMES:
        raise ValueError("r1 frozen manifest must contain 400 frames")
    artifacts = manifest.get("artifacts", {})
    if artifacts.get("ground_truth_sha256") != R1_GT_SHA256:
        raise ValueError("r1 manifest GT binding mismatch")
    if artifacts.get("selection_proof_sha256") != R1_PROOF_SHA256:
        raise ValueError("r1 manifest proof binding mismatch")
    return manifest


def _load_r1_parent_indexes(
    *,
    r1_proof_path: str | Path,
    parent_lines: list[str],
    parent_rows: list[dict[str, Any]],
    parent_corpus_sha256: str,
    parent_ground_truth_sha256: str,
) -> set[int]:
    proof = json.loads(Path(r1_proof_path).read_text(encoding="utf-8"))
    if proof.get("schema") != PROOF_SCHEMA:
        raise ValueError("r1 proof schema mismatch")
    if proof.get("revision") != R1_REVISION:
        raise ValueError("r1 proof revision mismatch")
    parent = proof.get("parent", {})
    if parent.get("corpus_sha256") != parent_corpus_sha256:
        raise ValueError("r1 proof parent corpus mismatch")
    if parent.get("ground_truth_sha256") != parent_ground_truth_sha256:
        raise ValueError("r1 proof parent GT mismatch")
    entries = proof.get("entries")
    if not isinstance(entries, list) or len(entries) != SMOKE_FRAMES:
        raise ValueError("r1 proof must contain exactly 400 entries")

    indexes: set[int] = set()
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("r1 proof entry must be an object")
        line_number = entry.get("parent_line_number")
        if (
            not isinstance(line_number, int)
            or isinstance(line_number, bool)
            or line_number < 1
            or line_number > len(parent_rows)
        ):
            raise ValueError("r1 proof parent line is invalid")
        index = line_number - 1
        if index in indexes:
            raise ValueError("r1 proof contains duplicate parent frame")
        row = parent_rows[index]
        if entry.get("video") != row.get("video"):
            raise ValueError("r1 proof video does not match parent")
        if entry.get("source") != row.get("source"):
            raise ValueError("r1 proof source does not match parent")
        if entry.get("source_frame") != row.get("source_frame"):
            raise ValueError("r1 proof source_frame does not match parent")
        row_sha = hashlib.sha256(parent_lines[index].encode("utf-8")).hexdigest()
        if entry.get("parent_row_sha256") != row_sha:
            raise ValueError("r1 proof parent row hash mismatch")
        indexes.add(index)
    return indexes


def select_r2_parent_indexes(
    rows: list[dict[str, Any]],
    *,
    excluded_indexes: set[int],
    count: int = SMOKE_FRAMES,
    seed: str = R2_SEED,
) -> tuple[dict[int, dict[str, Any]], list[str]]:
    if count != SMOKE_FRAMES:
        raise ValueError("r2 must contain exactly 400 frames")
    if any(index < 0 or index >= len(rows) for index in excluded_indexes):
        raise ValueError("excluded parent index is outside universe")
    identities = [_identity(row) for row in rows]
    tokens = [_canonical_json(identity) for identity in identities]
    if len(set(tokens)) != len(tokens):
        raise ValueError("parent frame identities must be unique")

    groups: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        groups[str(row["video"])].append(index)

    selected: dict[int, dict[str, Any]] = {}
    fallback_sequences: list[str] = []
    for sequence in sorted(groups):
        available = [index for index in groups[sequence] if index not in excluded_indexes]
        if not available:
            fallback_sequences.append(sequence)
            continue
        chosen = min(
            available,
            key=lambda index: (
                _rank(seed, "second-per-sequence", identities[index]),
                tokens[index],
            ),
        )
        selected[chosen] = {
            "selected_via": "second_per_sequence",
            "rank_sha256": _rank(seed, "second-per-sequence", identities[chosen]),
        }

    remaining = [
        index
        for index in range(len(rows))
        if index not in excluded_indexes and index not in selected
    ]
    remaining.sort(
        key=lambda index: (
            _rank(seed, "disjoint-hash-fill", identities[index]),
            tokens[index],
        )
    )
    need = count - len(selected)
    if need < 0 or need > len(remaining):
        raise ValueError("insufficient remaining parent frames for r2")
    for index in remaining[:need]:
        selected[index] = {
            "selected_via": "disjoint_hash_fill",
            "rank_sha256": _rank(seed, "disjoint-hash-fill", identities[index]),
        }
    if len(selected) != count:
        raise RuntimeError("r2 selection did not produce exactly 400 frames")
    if set(selected) & excluded_indexes:
        raise RuntimeError("r2 selection intersects excluded r1 frames")
    return selected, fallback_sequences


def _derive_rows_and_proof(
    *,
    parent_lines: list[str],
    parent_rows: list[dict[str, Any]],
    selected: dict[int, dict[str, Any]],
    output_kind: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
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
    output_rows: list[dict[str, Any]] = []
    entries: list[dict[str, Any]] = []
    for output_line_number, parent_index in enumerate(ordered, start=1):
        parent_row = parent_rows[parent_index]
        video = str(parent_row["video"])
        output_frame = next_frame[video]
        next_frame[video] += 1
        output_row = dict(parent_row)
        output_row["frame"] = output_frame
        output_rows.append(output_row)
        selection = selected[parent_index]
        entries.append(
            {
                "output_kind": output_kind,
                "output_line_number": output_line_number,
                "output_frame": output_frame,
                "parent_line_number": parent_index + 1,
                "parent_frame": int(parent_row["frame"]),
                "video": video,
                "source": parent_row["source"],
                "source_frame": parent_row["source_frame"],
                "source_sequence": parent_row["source_sequence"],
                "parent_identity_sha256": _canonical_sha256(_identity(parent_row)),
                "output_identity_sha256": _canonical_sha256(_identity(output_row)),
                "parent_row_sha256": hashlib.sha256(
                    parent_lines[parent_index].encode("utf-8")
                ).hexdigest(),
                "selected_via": selection["selected_via"],
                "rank_sha256": selection.get("rank_sha256"),
            }
        )
    return output_rows, entries


def _membership_sha(indexes: set[int]) -> str:
    return _canonical_sha256([index + 1 for index in sorted(indexes)])


def _write_artifact_set(
    *,
    revision: str,
    kind: str,
    rows: list[dict[str, Any]],
    proof: dict[str, Any],
    parent_manifest: dict[str, Any],
    parent_ground_truth_path: str | Path,
    output_ground_truth_path: str | Path,
    output_manifest_path: str | Path,
    output_proof_path: str | Path,
    output_hashes_path: str | Path,
    selection_metadata: dict[str, Any],
    partition_metadata: dict[str, Any],
) -> dict[str, Any]:
    gt_path = Path(output_ground_truth_path)
    gt_path.parent.mkdir(parents=True, exist_ok=True)
    gt_path.write_text(_serialize_rows(rows), encoding="utf-8")
    gt_sha = sha256_file(gt_path)

    _json_dump(output_proof_path, proof)
    proof_sha = sha256_file(output_proof_path)

    # Composition is intentionally derived only after the membership proof is frozen to disk.
    composition = _composition(rows)
    manifest = {
        "schema": BLIND_SCHEMA,
        "revision": revision,
        "kind": kind,
        "parent": {
            "revision": parent_manifest["revision"],
            "corpus_sha256": parent_manifest["corpus_sha256"],
            "ground_truth_sha256": sha256_file(parent_ground_truth_path),
            "frame_count": PARENT_FRAMES,
        },
        "selection": selection_metadata,
        "partition": partition_metadata,
        "tracking_supported": False,
        "artifacts": {
            "ground_truth": gt_path.name,
            "ground_truth_sha256": gt_sha,
            "selection_proof": Path(output_proof_path).name,
            "selection_proof_sha256": proof_sha,
        },
        "composition": composition,
        "composition_derived_after_selection_proof_sha256": proof_sha,
        "stage_semantics": STAGE_SEMANTICS,
    }
    manifest["corpus_sha256"] = _canonical_sha256(manifest)
    _json_dump(output_manifest_path, manifest)
    manifest_file_sha = sha256_file(output_manifest_path)

    hashes = {
        "schema": BLIND_HASHES_SCHEMA,
        "revision": revision,
        "parent_corpus_sha256": parent_manifest["corpus_sha256"],
        "corpus_sha256": manifest["corpus_sha256"],
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
    return {"manifest": manifest, "proof": proof, "hashes": hashes}


def build_blind_partition(
    *,
    parent_manifest_path: str | Path,
    parent_ground_truth_path: str | Path,
    r1_manifest_path: str | Path,
    r1_ground_truth_path: str | Path,
    r1_proof_path: str | Path,
    r2_ground_truth_path: str | Path,
    r2_manifest_path: str | Path,
    r2_proof_path: str | Path,
    r2_hashes_path: str | Path,
    holdout_ground_truth_path: str | Path,
    holdout_manifest_path: str | Path,
    holdout_proof_path: str | Path,
    holdout_hashes_path: str | Path,
) -> dict[str, Any]:
    parent_manifest, parent_lines, parent_rows = _load_locked_parent(
        parent_manifest_path,
        parent_ground_truth_path,
        expected_parent_revision=PARENT_REVISION,
        expected_parent_corpus_sha256=PARENT_CORPUS_SHA256,
        expected_parent_frames=PARENT_FRAMES,
    )
    parent_gt_sha = sha256_file(parent_ground_truth_path)
    _validate_r1_frozen_inputs(
        r1_manifest_path=r1_manifest_path,
        r1_ground_truth_path=r1_ground_truth_path,
        r1_proof_path=r1_proof_path,
    )
    r1_indexes = _load_r1_parent_indexes(
        r1_proof_path=r1_proof_path,
        parent_lines=parent_lines,
        parent_rows=parent_rows,
        parent_corpus_sha256=parent_manifest["corpus_sha256"],
        parent_ground_truth_sha256=parent_gt_sha,
    )
    r2_selected, fallback_sequences = select_r2_parent_indexes(
        parent_rows,
        excluded_indexes=r1_indexes,
    )
    r2_indexes = set(r2_selected)
    holdout_indexes = set(range(len(parent_rows))) - r1_indexes - r2_indexes
    if len(holdout_indexes) != HOLDOUT_FRAMES:
        raise RuntimeError("holdout complement must contain exactly 4200 frames")

    intersection_counts = {
        "r1_r2": len(r1_indexes & r2_indexes),
        "r1_holdout": len(r1_indexes & holdout_indexes),
        "r2_holdout": len(r2_indexes & holdout_indexes),
    }
    union_indexes = r1_indexes | r2_indexes | holdout_indexes
    partition_metadata = {
        "r1_count": len(r1_indexes),
        "r2_count": len(r2_indexes),
        "holdout_count": len(holdout_indexes),
        "parent_count": len(parent_rows),
        "intersections": intersection_counts,
        "union_count": len(union_indexes),
        "union_equals_parent": union_indexes == set(range(len(parent_rows))),
        "r1_parent_membership_sha256": _membership_sha(r1_indexes),
        "r2_parent_membership_sha256": _membership_sha(r2_indexes),
        "holdout_parent_membership_sha256": _membership_sha(holdout_indexes),
    }
    if any(intersection_counts.values()) or not partition_metadata["union_equals_parent"]:
        raise RuntimeError("r1/r2/holdout partition invariant failed")

    r2_rows, r2_entries = _derive_rows_and_proof(
        parent_lines=parent_lines,
        parent_rows=parent_rows,
        selected=r2_selected,
        output_kind="smoke400-r2",
    )
    r2_proof = {
        "schema": PROOF_SCHEMA,
        "revision": R2_REVISION,
        "parent": {
            "revision": parent_manifest["revision"],
            "corpus_sha256": parent_manifest["corpus_sha256"],
            "ground_truth_sha256": parent_gt_sha,
            "frame_count": len(parent_rows),
        },
        "selection": {
            "seed": R2_SEED,
            "frame_count": SMOKE_FRAMES,
            "identity_fields": list(IDENTITY_FIELDS),
            "excluded_revision": R1_REVISION,
            "excluded_proof_sha256": sha256_file(r1_proof_path),
            "rule": (
                "exclude exact r1 parent-line membership; choose one metadata-hash-ranked "
                "remaining frame per logical sequence where possible, then fill by global "
                "metadata hash rank from remaining frames; labels and candidate outcomes are forbidden"
            ),
            "fallback_sequences_without_remaining_frame": fallback_sequences,
            "fallback_count": len(fallback_sequences),
            "parent_sequence_count": len({str(row["video"]) for row in parent_rows}),
            "second_per_sequence_frames": sum(
                1 for entry in r2_entries if entry["selected_via"] == "second_per_sequence"
            ),
            "hash_fill_frames": sum(
                1 for entry in r2_entries if entry["selected_via"] == "disjoint_hash_fill"
            ),
            "candidate_result_inputs": [],
            "ground_truth_labels_used_for_selection": False,
            "selection_identity_sha256": _canonical_sha256(
                [
                    {
                        "video": entry["video"],
                        "parent_frame": entry["parent_frame"],
                        "source": entry["source"],
                        "source_frame": entry["source_frame"],
                        "source_sequence": entry["source_sequence"],
                    }
                    for entry in r2_entries
                ]
            ),
        },
        "partition": partition_metadata,
        "entries": r2_entries,
    }
    r2_result = _write_artifact_set(
        revision=R2_REVISION,
        kind="blind_smoke400_r2",
        rows=r2_rows,
        proof=r2_proof,
        parent_manifest=parent_manifest,
        parent_ground_truth_path=parent_ground_truth_path,
        output_ground_truth_path=r2_ground_truth_path,
        output_manifest_path=r2_manifest_path,
        output_proof_path=r2_proof_path,
        output_hashes_path=r2_hashes_path,
        selection_metadata=r2_proof["selection"],
        partition_metadata=partition_metadata,
    )

    holdout_selected = {
        index: {"selected_via": "exact_partition_remainder", "rank_sha256": None}
        for index in holdout_indexes
    }
    holdout_rows, holdout_entries = _derive_rows_and_proof(
        parent_lines=parent_lines,
        parent_rows=parent_rows,
        selected=holdout_selected,
        output_kind="holdout4200",
    )
    holdout_proof = {
        "schema": PROOF_SCHEMA,
        "revision": HOLDOUT_REVISION,
        "parent": {
            "revision": parent_manifest["revision"],
            "corpus_sha256": parent_manifest["corpus_sha256"],
            "ground_truth_sha256": parent_gt_sha,
            "frame_count": len(parent_rows),
        },
        "selection": {
            "seed": None,
            "frame_count": HOLDOUT_FRAMES,
            "identity_fields": list(IDENTITY_FIELDS),
            "rule": "exact set complement of frozen r1 and frozen r2 parent membership; no selection freedom",
            "excluded_revisions": [R1_REVISION, R2_REVISION],
            "candidate_result_inputs": [],
            "ground_truth_labels_used_for_selection": False,
        },
        "partition": partition_metadata,
        "entries": holdout_entries,
    }
    holdout_result = _write_artifact_set(
        revision=HOLDOUT_REVISION,
        kind="final_disjoint_holdout4200",
        rows=holdout_rows,
        proof=holdout_proof,
        parent_manifest=parent_manifest,
        parent_ground_truth_path=parent_ground_truth_path,
        output_ground_truth_path=holdout_ground_truth_path,
        output_manifest_path=holdout_manifest_path,
        output_proof_path=holdout_proof_path,
        output_hashes_path=holdout_hashes_path,
        selection_metadata=holdout_proof["selection"],
        partition_metadata=partition_metadata,
    )
    return {
        "r2": r2_result,
        "holdout": holdout_result,
        "partition": partition_metadata,
    }


def _validate_artifact_files(
    *,
    result: dict[str, Any],
    ground_truth_path: str | Path,
    manifest_path: str | Path,
    proof_path: str | Path,
    hashes_path: str | Path,
) -> None:
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    proof = json.loads(Path(proof_path).read_text(encoding="utf-8"))
    hashes = json.loads(Path(hashes_path).read_text(encoding="utf-8"))
    expected = result["manifest"]
    if manifest != expected:
        raise ValueError(f"{manifest.get('revision')}: manifest deterministic regeneration mismatch")
    if proof != result["proof"]:
        raise ValueError(f"{manifest.get('revision')}: proof deterministic regeneration mismatch")
    unsigned = dict(manifest)
    corpus_sha = unsigned.pop("corpus_sha256", None)
    if _canonical_sha256(unsigned) != corpus_sha:
        raise ValueError(f"{manifest.get('revision')}: corpus SHA mismatch")
    if manifest.get("tracking_supported") is not False:
        raise ValueError(f"{manifest.get('revision')}: tracking_supported must remain false")
    file_hashes = {
        "ground_truth": sha256_file(ground_truth_path),
        "manifest": sha256_file(manifest_path),
        "selection_proof": sha256_file(proof_path),
    }
    for key, actual in file_hashes.items():
        if hashes.get("files", {}).get(key, {}).get("sha256") != actual:
            raise ValueError(f"{manifest.get('revision')}: hash bundle mismatch for {key}")
    unsigned_hashes = dict(hashes)
    bundle_sha = unsigned_hashes.pop("hash_bundle_sha256", None)
    if _canonical_sha256(unsigned_hashes) != bundle_sha:
        raise ValueError(f"{manifest.get('revision')}: hash bundle canonical SHA mismatch")


def validate_blind_partition(
    *,
    parent_manifest_path: str | Path,
    parent_ground_truth_path: str | Path,
    r1_manifest_path: str | Path,
    r1_ground_truth_path: str | Path,
    r1_proof_path: str | Path,
    r2_ground_truth_path: str | Path,
    r2_manifest_path: str | Path,
    r2_proof_path: str | Path,
    r2_hashes_path: str | Path,
    holdout_ground_truth_path: str | Path,
    holdout_manifest_path: str | Path,
    holdout_proof_path: str | Path,
    holdout_hashes_path: str | Path,
) -> dict[str, Any]:
    import tempfile

    with tempfile.TemporaryDirectory() as temp_dir:
        temp = Path(temp_dir)
        temp_r2_gt = temp / Path(r2_ground_truth_path).name
        temp_r2_manifest = temp / Path(r2_manifest_path).name
        temp_r2_proof = temp / Path(r2_proof_path).name
        temp_r2_hashes = temp / Path(r2_hashes_path).name
        temp_holdout_gt = temp / Path(holdout_ground_truth_path).name
        temp_holdout_manifest = temp / Path(holdout_manifest_path).name
        temp_holdout_proof = temp / Path(holdout_proof_path).name
        temp_holdout_hashes = temp / Path(holdout_hashes_path).name
        regenerated = build_blind_partition(
            parent_manifest_path=parent_manifest_path,
            parent_ground_truth_path=parent_ground_truth_path,
            r1_manifest_path=r1_manifest_path,
            r1_ground_truth_path=r1_ground_truth_path,
            r1_proof_path=r1_proof_path,
            r2_ground_truth_path=temp_r2_gt,
            r2_manifest_path=temp_r2_manifest,
            r2_proof_path=temp_r2_proof,
            r2_hashes_path=temp_r2_hashes,
            holdout_ground_truth_path=temp_holdout_gt,
            holdout_manifest_path=temp_holdout_manifest,
            holdout_proof_path=temp_holdout_proof,
            holdout_hashes_path=temp_holdout_hashes,
        )
        if Path(r2_ground_truth_path).read_bytes() != temp_r2_gt.read_bytes():
            raise ValueError("r2 JSONL deterministic regeneration mismatch")
        if Path(holdout_ground_truth_path).read_bytes() != temp_holdout_gt.read_bytes():
            raise ValueError("holdout JSONL deterministic regeneration mismatch")
        _validate_artifact_files(
            result=regenerated["r2"],
            ground_truth_path=r2_ground_truth_path,
            manifest_path=r2_manifest_path,
            proof_path=r2_proof_path,
            hashes_path=r2_hashes_path,
        )
        _validate_artifact_files(
            result=regenerated["holdout"],
            ground_truth_path=holdout_ground_truth_path,
            manifest_path=holdout_manifest_path,
            proof_path=holdout_proof_path,
            hashes_path=holdout_hashes_path,
        )

    r2_proof = json.loads(Path(r2_proof_path).read_text(encoding="utf-8"))
    holdout_proof = json.loads(Path(holdout_proof_path).read_text(encoding="utf-8"))
    r2_indexes = {entry["parent_line_number"] - 1 for entry in r2_proof["entries"]}
    holdout_indexes = {
        entry["parent_line_number"] - 1 for entry in holdout_proof["entries"]
    }
    parent_manifest, parent_lines, parent_rows = _load_locked_parent(
        parent_manifest_path,
        parent_ground_truth_path,
        expected_parent_revision=PARENT_REVISION,
        expected_parent_corpus_sha256=PARENT_CORPUS_SHA256,
        expected_parent_frames=PARENT_FRAMES,
    )
    r1_indexes = _load_r1_parent_indexes(
        r1_proof_path=r1_proof_path,
        parent_lines=parent_lines,
        parent_rows=parent_rows,
        parent_corpus_sha256=parent_manifest["corpus_sha256"],
        parent_ground_truth_sha256=sha256_file(parent_ground_truth_path),
    )
    intersections = {
        "r1_r2": len(r1_indexes & r2_indexes),
        "r1_holdout": len(r1_indexes & holdout_indexes),
        "r2_holdout": len(r2_indexes & holdout_indexes),
    }
    union = r1_indexes | r2_indexes | holdout_indexes
    if any(intersections.values()) or union != set(range(PARENT_FRAMES)):
        raise ValueError("blind partition disjointness/union proof failed")
    return {
        "valid": True,
        "parent_revision": PARENT_REVISION,
        "parent_corpus_sha256": PARENT_CORPUS_SHA256,
        "r1_frames": len(r1_indexes),
        "r2_frames": len(r2_indexes),
        "holdout_frames": len(holdout_indexes),
        "intersections": intersections,
        "union_frames": len(union),
        "union_equals_original_5000": True,
        "r2_deterministic_regeneration": True,
        "holdout_deterministic_regeneration": True,
        "selection_candidate_result_inputs": [],
        "selection_ground_truth_labels_used": False,
        "tracking_supported": False,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Freeze/validate disjoint NightOwls r2 + holdout partition")
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("build", "validate"):
        item = sub.add_parser(command)
        item.add_argument("--parent-manifest", required=True)
        item.add_argument("--parent-ground-truth", required=True)
        item.add_argument("--r1-manifest", required=True)
        item.add_argument("--r1-ground-truth", required=True)
        item.add_argument("--r1-proof", required=True)
        item.add_argument("--r2-ground-truth", required=True)
        item.add_argument("--r2-manifest", required=True)
        item.add_argument("--r2-proof", required=True)
        item.add_argument("--r2-hashes", required=True)
        item.add_argument("--holdout-ground-truth", required=True)
        item.add_argument("--holdout-manifest", required=True)
        item.add_argument("--holdout-proof", required=True)
        item.add_argument("--holdout-hashes", required=True)
        if command == "validate":
            item.add_argument("--output")
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    kwargs = {
        "parent_manifest_path": args.parent_manifest,
        "parent_ground_truth_path": args.parent_ground_truth,
        "r1_manifest_path": args.r1_manifest,
        "r1_ground_truth_path": args.r1_ground_truth,
        "r1_proof_path": args.r1_proof,
        "r2_ground_truth_path": args.r2_ground_truth,
        "r2_manifest_path": args.r2_manifest,
        "r2_proof_path": args.r2_proof,
        "r2_hashes_path": args.r2_hashes,
        "holdout_ground_truth_path": args.holdout_ground_truth,
        "holdout_manifest_path": args.holdout_manifest,
        "holdout_proof_path": args.holdout_proof,
        "holdout_hashes_path": args.holdout_hashes,
    }
    try:
        if args.command == "build":
            result = build_blind_partition(**kwargs)
            print(
                f"r2_corpus_sha256={result['r2']['manifest']['corpus_sha256']} "
                f"holdout_corpus_sha256={result['holdout']['manifest']['corpus_sha256']}"
            )
            return 0
        report = validate_blind_partition(**kwargs)
        if args.output:
            _json_dump(args.output, report)
        print(json.dumps(report, indent=2))
        return 0
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
