from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

from spectratrack import blind_smoke_gate


def _rows(count: int = 5000, sequences: int = 100) -> list[dict]:
    counters = {index: 0 for index in range(sequences)}
    rows = []
    for index in range(count):
        sequence = index % sequences
        frame = counters[sequence]
        counters[sequence] += 1
        rows.append(
            {
                "frame": frame,
                "video": f"golden/public/nightowls-val-slice/recording-{sequence}",
                "source": f"extracted/nightowls_validation/image_{index:05d}.png",
                "source_frame": 7_000_000 + index,
                "source_sequence": f"NightOwls:recording-{sequence}",
                "objects": (
                    [
                        {
                            "label": "person",
                            "bbox": [1.0, 2.0, 3.0, 20.0],
                            "ignore": False,
                        }
                    ]
                    if index % 5 == 0
                    else []
                ),
                "tags": ["night_dark"],
            }
        )
    return rows


def _parent_files(tmp_path: Path, rows: list[dict]):
    gt = tmp_path / "parent.jsonl"
    lines = [json.dumps(row, sort_keys=True) for row in rows]
    gt.write_text("\n".join(lines) + "\n", encoding="utf-8")
    manifest = {
        "revision": blind_smoke_gate.PARENT_REVISION,
        "corpus_sha256": blind_smoke_gate.PARENT_CORPUS_SHA256,
    }
    return manifest, lines, gt


def _r1_proof(tmp_path: Path, rows: list[dict], lines: list[str], indexes: set[int]):
    entries = []
    for number, index in enumerate(sorted(indexes), start=1):
        row = rows[index]
        entries.append(
            {
                "smoke_line_number": number,
                "smoke_frame": number - 1,
                "parent_line_number": index + 1,
                "parent_frame": row["frame"],
                "video": row["video"],
                "source": row["source"],
                "source_frame": row["source_frame"],
                "source_sequence": row["source_sequence"],
                "parent_identity_sha256": "x",
                "smoke_identity_sha256": "y",
                "parent_row_sha256": hashlib.sha256(
                    lines[index].encode("utf-8")
                ).hexdigest(),
                "selected_via": "sequence_coverage",
                "rank_sha256": "z",
            }
        )
    proof = {
        "schema": blind_smoke_gate.PROOF_SCHEMA,
        "revision": blind_smoke_gate.R1_REVISION,
        "parent": {
            "revision": blind_smoke_gate.PARENT_REVISION,
            "corpus_sha256": blind_smoke_gate.PARENT_CORPUS_SHA256,
            "ground_truth_sha256": blind_smoke_gate.sha256_file(tmp_path / "parent.jsonl"),
            "frame_count": 5000,
        },
        "selection": {"frame_count": 400},
        "entries": entries,
    }
    path = tmp_path / "r1.proof.json"
    path.write_text(json.dumps(proof), encoding="utf-8")
    return path


def _outputs(root: Path):
    return {
        "r2_ground_truth_path": root / "r2.jsonl",
        "r2_manifest_path": root / "r2.manifest.json",
        "r2_proof_path": root / "r2.proof.json",
        "r2_hashes_path": root / "r2.hashes.json",
        "holdout_ground_truth_path": root / "holdout.jsonl",
        "holdout_manifest_path": root / "holdout.manifest.json",
        "holdout_proof_path": root / "holdout.proof.json",
        "holdout_hashes_path": root / "holdout.hashes.json",
    }


def test_r2_selection_is_label_and_outcome_independent():
    rows = _rows()
    excluded = set(range(400))
    selected_a, fallback_a = blind_smoke_gate.select_r2_parent_indexes(
        rows,
        excluded_indexes=excluded,
    )
    mutated = copy.deepcopy(rows)
    for index, row in enumerate(mutated):
        row["objects"] = [] if index % 2 else [{"label": "person", "ignore": False}]
        row["tags"] = ["candidate-inspired", str(index)]
        row["source_metadata"] = {"rewritten": True}
        row["candidate_outcome"] = "pass" if index % 3 else "fail"
        row["detections"] = [{"score": index / 5000.0}]
    selected_b, fallback_b = blind_smoke_gate.select_r2_parent_indexes(
        mutated,
        excluded_indexes=excluded,
    )
    assert selected_a == selected_b
    assert fallback_a == fallback_b
    assert set(selected_a).isdisjoint(excluded)
    assert len(selected_a) == 400


def test_r2_second_per_sequence_and_deterministic_fallback():
    rows = _rows(count=500, sequences=100)
    # Make sequence 0 a singleton by removing its later frames while retaining enough others.
    singleton = rows[0]
    rows = [singleton] + [row for row in rows[1:] if not row["video"].endswith("-0")]
    while len(rows) < 500:
        index = len(rows) + 10_000
        sequence = 1 + (index % 99)
        base = copy.deepcopy(rows[1 + (index % (len(rows) - 1))])
        base["source"] = f"extra_{index}.png"
        base["source_frame"] = 8_000_000 + index
        base["frame"] = index
        base["video"] = f"golden/public/nightowls-val-slice/recording-{sequence}"
        base["source_sequence"] = f"NightOwls:recording-{sequence}"
        rows.append(base)
    excluded = {0}
    selected, fallback = blind_smoke_gate.select_r2_parent_indexes(
        rows,
        excluded_indexes=excluded,
    )
    assert rows[0]["video"] in fallback
    assert len(selected) == 400
    assert 0 not in selected
    covered = {
        rows[index]["video"]
        for index, meta in selected.items()
        if meta["selected_via"] == "second_per_sequence"
    }
    assert rows[0]["video"] not in covered


def test_build_and_validate_exact_disjoint_partition(monkeypatch, tmp_path: Path):
    rows = _rows()
    parent_manifest, lines, parent_gt = _parent_files(tmp_path, rows)
    r1_indexes = set(range(400))
    r1_proof = _r1_proof(tmp_path, rows, lines, r1_indexes)
    r1_manifest = tmp_path / "r1.manifest.json"
    r1_gt = tmp_path / "r1.jsonl"
    r1_manifest.write_text("{}", encoding="utf-8")
    r1_gt.write_text("dummy\n", encoding="utf-8")

    monkeypatch.setattr(
        blind_smoke_gate,
        "_load_locked_parent",
        lambda *_args, **_kwargs: (parent_manifest, lines, rows),
    )
    monkeypatch.setattr(
        blind_smoke_gate,
        "_validate_r1_frozen_inputs",
        lambda **_kwargs: {},
    )
    out = _outputs(tmp_path / "artifacts")
    result = blind_smoke_gate.build_blind_partition(
        parent_manifest_path=tmp_path / "parent.manifest.json",
        parent_ground_truth_path=parent_gt,
        r1_manifest_path=r1_manifest,
        r1_ground_truth_path=r1_gt,
        r1_proof_path=r1_proof,
        **out,
    )
    partition = result["partition"]
    assert partition["r1_count"] == 400
    assert partition["r2_count"] == 400
    assert partition["holdout_count"] == 4200
    assert partition["intersections"] == {
        "r1_r2": 0,
        "r1_holdout": 0,
        "r2_holdout": 0,
    }
    assert partition["union_count"] == 5000
    assert partition["union_equals_parent"] is True
    assert result["r2"]["manifest"]["tracking_supported"] is False
    assert result["holdout"]["manifest"]["tracking_supported"] is False
    assert result["r2"]["manifest"]["selection"]["candidate_result_inputs"] == []
    assert (
        result["r2"]["manifest"]["selection"]["ground_truth_labels_used_for_selection"]
        is False
    )
    assert (
        result["holdout"]["manifest"]["selection"]["rule"]
        == "exact set complement of frozen r1 and frozen r2 parent membership; no selection freedom"
    )

    report = blind_smoke_gate.validate_blind_partition(
        parent_manifest_path=tmp_path / "parent.manifest.json",
        parent_ground_truth_path=parent_gt,
        r1_manifest_path=r1_manifest,
        r1_ground_truth_path=r1_gt,
        r1_proof_path=r1_proof,
        **out,
    )
    assert report["valid"] is True
    assert report["union_equals_original_5000"] is True
    assert report["r2_deterministic_regeneration"] is True
    assert report["holdout_deterministic_regeneration"] is True


def test_stage_semantics_separate_discovery_blind_holdout_and_full_characterization():
    semantics = blind_smoke_gate.STAGE_SEMANTICS
    assert "discovery" in semantics["r1"]["role"]
    assert "blind validation" in semantics["r2"]["role"]
    assert "final disjoint" in semantics["holdout"]["role"]
    assert "not statistically independent" in semantics["full5000"]["role"]
