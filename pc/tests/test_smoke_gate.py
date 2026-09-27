from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from spectratrack import smoke_gate


def _write_parent(tmp_path: Path, *, frames: int = 5000, sequences: int = 20):
    gt = tmp_path / "nightowls-val-slice.jsonl"
    counters = {index: 0 for index in range(sequences)}
    rows = []
    for index in range(frames):
        sequence = index % sequences
        frame = counters[sequence]
        counters[sequence] += 1
        objects = []
        if index % 5 == 0:
            objects.append(
                {
                    "id": None,
                    "label": "person",
                    "bbox": [10.0, 20.0, 30.0, 80.0],
                    "ignore": False,
                    "attributes": ["nightowls_occluded_false"],
                }
            )
        if index % 11 == 0:
            objects.append(
                {
                    "id": None,
                    "label": "person",
                    "bbox": [40.0, 20.0, 80.0, 90.0],
                    "ignore": True,
                    "attributes": ["nightowls_official_ignore_region"],
                }
            )
        rows.append(
            {
                "allow_out_of_bounds": True,
                "frame": frame,
                "objects": objects,
                "source": f"extracted/nightowls_validation/image_{index:05d}.png",
                "source_frame": 7_000_000 + index,
                "source_metadata": {
                    "dataset": "NightOwls",
                    "image_id": 7_000_000 + index,
                    "daytime": "night",
                },
                "source_sequence": f"NightOwls:recording-{sequence}",
                "tags": ["night_dark"],
                "video": f"golden/public/nightowls-val-slice/recording-{sequence}",
            }
        )
    gt.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    manifest = {
        "schema": "spectratrack-cctv-corpus-v1",
        "revision": "test-parent-r1",
        "human_confirmation": {
            "confirmed": True,
            "reviewer": "pytest",
            "kind": "official_public_dataset_ground_truth",
        },
        "coverage_profile": "public-dataset",
        "dataset_imports": [{"tracking_supported": False}],
        "allowed_labels": ["person"],
        "splits": {
            "train": None,
            "golden": {
                "ground_truth": gt.name,
                "ground_truth_sha256": smoke_gate.sha256_file(gt),
                "videos": [],
            },
        },
        "golden_coverage": {},
    }
    manifest["corpus_sha256"] = smoke_gate._canonical_sha256(manifest)
    manifest_path = tmp_path / "parent.manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return gt, manifest_path, manifest, rows


def _paths(root: Path):
    return {
        "output_ground_truth_path": root / "nightowls-public-smoke400-r1.jsonl",
        "output_manifest_path": root / "nightowls-public-smoke400-r1.manifest.json",
        "output_proof_path": root / "nightowls-public-smoke400-r1.selection-proof.json",
        "output_hashes_path": root / "nightowls-public-smoke400-r1.hashes.json",
    }


def _build(tmp_path: Path):
    parent_gt, parent_manifest_path, parent_manifest, rows = _write_parent(tmp_path)
    outputs = _paths(tmp_path / "smoke")
    result = smoke_gate.build_smoke_slice(
        parent_manifest_path=parent_manifest_path,
        parent_ground_truth_path=parent_gt,
        expected_parent_revision=parent_manifest["revision"],
        expected_parent_corpus_sha256=parent_manifest["corpus_sha256"],
        expected_parent_frames=5000,
        **outputs,
    )
    return parent_gt, parent_manifest_path, parent_manifest, rows, outputs, result


def test_selection_is_independent_of_labels_and_candidate_outcomes(tmp_path: Path):
    _gt, _manifest_path, _manifest, rows = _write_parent(tmp_path)
    original = smoke_gate.select_parent_indexes(rows)
    mutated = copy.deepcopy(rows)
    for index, row in enumerate(mutated):
        row["objects"] = [] if index % 2 else [{"label": "person", "ignore": False}]
        row["tags"] = ["candidate-dependent-noise", str(index)]
        row["source_metadata"] = {"rewritten": True}
        row["detections"] = [{"score": index / 5000.0}]
        row["candidate_outcome"] = "pass" if index % 3 else "fail"
    assert smoke_gate.select_parent_indexes(mutated) == original


def test_build_smoke400_is_exact_unique_and_sequence_covering(tmp_path: Path):
    _parent_gt, _parent_manifest_path, _parent_manifest, _rows, _outputs, result = _build(tmp_path)
    manifest = result["manifest"]
    proof = result["proof"]
    assert manifest["revision"] == smoke_gate.SMOKE_REVISION
    assert manifest["selection"]["frame_count"] == 400
    assert manifest["selection"]["parent_sequences"] == 20
    assert manifest["selection"]["smoke_sequences"] == 20
    assert manifest["selection"]["sequence_coverage_frames"] == 20
    assert manifest["selection"]["hash_fill_frames"] == 380
    assert manifest["tracking_supported"] is False
    assert manifest["compute_savings"]["frame_fraction"] == pytest.approx(0.08)
    assert manifest["compute_savings"]["approx_variable_frame_compute_reduction"] == pytest.approx(0.92)
    assert manifest["compute_savings"]["frame_count_speedup_factor"] == pytest.approx(12.5)
    assert len(proof["entries"]) == 400
    assert len({entry["parent_line_number"] for entry in proof["entries"]}) == 400
    assert all(1 <= entry["parent_line_number"] <= 5000 for entry in proof["entries"])
    composition = manifest["composition"]
    assert composition["positive_frames"] + composition["true_negative_frames"] == 400
    assert composition["scored_person_objects"] >= composition["positive_frames"]


def test_build_is_byte_deterministic(tmp_path: Path):
    parent_gt, parent_manifest_path, parent_manifest, _rows = _write_parent(tmp_path)
    out_a = _paths(tmp_path / "a")
    out_b = _paths(tmp_path / "b")
    kwargs = {
        "parent_manifest_path": parent_manifest_path,
        "parent_ground_truth_path": parent_gt,
        "expected_parent_revision": parent_manifest["revision"],
        "expected_parent_corpus_sha256": parent_manifest["corpus_sha256"],
        "expected_parent_frames": 5000,
    }
    smoke_gate.build_smoke_slice(**kwargs, **out_a)
    smoke_gate.build_smoke_slice(**kwargs, **out_b)
    for key in out_a:
        assert Path(out_a[key]).read_bytes() == Path(out_b[key]).read_bytes()


def test_validate_proves_regeneration_no_foreign_frames_and_tracking_false(tmp_path: Path):
    parent_gt, parent_manifest_path, parent_manifest, _rows, outputs, _result = _build(tmp_path)
    report = smoke_gate.validate_smoke_slice(
        parent_manifest_path=parent_manifest_path,
        parent_ground_truth_path=parent_gt,
        smoke_ground_truth_path=outputs["output_ground_truth_path"],
        smoke_manifest_path=outputs["output_manifest_path"],
        selection_proof_path=outputs["output_proof_path"],
        hashes_path=outputs["output_hashes_path"],
        expected_parent_revision=parent_manifest["revision"],
        expected_parent_corpus_sha256=parent_manifest["corpus_sha256"],
        expected_parent_frames=5000,
    )
    assert report["valid"] is True
    assert report["frame_count"] == 400
    assert report["unique_frame_count"] == 400
    assert report["foreign_frames"] == 0
    assert report["deterministic_regeneration"] is True
    assert report["candidate_result_inputs"] == []
    assert report["ground_truth_labels_used_for_selection"] is False
    assert report["tracking_supported"] is False


def test_validate_rejects_foreign_or_mutated_smoke_frame(tmp_path: Path):
    parent_gt, parent_manifest_path, parent_manifest, _rows, outputs, _result = _build(tmp_path)
    smoke_path = Path(outputs["output_ground_truth_path"])
    lines = smoke_path.read_text(encoding="utf-8").splitlines()
    row = json.loads(lines[0])
    row["source_frame"] = 99_999_999
    lines[0] = json.dumps(row, sort_keys=True)
    smoke_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="deterministic regeneration"):
        smoke_gate.validate_smoke_slice(
            parent_manifest_path=parent_manifest_path,
            parent_ground_truth_path=parent_gt,
            smoke_ground_truth_path=smoke_path,
            smoke_manifest_path=outputs["output_manifest_path"],
            selection_proof_path=outputs["output_proof_path"],
            hashes_path=outputs["output_hashes_path"],
            expected_parent_revision=parent_manifest["revision"],
            expected_parent_corpus_sha256=parent_manifest["corpus_sha256"],
            expected_parent_frames=5000,
        )


def test_validate_requires_production_smoke400_count(tmp_path: Path):
    parent_gt, parent_manifest_path, parent_manifest, _rows = _write_parent(tmp_path)
    outputs = _paths(tmp_path / "smoke399")
    smoke_gate.build_smoke_slice(
        parent_manifest_path=parent_manifest_path,
        parent_ground_truth_path=parent_gt,
        expected_parent_revision=parent_manifest["revision"],
        expected_parent_corpus_sha256=parent_manifest["corpus_sha256"],
        expected_parent_frames=5000,
        count=399,
        **outputs,
    )
    with pytest.raises(ValueError, match="exactly 400"):
        smoke_gate.validate_smoke_slice(
            parent_manifest_path=parent_manifest_path,
            parent_ground_truth_path=parent_gt,
            smoke_ground_truth_path=outputs["output_ground_truth_path"],
            smoke_manifest_path=outputs["output_manifest_path"],
            selection_proof_path=outputs["output_proof_path"],
            hashes_path=outputs["output_hashes_path"],
            expected_parent_revision=parent_manifest["revision"],
            expected_parent_corpus_sha256=parent_manifest["corpus_sha256"],
            expected_parent_frames=5000,
        )


def test_parent_contract_is_fail_closed(tmp_path: Path):
    parent_gt, parent_manifest_path, parent_manifest, _rows = _write_parent(tmp_path)
    with pytest.raises(ValueError, match="parent revision mismatch"):
        smoke_gate.build_smoke_slice(
            parent_manifest_path=parent_manifest_path,
            parent_ground_truth_path=parent_gt,
            expected_parent_revision="wrong-parent",
            expected_parent_corpus_sha256=parent_manifest["corpus_sha256"],
            expected_parent_frames=5000,
            **_paths(tmp_path / "smoke"),
        )


def test_promotion_policy_never_turns_smoke_pass_into_final_claim():
    policy = smoke_gate.PROMOTION_POLICY
    assert policy["stage1"]["pass_meaning"] == "survives triage; not a final quality claim"
    assert policy["stage2"]["final_claims_require_full5000"] is True
    assert "every smoke survivor" in policy["stage2"]["required_for"]
    assert policy["stage1"]["obvious_reject"]["false_negative_rule"]["require_both"] is True
    assert policy["stage1"]["obvious_reject"]["false_positive_rule"]["require_both"] is True
