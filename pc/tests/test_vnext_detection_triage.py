import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from spectratrack.qa_benchmark import GroundTruthFrame, GroundTruthObject
from spectratrack.research.vnext_detection_corpus import _apply_frame_selector
from spectratrack.research.vnext_detection_selector import (
    load_frozen_frame_selector,
    select_ground_truth_frames,
)
from spectratrack.research.vnext_detection_triage import (
    load_candidate_matrix,
    score_candidate,
)


def _write_manifest(tmp_path: Path, frames: list[dict], revision: str = "smoke-r1") -> tuple[Path, str]:
    path = tmp_path / "smoke.json"
    path.write_text(
        json.dumps({"revision": revision, "frame_ids": frames}, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


def _gt(video: str, frame: int) -> GroundTruthFrame:
    return GroundTruthFrame(
        video,
        frame,
        ("night",),
        (GroundTruthObject(f"p{frame}", "person", (1.0, 2.0, 9.0, 18.0)),),
    )


def test_selector_binds_exact_hash_count_revision_and_ids(tmp_path):
    path, digest = _write_manifest(
        tmp_path,
        [{"video": "b", "frame": 2}, {"video": "a", "frame": 1}],
    )
    selector = load_frozen_frame_selector(
        path,
        expected_sha256=digest,
        expected_count=2,
        expected_revision="smoke-r1",
    )
    selected = select_ground_truth_frames([_gt("a", 1), _gt("b", 2), _gt("c", 3)], selector)

    assert [(item.video, item.frame) for item in selected] == [("b", 2), ("a", 1)]
    assert selector.manifest_sha256 == digest
    assert len(selector.selection_sha256) == 64


def test_selector_rejects_hash_count_duplicates_and_missing_ids(tmp_path):
    path, digest = _write_manifest(
        tmp_path,
        [{"video": "a", "frame": 1}, {"video": "a", "frame": 1}],
    )
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        load_frozen_frame_selector(path, expected_sha256="0" * 64, expected_count=2)
    with pytest.raises(ValueError, match="count mismatch"):
        load_frozen_frame_selector(path, expected_sha256=digest, expected_count=3)
    with pytest.raises(ValueError, match="duplicate"):
        load_frozen_frame_selector(path, expected_sha256=digest, expected_count=2)

    path, digest = _write_manifest(tmp_path, [{"video": "missing", "frame": 9}])
    selector = load_frozen_frame_selector(path, expected_sha256=digest, expected_count=1)
    with pytest.raises(ValueError, match="absent from ground truth"):
        select_ground_truth_frames([_gt("a", 1)], selector)


def test_corpus_selector_rejects_secondary_selection_controls(tmp_path):
    path, digest = _write_manifest(tmp_path, [{"video": "a", "frame": 1}])
    base = dict(
        frame_manifest=str(path),
        frame_manifest_sha256=digest,
        frame_manifest_count=1,
        frame_manifest_revision="smoke-r1",
        include_video=[],
        frame_limit_per_video=0,
    )
    selected, record = _apply_frame_selector(SimpleNamespace(**base), [_gt("a", 1), _gt("b", 2)])
    assert [(item.video, item.frame) for item in selected] == [("a", 1)]
    assert record["frame_count"] == 1

    with pytest.raises(ValueError, match="include-video"):
        _apply_frame_selector(SimpleNamespace(**{**base, "include_video": ["a"]}), [_gt("a", 1)])
    with pytest.raises(ValueError, match="frame-limit-per-video"):
        _apply_frame_selector(SimpleNamespace(**{**base, "frame_limit_per_video": 1}), [_gt("a", 1)])


def test_committed_smoke_matrix_is_locked_and_valid():
    matrix_path = Path(__file__).parents[1] / "benchmarks" / "vnext" / "detection" / "smoke400_candidate_matrix.json"
    matrix, digest = load_candidate_matrix(matrix_path)

    assert matrix["expected_frame_count"] == 400
    assert matrix["control_id"] == "control-hard-nms-tile640"
    assert [item["id"] for item in matrix["experiments"] if item["enabled"]] == [
        "control-hard-nms-tile640",
        "candidate-hard-nms-tile512",
    ]
    assert len(digest) == 64


def _report(precision: float, recall: float, f1: float, *, bbox: float = 0.75, center: float = 0.05):
    return {
        "methods": {
            "hard-nms": {
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "tp": 100,
                "fp": 20,
                "fn": 30,
                "bbox_localization_iou": bbox,
                "center_error_gt_diag_ratio": center,
            }
        }
    }


def test_smoke_scoring_promotes_rejects_or_marks_ambiguous():
    matrix_path = Path(__file__).parents[1] / "benchmarks" / "vnext" / "detection" / "smoke400_candidate_matrix.json"
    matrix, _ = load_candidate_matrix(matrix_path)
    control, candidate = matrix["experiments"]
    rules = matrix["decision_rules"]
    base = _report(0.80, 0.70, 0.746)

    promoted = score_candidate(
        base, _report(0.80, 0.72, 0.758),
        control=control, candidate=candidate, rules=rules,
    )
    rejected = score_candidate(
        base, _report(0.78, 0.69, 0.731),
        control=control, candidate=candidate, rules=rules,
    )
    ambiguous = score_candidate(
        base, _report(0.80, 0.705, 0.749),
        control=control, candidate=candidate, rules=rules,
    )

    assert promoted["recommendation"] == "PROMOTE_TO_FULL"
    assert rejected["recommendation"] == "REJECT"
    assert ambiguous["recommendation"] == "AMBIGUOUS"
    assert all(result["full5000_not_run"] for result in (promoted, rejected, ambiguous))


def test_selector_accepts_a5_smoke_selection_proof_schema(tmp_path):
    proof = {
        "schema": "spectratrack-smoke-selection-proof-v1",
        "revision": "nightowls-public-smoke400-r1",
        "parent": {
            "revision": "nightowls-public-slice5000-r1",
            "corpus_sha256": "a" * 64,
        },
        "selection": {
            "frame_count": 2,
            "selection_identity_sha256": "b" * 64,
        },
        "entries": [
            {"video": "a", "smoke_frame": 0, "parent_frame": 7},
            {"video": "b", "smoke_frame": 0, "parent_frame": 11},
        ],
    }
    path = tmp_path / "selection-proof.json"
    path.write_text(json.dumps(proof, sort_keys=True) + "\n", encoding="utf-8")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()

    selector = load_frozen_frame_selector(
        path,
        expected_sha256=digest,
        expected_count=2,
        expected_revision="nightowls-public-smoke400-r1",
    )

    assert selector.frame_ids == (("a", 0), ("b", 0))
    assert selector.selection_sha256 == "b" * 64
    assert selector.source_corpus_revision == "nightowls-public-slice5000-r1"
    assert selector.source_corpus_sha256 == "a" * 64
