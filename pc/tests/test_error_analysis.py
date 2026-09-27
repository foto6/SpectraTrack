import json
from pathlib import Path

import pytest

from spectratrack.error_analysis import (
    TAXONOMY,
    build_error_report,
    load_frame_manifest,
)
from spectratrack.qa_benchmark import (
    PredictedObject,
    evaluate_frames,
    ground_truth_sha256,
    load_ground_truth,
)


def _write_json(path: Path, payload) -> Path:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _write_jsonl(path: Path, rows) -> Path:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    return path


def _result_from_predictions(
    gt_path: Path,
    predictions_by_frame,
    *,
    name="CONTROL",
    selection=None,
):
    frames = load_ground_truth(gt_path)
    metrics = evaluate_frames(frames, predictions_by_frame)
    result = {
        "schema_version": 1,
        "run_name": name,
        "revision": name.lower(),
        "ground_truth_sha256": ground_truth_sha256(gt_path),
        "evaluation": {"label": "person", "match_iou": 0.5},
        "settings": {},
        "performance": {},
        "metrics": metrics,
    }
    if selection is not None:
        result["selection"] = selection
    return result


def _pred(bbox, score=0.9):
    return PredictedObject(tuple(float(v) for v in bbox), "person", score)


def test_manifest_hash_is_order_independent_but_file_hash_is_not(tmp_path: Path):
    first = _write_json(
        tmp_path / "first.json",
        {"frames": [{"video": "b.mp4", "frame": 2}, {"video": "a.mp4", "frame": 1}]},
    )
    second = _write_json(
        tmp_path / "second.json",
        {"frames": [{"video": "a.mp4", "frame": 1}, {"video": "b.mp4", "frame": 2}]},
    )
    a = load_frame_manifest(first)
    b = load_frame_manifest(second)
    assert a.frame_ids_sha256 == b.frame_ids_sha256
    assert a.file_sha256 != b.file_sha256


def test_manifest_rejects_duplicate_frames(tmp_path: Path):
    manifest = _write_json(
        tmp_path / "manifest.json",
        {"frames": [{"video": "a.mp4", "frame": 1}, {"video": "a.mp4", "frame": 1}]},
    )
    with pytest.raises(ValueError, match="duplicate frame IDs"):
        load_frame_manifest(manifest)


def test_a5_smoke_manifest_uses_frozen_ground_truth_as_exact_frame_list(tmp_path: Path):
    gt = _write_jsonl(
        tmp_path / "smoke.jsonl",
        [
            {"video": "a.mp4", "frame": 0, "tags": [], "objects": []},
            {"video": "b.mp4", "frame": 3, "tags": [], "objects": []},
        ],
    )
    manifest = _write_json(
        tmp_path / "manifest.json",
        {
            "schema": "spectratrack-smoke-slice-v1",
            "revision": "smoke-r1",
            "corpus_sha256": "c" * 64,
            "artifacts": {
                "ground_truth": "smoke.jsonl",
                "ground_truth_sha256": ground_truth_sha256(gt),
            },
            "selection": {"frame_count": 2},
        },
    )
    selection = load_frame_manifest(manifest, ground_truth_path=gt)
    assert selection.schema == "spectratrack-smoke-slice-v1"
    assert selection.revision == "smoke-r1"
    assert selection.corpus_sha256 == "c" * 64
    assert [item.frame_id for item in selection.frames] == ["a.mp4#0", "b.mp4#3"]

    gt.write_text(gt.read_text(encoding="utf-8") + " ", encoding="utf-8")
    with pytest.raises(ValueError, match="does not match frozen smoke manifest"):
        load_frame_manifest(manifest, ground_truth_path=gt)


def test_taxonomy_uses_geometry_and_metadata_without_guessing(tmp_path: Path):
    gt = _write_jsonl(
        tmp_path / "gt.jsonl",
        [
            {
                "video": "clip.mp4",
                "frame": 0,
                "tags": ["night_dark"],
                "objects": [
                    {
                        "id": "p0",
                        "label": "person",
                        "bbox": [10, 10, 20, 40],
                        "attributes": ["small", "nightowls_occluded_true"],
                    }
                ],
            },
            {
                "video": "clip.mp4",
                "frame": 1,
                "tags": [],
                "objects": [],
            },
            {
                "video": "clip.mp4",
                "frame": 2,
                "tags": [],
                "objects": [
                    {"id": "p2", "label": "person", "bbox": [100, 20, 140, 140]}
                ],
            },
            {
                "video": "clip.mp4",
                "frame": 3,
                "tags": [],
                "objects": [
                    {"id": "p3", "label": "person", "bbox": [0, 10, 150, 40]}
                ],
            },
        ],
    )
    manifest = _write_json(
        tmp_path / "manifest.json",
        {"frames": [{"video": "clip.mp4", "frame": i} for i in range(4)]},
    )
    predictions = {
        ("clip.mp4", 0): [_pred([16, 10, 26, 40])],
        ("clip.mp4", 1): [_pred([300, 200, 340, 320], 0.8)],
        ("clip.mp4", 2): [
            _pred([100, 20, 140, 140], 0.95),
            _pred([102, 22, 142, 142], 0.7),
        ],
        ("clip.mp4", 3): [],
    }
    result = _write_json(
        tmp_path / "control.json",
        _result_from_predictions(gt, predictions),
    )
    observations = _write_jsonl(
        tmp_path / "control.observations.jsonl",
        [
            {
                "video": "clip.mp4",
                "frame": 0,
                "width": 400,
                "height": 300,
                "predictions": [{"bbox": [16, 10, 26, 40], "label": "person", "score": 0.9}],
            },
            {
                "video": "clip.mp4",
                "frame": 1,
                "width": 400,
                "height": 300,
                "predictions": [{"bbox": [300, 200, 340, 320], "label": "person", "score": 0.8}],
            },
            {
                "video": "clip.mp4",
                "frame": 2,
                "width": 400,
                "height": 300,
                "predictions": [
                    {"bbox": [100, 20, 140, 140], "label": "person", "score": 0.95},
                    {"bbox": [102, 22, 142, 142], "label": "person", "score": 0.7},
                ],
            },
            {
                "video": "clip.mp4",
                "frame": 3,
                "width": 400,
                "height": 300,
                "predictions": [],
            },
        ],
    )

    report = build_error_report(
        ground_truth_path=gt,
        manifest_path=manifest,
        control_result_path=result,
        control_observations_path=observations,
    )
    taxonomy = report["control"]["taxonomy"]
    assert tuple(taxonomy) == TAXONOMY
    for category in (
        "missed_small_person",
        "low_contrast_night_darkness",
        "occlusion_crowd",
        "localization_iou_miss",
        "duplicate_fragmented_detection",
        "background_false_positive",
        "edge_of_frame",
        "scale_extreme_aspect",
    ):
        assert taxonomy[category]["count"] >= 1
    assert taxonomy["unknown"]["count"] == 0


def test_summary_only_false_positive_stays_unknown(tmp_path: Path):
    gt = _write_jsonl(
        tmp_path / "gt.jsonl",
        [{"video": "clip.mp4", "frame": 0, "tags": [], "objects": []}],
    )
    manifest = _write_json(
        tmp_path / "manifest.json",
        {"frames": [{"video": "clip.mp4", "frame": 0}]},
    )
    result_payload = _result_from_predictions(
        gt,
        {("clip.mp4", 0): [_pred([100, 100, 120, 150]), _pred([200, 100, 220, 150])]},
    )
    result = _write_json(tmp_path / "control.json", result_payload)
    report = build_error_report(
        ground_truth_path=gt,
        manifest_path=manifest,
        control_result_path=result,
    )
    assert report["control"]["false_positive_count"] == 2
    assert report["control"]["taxonomy"]["unknown"]["false_positive_count"] == 2
    assert report["control"]["taxonomy"]["background_false_positive"]["count"] == 0


def test_subset_manifest_requires_exact_selection_proof(tmp_path: Path):
    gt = _write_jsonl(
        tmp_path / "gt.jsonl",
        [
            {"video": "clip.mp4", "frame": 0, "tags": [], "objects": []},
            {"video": "clip.mp4", "frame": 1, "tags": [], "objects": []},
        ],
    )
    manifest = _write_json(
        tmp_path / "manifest.json",
        {"frames": [{"video": "clip.mp4", "frame": 0}]},
    )
    result = _write_json(
        tmp_path / "control.json",
        _result_from_predictions(gt, {("clip.mp4", 0): [], ("clip.mp4", 1): []}),
    )
    with pytest.raises(ValueError, match="cannot prove qa_benchmark result"):
        build_error_report(
            ground_truth_path=gt,
            manifest_path=manifest,
            control_result_path=result,
        )


def test_subset_manifest_accepts_matching_result_selection_hash(tmp_path: Path):
    gt = _write_jsonl(
        tmp_path / "gt.jsonl",
        [
            {
                "video": "clip.mp4",
                "frame": 0,
                "tags": ["dark"],
                "objects": [{"id": "p0", "label": "person", "bbox": [10, 10, 20, 40]}],
            },
            {
                "video": "clip.mp4",
                "frame": 1,
                "tags": [],
                "objects": [{"id": "p1", "label": "person", "bbox": [20, 20, 40, 80]}],
            },
        ],
    )
    manifest = _write_json(
        tmp_path / "manifest.json",
        {"frames": [{"video": "clip.mp4", "frame": 0}]},
    )
    selection = load_frame_manifest(manifest)
    selected_frame = load_ground_truth(gt)[0]
    metrics = evaluate_frames([selected_frame], {("clip.mp4", 0): []})
    result_payload = {
        "schema_version": 1,
        "run_name": "CONTROL",
        "revision": "control",
        "ground_truth_sha256": ground_truth_sha256(gt),
        "evaluation": {"label": "person", "match_iou": 0.5},
        "selection": {"selected_frame_ids_sha256": selection.frame_ids_sha256},
        "settings": {},
        "performance": {},
        "metrics": metrics,
    }
    result = _write_json(tmp_path / "control.json", result_payload)
    report = build_error_report(
        ground_truth_path=gt,
        manifest_path=manifest,
        control_result_path=result,
    )
    assert report["control"]["selection_proof"] == "result_selection_hash"
    assert report["control"]["false_negative_count"] == 1


def test_candidate_delta_generates_bounded_actionable_guidance(tmp_path: Path):
    gt = _write_jsonl(
        tmp_path / "gt.jsonl",
        [
            {
                "video": "night.mp4",
                "frame": 0,
                "tags": ["dark"],
                "objects": [
                    {"id": "p1", "label": "person", "bbox": [100, 20, 118, 62], "attributes": ["small"]}
                ],
            },
            {"video": "night.mp4", "frame": 1, "tags": ["dark"], "objects": []},
        ],
    )
    manifest = _write_json(
        tmp_path / "manifest.json",
        {"frames": [{"video": "night.mp4", "frame": 0}, {"video": "night.mp4", "frame": 1}]},
    )
    control_predictions = {
        ("night.mp4", 0): [],
        ("night.mp4", 1): [_pred([250, 100, 280, 180], 0.8)],
    }
    candidate_predictions = {("night.mp4", 0): [], ("night.mp4", 1): []}
    control_result = _write_json(
        tmp_path / "control.json",
        _result_from_predictions(gt, control_predictions, name="CONTROL"),
    )
    candidate_result = _write_json(
        tmp_path / "candidate.json",
        _result_from_predictions(gt, candidate_predictions, name="CANDIDATE"),
    )
    control_obs = _write_jsonl(
        tmp_path / "control.obs.jsonl",
        [
            {"video": "night.mp4", "frame": 0, "width": 400, "height": 300, "predictions": []},
            {
                "video": "night.mp4",
                "frame": 1,
                "width": 400,
                "height": 300,
                "predictions": [{"bbox": [250, 100, 280, 180], "label": "person", "score": 0.8}],
            },
        ],
    )
    candidate_obs = _write_jsonl(
        tmp_path / "candidate.obs.jsonl",
        [
            {"video": "night.mp4", "frame": 0, "width": 400, "height": 300, "predictions": []},
            {"video": "night.mp4", "frame": 1, "width": 400, "height": 300, "predictions": []},
        ],
    )
    first = build_error_report(
        ground_truth_path=gt,
        manifest_path=manifest,
        control_result_path=control_result,
        candidate_result_path=candidate_result,
        control_observations_path=control_obs,
        candidate_observations_path=candidate_obs,
        top_k=3,
    )
    second = build_error_report(
        ground_truth_path=gt,
        manifest_path=manifest,
        control_result_path=control_result,
        candidate_result_path=candidate_result,
        control_observations_path=control_obs,
        candidate_observations_path=candidate_obs,
        top_k=3,
    )
    assert first == second
    assert first["report_sha256"] == second["report_sha256"]
    assert first["comparison"]["category_count_delta"]["background_false_positive"] == -1
    guidance = " ".join(first["comparison"]["guidance"])
    assert "reduces background false-positive events by 1" in guidance
    assert "recovers 0 low-contrast/night" in guidance
    assert first["limits"]["top_k_representatives_per_category"] == 3


def test_existing_example_fixture_is_supported(tmp_path: Path):
    root = Path(__file__).resolve().parents[1]
    gt = root / "benchmarks" / "ground_truth.example.jsonl"
    frames = load_ground_truth(gt)
    manifest = _write_json(
        tmp_path / "manifest.json",
        {"frames": [{"video": frame.video, "frame": frame.frame} for frame in frames]},
    )
    predictions = {(frame.video, frame.frame): [] for frame in frames}
    result = _write_json(
        tmp_path / "control.json",
        _result_from_predictions(gt, predictions),
    )
    report = build_error_report(
        ground_truth_path=gt,
        manifest_path=manifest,
        control_result_path=result,
    )
    assert report["frame_count"] == 3
    assert report["control"]["taxonomy"]["low_contrast_night_darkness"]["count"] == 2
    assert report["control"]["taxonomy"]["missed_small_person"]["count"] == 2


def test_top_k_is_hard_bounded(tmp_path: Path):
    gt = _write_jsonl(
        tmp_path / "gt.jsonl",
        [{"video": "clip.mp4", "frame": 0, "tags": [], "objects": []}],
    )
    manifest = _write_json(
        tmp_path / "manifest.json",
        {"frames": [{"video": "clip.mp4", "frame": 0}]},
    )
    result = _write_json(
        tmp_path / "control.json",
        _result_from_predictions(gt, {("clip.mp4", 0): []}),
    )
    with pytest.raises(ValueError, match="top_k"):
        build_error_report(
            ground_truth_path=gt,
            manifest_path=manifest,
            control_result_path=result,
            top_k=26,
        )
