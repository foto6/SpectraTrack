import json
from pathlib import Path

import pytest

from spectratrack.qa_benchmark import (
    SCHEMA_VERSION,
    GroundTruthFrame,
    GroundTruthObject,
    PredictedObject,
    compare_results,
    evaluate_frames,
    load_ground_truth,
)


def gt(frame, bbox=(0.0, 0.0, 10.0, 20.0), object_id="p1", tags=("dark",)):
    return GroundTruthFrame(
        video="clip.mp4",
        frame=frame,
        tags=tags,
        objects=(GroundTruthObject(object_id, "person", bbox),),
    )


def pred(bbox=(0.0, 0.0, 10.0, 20.0), track_id=None):
    return PredictedObject(bbox, "person", 0.9, track_id)


def test_recall_precision_and_size_breakdown():
    frames = [gt(0), gt(1, bbox=(20.0, 0.0, 30.0, 60.0), object_id="p2")]
    predictions = {
        ("clip.mp4", 0): [pred(), pred((100.0, 100.0, 120.0, 140.0))],
        ("clip.mp4", 1): [],
    }
    metrics = evaluate_frames(frames, predictions)
    assert metrics["tp"] == 1
    assert metrics["false_negatives"] == 1
    assert metrics["false_positives"] == 1
    assert metrics["recall"] == pytest.approx(0.5)
    assert metrics["precision"] == pytest.approx(0.5)
    assert metrics["by_size"]["height_lt_24"]["recall"] == 1.0
    assert metrics["by_size"]["height_48_95"]["recall"] == 0.0
    assert "precision" not in metrics["by_size"]["height_lt_24"]


def test_object_attribute_reports_recall_without_fake_precision():
    frame = GroundTruthFrame(
        video="clip.mp4",
        frame=0,
        tags=("dark",),
        objects=(GroundTruthObject("p1", "person", (0.0, 0.0, 10.0, 20.0), attributes=("occluded",)),),
    )
    metrics = evaluate_frames([frame], {("clip.mp4", 0): []})
    assert metrics["by_attribute"]["occluded"] == {"tp": 0, "fn": 1, "gt": 1, "recall": 0.0}
    assert "precision" not in metrics["by_attribute"]["occluded"]


def test_ignored_truth_suppresses_overlapping_false_positive():
    frame = GroundTruthFrame(
        video="clip.mp4",
        frame=0,
        tags=(),
        objects=(GroundTruthObject(None, "person", (0.0, 0.0, 10.0, 20.0), ignore=True),),
    )
    metrics = evaluate_frames([frame], {("clip.mp4", 0): [pred()]})
    assert metrics["gt"] == 0
    assert metrics["false_positives"] == 0


def test_id_switch_and_fragmentation_are_counted_on_annotated_frames():
    frames = [gt(0), gt(1), gt(2), gt(3)]
    detections = {("clip.mp4", i): [pred()] for i in range(4)}
    tracks = {
        ("clip.mp4", 0): [pred(track_id=7)],
        ("clip.mp4", 1): [],
        ("clip.mp4", 2): [pred(track_id=8)],
        ("clip.mp4", 3): [pred(track_id=8)],
    }
    metrics = evaluate_frames(frames, detections, tracks)
    assert metrics["tracking"]["id_switches"] == 1
    assert metrics["tracking"]["fragmentations"] == 1


def test_compare_flags_new_false_negative():
    common = {
        "schema_version": SCHEMA_VERSION,
        "ground_truth_sha256": "same",
        "evaluation": {"label": "person", "match_iou": 0.5},
        "performance": {"fps": 30.0, "peak_vram_mb": None},
    }
    baseline = {
        **common,
        "run_name": "BASELINE",
        "metrics": {
            "recall": 1.0,
            "precision": 1.0,
            "false_negatives": 0,
            "false_positives": 0,
            "matched_ground_truth": ["clip.mp4#0#p1"],
            "tracking": {"id_switches": 0, "fragmentations": 0},
        },
    }
    candidate = {
        **common,
        "run_name": "AGENT 1",
        "metrics": {
            "recall": 0.0,
            "precision": 1.0,
            "false_negatives": 1,
            "false_positives": 0,
            "matched_ground_truth": [],
            "tracking": {"id_switches": 0, "fragmentations": 0},
        },
    }
    comparison = compare_results(baseline, candidate)
    assert not comparison["passed"]
    assert comparison["new_false_negatives"] == ["clip.mp4#0#p1"]
    assert any("new false negatives" in reason for reason in comparison["reasons"])


def test_loader_rejects_duplicate_video_frame(tmp_path: Path):
    path = tmp_path / "gt.jsonl"
    row = {"video": "x.mp4", "frame": 0, "objects": []}
    path.write_text(json.dumps(row) + "\n" + json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate frame"):
        load_ground_truth(path)


def test_missing_prediction_frame_becomes_explicit_false_negative():
    metrics = evaluate_frames([gt(5)], {})
    assert metrics["false_negatives"] == 1
    assert metrics["missing_prediction_frames"] == ["clip.mp4#5"]


def test_compare_rejects_different_ground_truth():
    baseline = {
        "schema_version": SCHEMA_VERSION,
        "run_name": "BASELINE",
        "ground_truth_sha256": "a",
        "evaluation": {"label": "person", "match_iou": 0.5},
        "metrics": {},
    }
    candidate = {
        "schema_version": SCHEMA_VERSION,
        "run_name": "candidate",
        "ground_truth_sha256": "b",
        "evaluation": {"label": "person", "match_iou": 0.5},
        "metrics": {},
    }
    with pytest.raises(ValueError, match="different ground-truth"):
        compare_results(baseline, candidate)

def test_loader_rejects_boolean_frame(tmp_path: Path):
    path = tmp_path / "gt.jsonl"
    row = {"video": "x.mp4", "frame": True, "objects": []}
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="frame must be a non-negative integer"):
        load_ground_truth(path)


def test_loader_rejects_non_boolean_ignore(tmp_path: Path):
    path = tmp_path / "gt.jsonl"
    row = {
        "video": "x.mp4",
        "frame": 0,
        "objects": [
            {"id": "p1", "label": "person", "bbox": [0, 0, 10, 20], "ignore": "false"}
        ],
    }
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="ignore must be a boolean"):
        load_ground_truth(path)


def test_loader_rejects_duplicate_object_id_within_frame(tmp_path: Path):
    path = tmp_path / "gt.jsonl"
    row = {
        "video": "x.mp4",
        "frame": 0,
        "objects": [
            {"id": "p1", "label": "person", "bbox": [0, 0, 10, 20]},
            {"id": "p1", "label": "person", "bbox": [20, 0, 30, 20]},
        ],
    }
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate object id"):
        load_ground_truth(path)


def test_compare_rejects_different_schema_versions():
    common = {
        "ground_truth_sha256": "same",
        "evaluation": {"label": "person", "match_iou": 0.5},
        "metrics": {},
    }
    baseline = {**common, "schema_version": SCHEMA_VERSION}
    candidate = {**common, "schema_version": SCHEMA_VERSION - 1}
    with pytest.raises(ValueError, match="different result schema versions"):
        compare_results(baseline, candidate)

