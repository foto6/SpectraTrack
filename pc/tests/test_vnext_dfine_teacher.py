import json

import pytest

from spectratrack.qa_benchmark import GroundTruthFrame, GroundTruthObject, PredictedObject
from spectratrack.research.vnext_dfine_teacher import (
    RAW_SCHEMA,
    TEACHER_SCHEMA,
    _diagnose_frame,
    load_teacher_file,
    normalize_external_teacher_file,
    normalize_teacher_detections,
)


def test_normalize_teacher_scales_normalized_xyxy():
    detections = normalize_teacher_detections(
        [
            {
                "bbox": [0.1, 0.2, 0.5, 0.8],
                "score": 0.8,
                "class_id": 1,
                "label": "person",
                "coordinate_space": "xyxy_norm",
            }
        ],
        width=200,
        height=100,
        score_floor=0.1,
        person_class_ids={1},
        person_labels={"person"},
        dedupe_iou=0.95,
    )

    assert len(detections) == 1
    assert detections[0].bbox == pytest.approx((20.0, 20.0, 100.0, 80.0))


def test_normalize_teacher_remaps_input_crop_coordinates():
    detections = normalize_teacher_detections(
        [
            {
                "bbox": [10, 20, 50, 80],
                "score": 0.9,
                "label": "person",
                "coordinate_space": "xyxy_input",
                "input_size": [100, 100],
                "source_region": [200, 300, 400, 500],
            }
        ],
        width=1000,
        height=800,
        score_floor=0.1,
        person_class_ids=set(),
        person_labels={"person"},
        dedupe_iou=0.95,
    )

    assert detections[0].bbox == pytest.approx((220.0, 340.0, 300.0, 460.0))


def test_normalize_teacher_score_floor_and_duplicate_boxes_are_deterministic():
    detections = normalize_teacher_detections(
        [
            {"bbox": [10, 10, 50, 80], "score": 0.91, "label": "person"},
            {"bbox": [10.2, 10.2, 50.2, 80.2], "score": 0.73, "label": "person"},
            {"bbox": [60, 10, 90, 70], "score": 0.04, "label": "person"},
            {"bbox": [100, 20, 140, 90], "score": 0.99, "label": "car"},
        ],
        width=200,
        height=100,
        score_floor=0.05,
        person_class_ids=set(),
        person_labels={"person"},
        dedupe_iou=0.95,
    )

    assert len(detections) == 1
    assert detections[0].score == pytest.approx(0.91)
    assert detections[0].bbox == pytest.approx((10.0, 10.0, 50.0, 80.0))


@pytest.mark.parametrize("score", [-0.1, 1.1, float("nan")])
def test_normalize_teacher_rejects_invalid_scores(score):
    with pytest.raises(ValueError, match="score"):
        normalize_teacher_detections(
            [{"bbox": [0, 0, 10, 10], "score": score, "label": "person"}],
            width=20,
            height=20,
            score_floor=0.0,
            person_class_ids=set(),
            person_labels={"person"},
            dedupe_iou=0.95,
        )


def _metadata(corpus_revision: str) -> dict:
    return {
        "type": "metadata",
        "schema": RAW_SCHEMA,
        "corpus_revision": corpus_revision,
        "source_revision": "frozen-source-r1",
        "source_sha256": "b" * 64,
        "provenance": {
            "model_family": "D-FINE",
            "model_variant": "D-FINE-X",
            "training_recipe": "Objects365+COCO",
            "weights_sha256": "a" * 64,
            "model_source": "local/checkpoint.pth",
            "model_revision": "dfine-x-o365-coco-r1",
            "code_source": "https://github.com/Peterande/D-FINE",
            "code_revision": "deadbeef",
            "runtime": "fixture",
            "runtime_version": "1",
        },
    }


def test_external_teacher_roundtrip_preserves_provenance_and_marks_nightowls_eval_only(tmp_path):
    raw = tmp_path / "raw.jsonl"
    out = tmp_path / "teacher.jsonl"
    records = [
        _metadata("nightowls-public-r1-smoke400"),
        {
            "type": "frame",
            "video": "nightowls/frame.jpg",
            "frame": 0,
            "width": 100,
            "height": 50,
            "detections": [
                {
                    "bbox": [0.1, 0.1, 0.5, 0.9],
                    "score": 0.8,
                    "class_id": 0,
                    "label": "person",
                    "coordinate_space": "xyxy_norm",
                }
            ],
        },
    ]
    raw.write_text("\n".join(json.dumps(row) for row in records) + "\n", encoding="utf-8")

    summary = normalize_external_teacher_file(
        raw,
        out,
        score_floor=0.1,
        person_class_ids={0},
        person_labels={"person"},
        dedupe_iou=0.95,
    )
    metadata, predictions = load_teacher_file(out)

    assert summary["schema"] == TEACHER_SCHEMA
    assert summary["training_selection_eligible"] is False
    assert metadata["provenance"]["weights_sha256"] == "a" * 64
    assert metadata["source_sha256"] == "b" * 64
    assert predictions[("nightowls/frame.jpg", 0)][0].bbox == pytest.approx(
        (10.0, 5.0, 50.0, 45.0)
    )


def test_teacher_diagnostics_separate_recovered_fn_and_false_positive():
    frame = GroundTruthFrame(
        video="clip",
        frame=3,
        tags=("night",),
        objects=(
            GroundTruthObject("p1", "person", (0, 0, 20, 40)),
            GroundTruthObject("p2", "person", (50, 0, 70, 40)),
        ),
    )
    teacher = [
        PredictedObject((0, 0, 20, 40), "person", 0.8),
        PredictedObject((50, 0, 70, 40), "person", 0.7),
        PredictedObject((80, 0, 95, 30), "person", 0.9),
    ]
    baseline = [PredictedObject((0, 0, 20, 40), "person", 0.8)]

    result = _diagnose_frame(frame, teacher, baseline, 0.5)

    assert len(result["teacher_recovers_baseline_fn"]) == 1
    assert result["teacher_recovers_baseline_fn"][0]["object_id"] == "p2"
    assert len(result["teacher_false_positives"]) == 1
    assert result["teacher_false_positives"][0]["score"] == pytest.approx(0.9)
