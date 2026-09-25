import json

import pytest

from spectratrack.detection_benchmark import (
    load_manifest,
    match_person_detections,
    person_height_bucket,
)
from spectratrack.types import Detection


def test_match_person_detections_counts_duplicate_prediction_as_false_positive():
    truth = [(10.0, 10.0, 50.0, 110.0)]
    detections = [
        Detection((10.0, 10.0, 50.0, 110.0), 0.9, 0, "person"),
        Detection((11.0, 12.0, 49.0, 108.0), 0.7, 0, "person"),
        Detection((10.0, 10.0, 50.0, 110.0), 0.99, 2, "car"),
    ]

    tp, fp, fn, matched = match_person_detections(detections, truth, 0.5)

    assert (tp, fp, fn) == (1, 1, 0)
    assert matched == {0}


@pytest.mark.parametrize(
    ("box", "bucket"),
    [
        ((0.0, 0.0, 10.0, 31.9), "<32px"),
        ((0.0, 0.0, 10.0, 32.0), "32-63px"),
        ((0.0, 0.0, 10.0, 63.9), "32-63px"),
        ((0.0, 0.0, 10.0, 64.0), "64px+"),
    ],
)
def test_person_height_bucket(box, bucket):
    assert person_height_bucket(box) == bucket


def test_load_manifest_validates_person_boxes(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({
        "samples": [{
            "image": "frame.jpg",
            "persons": [[10, 20, 30, 80]],
            "tags": ["night", "compressed"],
        }]
    }), encoding="utf-8")

    loaded = load_manifest(manifest)
    assert loaded["samples"][0]["persons"][0] == [10, 20, 30, 80]


def test_load_manifest_rejects_empty_samples(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"samples": []}), encoding="utf-8")

    with pytest.raises(ValueError):
        load_manifest(manifest)
