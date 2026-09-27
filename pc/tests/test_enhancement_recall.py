import numpy as np
import pytest

from spectratrack.enhancement_recall import (
    corroborate_enhanced_detections,
    detect_people_with_adaptive_regions,
    iter_adaptive_regions,
    translate_detection,
)
from spectratrack.types import Detection


def test_iter_adaptive_regions_keeps_raw_evidence_and_marks_processing():
    frame = np.full((240, 320, 3), 12, dtype=np.uint8)
    tiles = list(iter_adaptive_regions(frame, [(0, 0, 320, 240)]))
    assert len(tiles) == 1

    region, raw_tile, enhanced_tile, quality, operations = tiles[0]
    assert region == (0, 0, 320, 240)
    assert raw_tile.shape == frame.shape
    assert enhanced_tile is not None
    assert enhanced_tile.shape == frame.shape
    assert quality["darkness"] > 0.8
    assert "low_light" in operations


@pytest.mark.parametrize(
    "region",
    [
        (-1, 0, 100, 100),
        (0, 0, 321, 100),
        (10, 10, 10, 20),
        (10, 10, 20, 10),
    ],
)
def test_iter_adaptive_regions_rejects_invalid_regions(region):
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    with pytest.raises(ValueError):
        list(iter_adaptive_regions(frame, [region]))


def test_translate_detection_preserves_metadata():
    detection = Detection((10, 20, 30, 80), 0.4, 0, "person", (0.1, 0.2))
    translated = translate_detection(detection, 100, 200)
    assert translated.bbox == (110, 220, 130, 280)
    assert translated.score == detection.score
    assert translated.class_id == detection.class_id
    assert translated.label == detection.label
    assert translated.appearance == detection.appearance


def test_enhanced_candidate_requires_raw_corroboration():
    raw = [Detection((100, 100, 120, 160), 0.10, 0, "person")]
    enhanced = [
        Detection((102, 101, 121, 161), 0.32, 0, "person"),
        Detection((300, 300, 320, 360), 0.40, 0, "person"),
        Detection((102, 101, 121, 161), 0.70, 2, "car"),
    ]
    kept = corroborate_enhanced_detections(raw, enhanced)
    assert len(kept) == 1
    assert kept[0].label == "person"
    assert kept[0].bbox == (102, 101, 121, 161)


def test_adaptive_region_pipeline_accepts_strong_raw_and_corroborated_enhanced_only():
    frame = np.full((240, 320, 3), 12, dtype=np.uint8)

    def fake_detect(tile, threshold):
        if threshold <= 0.08:
            return [
                Detection((50, 40, 60, 80), 0.10, 0, "person"),
                Detection((150, 40, 160, 80), 0.25, 0, "person"),
            ]
        return [
            Detection((50, 40, 60, 80), 0.30, 0, "person"),
            Detection((250, 40, 260, 80), 0.45, 0, "person"),
        ]

    detections = detect_people_with_adaptive_regions(
        frame,
        [(0, 0, 320, 240)],
        fake_detect,
        person_conf=0.18,
        probe_conf=0.08,
    )
    boxes = {d.bbox for d in detections}
    assert boxes == {
        (50, 40, 60, 80),
        (150, 40, 160, 80),
    }


def test_adaptive_region_pipeline_offsets_tile_local_detections():
    frame = np.full((300, 400, 3), 12, dtype=np.uint8)

    def fake_detect(_tile, threshold):
        if threshold <= 0.08:
            return [Detection((10, 20, 20, 60), 0.25, 0, "person")]
        return []

    detections = detect_people_with_adaptive_regions(
        frame,
        [(100, 50, 300, 250)],
        fake_detect,
        person_conf=0.18,
        probe_conf=0.08,
    )
    assert [d.bbox for d in detections] == [(110, 70, 120, 110)]


def test_adaptive_region_pipeline_rejects_invalid_threshold_order():
    frame = np.zeros((240, 320, 3), dtype=np.uint8)

    with pytest.raises(ValueError):
        detect_people_with_adaptive_regions(
            frame,
            [(0, 0, 320, 240)],
            lambda _tile, _threshold: [],
            person_conf=0.05,
            probe_conf=0.08,
        )
