import numpy as np
import pytest

from spectratrack.enhancement_recall import (
    corroborate_enhanced_detections,
    detect_people_with_adaptive_tiles,
    iter_adaptive_tiles,
    tiled_regions,
    translate_detection,
)
from spectratrack.types import Detection


def test_tiled_regions_cover_edges_and_overlap():
    regions = tiled_regions(1280, 720, tile_size=512, overlap=0.20)
    assert regions[0] == (0, 0, 512, 512)
    assert regions[-1] == (768, 208, 1280, 720)
    assert len(regions) == 6


def test_tiled_regions_small_frame_uses_single_tile():
    assert tiled_regions(320, 240, tile_size=512, overlap=0.20) == [(0, 0, 320, 240)]


@pytest.mark.parametrize(
    ("width", "height", "tile_size", "overlap"),
    [
        (0, 720, 512, 0.20),
        (1280, 720, 32, 0.20),
        (1280, 720, 512, 0.80),
    ],
)
def test_tiled_regions_reject_invalid_settings(width, height, tile_size, overlap):
    with pytest.raises(ValueError):
        tiled_regions(width, height, tile_size=tile_size, overlap=overlap)


def test_iter_adaptive_tiles_keeps_raw_evidence_and_marks_processing():
    frame = np.full((240, 320, 3), 12, dtype=np.uint8)
    tiles = list(iter_adaptive_tiles(frame, tile_size=512, overlap=0.20))
    assert len(tiles) == 1

    region, raw_tile, enhanced_tile, quality, operations = tiles[0]
    assert region == (0, 0, 320, 240)
    assert raw_tile.shape == frame.shape
    assert enhanced_tile is not None
    assert enhanced_tile.shape == frame.shape
    assert quality["darkness"] > 0.8
    assert "low_light" in operations


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


def test_adaptive_tile_pipeline_accepts_strong_raw_and_corroborated_enhanced_only():
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

    detections = detect_people_with_adaptive_tiles(
        frame,
        fake_detect,
        person_conf=0.18,
        probe_conf=0.08,
        tile_size=512,
    )
    boxes = {d.bbox for d in detections}
    assert boxes == {
        (50, 40, 60, 80),
        (150, 40, 160, 80),
    }


def test_adaptive_tile_pipeline_rejects_invalid_threshold_order():
    frame = np.zeros((240, 320, 3), dtype=np.uint8)

    with pytest.raises(ValueError):
        detect_people_with_adaptive_tiles(
            frame,
            lambda _tile, _threshold: [],
            person_conf=0.05,
            probe_conf=0.08,
        )
