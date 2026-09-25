import numpy as np

from spectratrack.detector import YoloOnnxDetector, merge_detections, tiled_regions
from spectratrack.types import Detection


def _fake_detector():
    detector = YoloOnnxDetector.__new__(YoloOnnxDetector)
    detector.iou_threshold = 0.45

    def detect_at_confidence(frame, conf_threshold):
        mask = frame[:, :, 0] > 0
        ys, xs = np.where(mask)
        if not len(xs):
            return []
        score = float(frame[:, :, 0].max()) / 255.0
        if score < conf_threshold:
            return []
        return [Detection(
            (float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)),
            score,
            0,
            "person",
        )]

    detector._detect_at_confidence = detect_at_confidence
    return detector


def test_tiled_regions_cover_edges_and_overlap():
    regions = tiled_regions(1280, 720, tile_size=512, overlap=0.20)
    assert regions[0] == (0, 0, 512, 512)
    assert regions[-1] == (768, 208, 1280, 720)
    assert len(regions) == 6


def test_tiled_detection_remaps_coordinates_and_merges_overlap_duplicates():
    detector = _fake_detector()
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    frame[300:330, 700:710, 0] = 80
    detections = detector.detect_tiled(frame, conf_threshold=0.18, tile_size=512, overlap=0.20)
    assert len(detections) == 1
    assert detections[0].bbox == (700.0, 300.0, 710.0, 330.0)


def test_enhanced_candidate_requires_raw_probe_corroboration():
    detector = _fake_detector()
    raw = np.zeros((720, 1280, 3), dtype=np.uint8)
    raw[300:330, 700:710, 0] = 30
    enhanced = raw.copy()
    enhanced[300:330, 700:710, 0] = 80

    assert detector.detect_people_recall(raw, person_conf=0.18, probe_conf=0.08) == []
    recovered = detector.detect_people_recall(raw, enhanced, person_conf=0.18, probe_conf=0.08)
    assert len(recovered) == 1

    hallucinated = np.zeros_like(raw)
    hallucinated[100:130, 100:110, 0] = 80
    assert detector.detect_people_recall(raw, hallucinated, person_conf=0.18, probe_conf=0.08) == []


def test_merge_detections_keeps_best_same_class_box():
    merged = merge_detections([
        Detection((10, 10, 40, 80), 0.4, 0, "person"),
        Detection((12, 12, 41, 79), 0.8, 0, "person"),
        Detection((12, 12, 41, 79), 0.7, 2, "car"),
    ])
    assert len(merged) == 2
    assert sorted((d.class_id, d.score) for d in merged) == [(0, 0.8), (2, 0.7)]
