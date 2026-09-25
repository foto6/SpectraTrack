import numpy as np
import pytest

from spectratrack.detector import (
    YoloOnnxDetector,
    _merge_detections,
    _looks_like_end2end,
    _tile_regions,
    _tile_starts,
    decode_end2end_predictions,
)
from spectratrack.types import Detection


def test_end2end_layout_detected():
    pred = np.array([
        [10, 20, 110, 220, 0.9, 0],
        [200, 50, 300, 180, 0.8, 2],
    ], dtype=np.float32)
    assert _looks_like_end2end(pred, 80)


def test_raw_small_feature_layout_not_assumed_for_two_labels():
    pred = np.zeros((10, 6), dtype=np.float32)
    assert not _looks_like_end2end(pred, 2)


def test_decode_end2end_xyxy_and_nms():
    pred = np.array([
        [100, 100, 300, 300, 0.95, 0],
        [110, 110, 295, 295, 0.70, 0],
        [400, 120, 500, 260, 0.80, 2],
    ], dtype=np.float32)
    out = decode_end2end_predictions(
        pred,
        frame_w=640,
        frame_h=640,
        input_w=640,
        input_h=640,
        scale=1.0,
        pad_x=0.0,
        pad_y=0.0,
        labels=["person"] + [f"c{i}" for i in range(1,80)],
        conf_threshold=0.3,
        iou_threshold=0.45,
    )
    assert len(out) == 2
    assert out[0].score >= 0.8
    assert {d.class_id for d in out} == {0, 2}


def test_decode_normalized_end2end():
    pred = np.array([[0.25, 0.25, 0.75, 0.75, 0.9, 0]], dtype=np.float32)
    out = decode_end2end_predictions(
        pred, 640, 640, 640, 640, 1.0, 0.0, 0.0,
        ["person"] + [f"c{i}" for i in range(1,80)], 0.3, 0.45
    )
    assert out[0].bbox == (160.0, 160.0, 480.0, 480.0)



def test_decode_end2end_supports_class_specific_person_threshold():
    pred = np.array([
        [100, 100, 200, 300, 0.18, 0],
        [250, 120, 400, 300, 0.34, 2],
    ], dtype=np.float32)
    out = decode_end2end_predictions(
        pred,
        frame_w=640,
        frame_h=640,
        input_w=640,
        input_h=640,
        scale=1.0,
        pad_x=0.0,
        pad_y=0.0,
        labels=["person", "bicycle", "car"] + [f"c{i}" for i in range(3, 80)],
        conf_threshold=0.35,
        iou_threshold=0.45,
        class_thresholds={"person": 0.15},
    )
    assert len(out) == 1
    assert out[0].label == "person"
    assert out[0].score == pytest.approx(0.18)


def test_tile_starts_cover_final_edge_without_out_of_bounds():
    assert _tile_starts(1920, 640, 0.20) == [0, 512, 1024, 1280]
    assert _tile_starts(500, 640, 0.20) == [0]


def test_merge_detections_is_class_aware_and_keeps_highest_score():
    detections = [
        Detection((10, 10, 110, 210), 0.90, 0, "person"),
        Detection((12, 12, 108, 208), 0.55, 0, "person"),
        Detection((12, 12, 108, 208), 0.80, 2, "car"),
    ]
    merged = _merge_detections(detections, 0.5)
    assert len(merged) == 2
    assert {(d.label, round(d.score, 2)) for d in merged} == {("person", 0.90), ("car", 0.80)}


class _FakeTiledDetector(YoloOnnxDetector):
    def __init__(self):
        self.labels = ["person", "car"]
        self.class_thresholds = {}
        self.conf_threshold = 0.35
        self.iou_threshold = 0.45

    def _detect_once(self, frame_bgr, class_thresholds=None):
        if frame_bgr.shape[1] == 400 and int(frame_bgr[0, 0, 0]) == 2:
            return [Detection((10.0, 20.0, 30.0, 80.0), 0.40, 0, "person")]
        return []


def test_people_recall_offsets_tile_detections_to_full_frame():
    frame = np.zeros((400, 800, 3), dtype=np.uint8)
    frame[:, :400] = 1
    frame[:, 400:] = 2
    detector = _FakeTiledDetector()

    detections = detector.detect_people_recall(
        frame,
        person_threshold=0.12,
        tile_size=400,
        tile_overlap=0.0,
    )

    assert len(detections) == 1
    assert detections[0].bbox == (410.0, 20.0, 430.0, 80.0)



class _FakeAdaptiveDetector(YoloOnnxDetector):
    def __init__(self):
        self.labels = ["person", "car"]
        self.class_thresholds = {}
        self.conf_threshold = 0.35
        self.iou_threshold = 0.45
        self.calls = []

    def _detect_once(self, frame_bgr, class_thresholds=None):
        threshold = float((class_thresholds or {}).get("person", self.conf_threshold))
        self.calls.append((frame_bgr.shape[:2], threshold, float(np.mean(frame_bgr))))
        if frame_bgr.shape[1] > 400:
            return []
        if threshold <= 0.08:
            return [Detection((50.0, 40.0, 70.0, 100.0), 0.10, 0, "person")]
        if float(np.mean(frame_bgr)) > 12.5:
            return [
                Detection((52.0, 42.0, 72.0, 102.0), 0.30, 0, "person"),
                Detection((250.0, 40.0, 270.0, 100.0), 0.45, 0, "person"),
            ]
        return []


def test_people_recall_adaptive_uses_a1_regions_and_raw_corroboration():
    frame = np.full((400, 800, 3), 12, dtype=np.uint8)
    detector = _FakeAdaptiveDetector()

    assert _tile_regions(800, 400, 400, 0.0) == [
        (0, 0, 400, 400),
        (400, 0, 800, 400),
    ]
    detections = detector.detect_people_recall(
        frame,
        person_threshold=0.18,
        tile_size=400,
        tile_overlap=0.0,
        merge_iou_threshold=0.5,
        enhancement_mode="adaptive",
    )

    # Each tile gets one weak raw probe and one enhanced pass. The unsupported
    # enhanced candidate at x=250 is rejected because no raw probe corroborates it.
    assert len(detector.calls) == 5  # one full-frame pass + 2 raw + 2 enhanced
    assert {(round(d.bbox[0]), round(d.bbox[1])) for d in detections} == {
        (52, 42),
        (452, 42),
    }


def test_people_recall_adaptive_still_uses_a1_final_nms(monkeypatch):
    detector = _FakeTiledDetector()
    frame = np.zeros((400, 800, 3), dtype=np.uint8)
    captured = {}

    def fake_adaptive(_frame, regions, _callback, **_kwargs):
        captured["regions"] = list(regions)
        return [
            Detection((100.0, 100.0, 160.0, 220.0), 0.80, 0, "person"),
            Detection((102.0, 102.0, 158.0, 218.0), 0.55, 0, "person"),
        ]

    monkeypatch.setattr("spectratrack.detector.detect_people_with_adaptive_regions", fake_adaptive)
    detections = detector.detect_people_recall(
        frame,
        person_threshold=0.18,
        tile_size=400,
        tile_overlap=0.0,
        merge_iou_threshold=0.5,
        enhancement_mode="adaptive",
    )

    assert captured["regions"] == [(0, 0, 400, 400), (400, 0, 800, 400)]
    assert len(detections) == 1
    assert detections[0].score == pytest.approx(0.80)



def _instrumented_fake_detector():
    class FakeSession:
        def __init__(self):
            self.calls = 0

        def run(self, _outputs, _feeds):
            self.calls += 1
            rows = [
                [5 + i, 5 + i, 20 + i, 30 + i, 0.9, 0]
                for i in range(10)
            ]
            return [np.asarray([rows], dtype=np.float32)]

    detector = YoloOnnxDetector.__new__(YoloOnnxDetector)
    detector.input_w = 64
    detector.input_h = 64
    detector.input_name = "images"
    detector.session = FakeSession()
    detector.labels = ["person"] + [f"c{i}" for i in range(1, 80)]
    detector.conf_threshold = 0.3
    detector.iou_threshold = 0.45
    detector.class_thresholds = {}
    detector.last_stage_ms = {}
    detector.last_inference_calls = 0
    return detector


def test_detector_records_preprocess_inference_and_postprocess_timings():
    detector = _instrumented_fake_detector()
    detections = detector.detect(np.zeros((64, 64, 3), dtype=np.uint8))

    assert detections
    assert set(detector.last_stage_ms) == {"preprocess", "inference", "postprocess"}
    assert all(value >= 0.0 for value in detector.last_stage_ms.values())
    assert detector.last_inference_calls == 1


def test_people_recall_counts_all_raw_tile_inference_calls():
    detector = _instrumented_fake_detector()
    frame = np.zeros((64, 128, 3), dtype=np.uint8)

    detector.detect_people_recall(
        frame,
        person_threshold=0.18,
        tile_size=64,
        tile_overlap=0.0,
        enhancement_mode="off",
    )

    assert detector.last_inference_calls == 3
    assert detector.session.calls == 3
    assert detector.last_stage_ms["inference"] >= 0.0


def test_adaptive_people_recall_counts_enhanced_inference_calls():
    detector = _instrumented_fake_detector()
    frame = np.full((64, 128, 3), 12, dtype=np.uint8)

    detector.detect_people_recall(
        frame,
        person_threshold=0.18,
        tile_size=64,
        tile_overlap=0.0,
        enhancement_mode="adaptive",
    )

    assert detector.last_inference_calls == 5
    assert detector.session.calls == 5
