import numpy as np

from spectratrack.detector import _looks_like_end2end, decode_end2end_predictions


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
