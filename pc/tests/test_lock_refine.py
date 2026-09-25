import cv2
import numpy as np
import pytest

from spectratrack.lock_refine import LockRefiner


def textured_target():
    frame = np.zeros((240, 360, 3), dtype=np.uint8)
    rng = np.random.default_rng(42)
    for x, y in rng.integers([90, 70], [190, 170], size=(80, 2)):
        cv2.circle(frame, (int(x), int(y)), 2, (255, 255, 255), -1)
    cv2.rectangle(frame, (90, 70), (190, 170), (120, 120, 120), 1)
    return frame


def test_lock_refiner_tracks_translation():
    first = textured_target()
    dx, dy = 13.0, -7.0
    mat = np.array([[1.0, 0.0, dx], [0.0, 1.0, dy]], dtype=np.float32)
    second = cv2.warpAffine(first, mat, (first.shape[1], first.shape[0]))

    refiner = LockRefiner()
    bbox = (90.0, 70.0, 190.0, 170.0)
    assert refiner.initialize(first, 7, bbox)
    result = refiner.update(second, 7, bbox)

    assert result.valid
    assert result.bbox[0] == pytest.approx(103.0, abs=2.0)
    assert result.bbox[1] == pytest.approx(63.0, abs=2.0)
    assert result.inlier_ratio >= 0.45


def test_lock_refiner_resets_on_track_change():
    frame = textured_target()
    refiner = LockRefiner()
    bbox = (90.0, 70.0, 190.0, 170.0)
    refiner.initialize(frame, 1, bbox)
    result = refiner.update(frame, 2, bbox)
    assert not result.valid
    assert refiner.track_id == 2


def test_lock_refiner_does_not_double_apply_current_detector_box():
    first = textured_target()
    dx, dy = 13.0, -7.0
    mat = np.array([[1.0, 0.0, dx], [0.0, 1.0, dy]], dtype=np.float32)
    second = cv2.warpAffine(first, mat, (first.shape[1], first.shape[0]))

    refiner = LockRefiner()
    old_bbox = (90.0, 70.0, 190.0, 170.0)
    current_detector_bbox = (103.0, 63.0, 203.0, 163.0)
    assert refiner.initialize(first, 7, old_bbox)
    result = refiner.update(second, 7, current_detector_bbox)

    assert result.valid
    assert result.bbox[0] == pytest.approx(103.0, abs=2.0)
    assert result.bbox[1] == pytest.approx(63.0, abs=2.0)
