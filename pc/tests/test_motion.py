import cv2
import numpy as np
import pytest

from spectratrack.motion import GlobalMotionEstimator


def synthetic_scene(width=640, height=360):
    img = np.zeros((height, width, 3), dtype=np.uint8)
    rng = np.random.default_rng(1234)
    for x, y in rng.integers([20, 20], [width-20, height-20], size=(120, 2)):
        cv2.circle(img, (int(x), int(y)), 2, (255, 255, 255), -1)
    cv2.rectangle(img, (80, 70), (180, 140), (180, 180, 180), 2)
    return img


def test_global_motion_estimator_translation():
    first = synthetic_scene()
    dx, dy = 17.0, -9.0
    mat = np.array([[1.0, 0.0, dx], [0.0, 1.0, dy]], dtype=np.float32)
    second = cv2.warpAffine(first, mat, (first.shape[1], first.shape[0]))

    est = GlobalMotionEstimator()
    initial = est.update(first)
    assert not initial.valid
    motion = est.update(second)
    assert motion.valid
    assert motion.dx == pytest.approx(dx, abs=1.5)
    assert motion.dy == pytest.approx(dy, abs=1.5)
    assert motion.inlier_ratio >= 0.35
    assert motion.scale == pytest.approx(1.0, abs=0.03)


def test_reset_forgets_previous_frame():
    frame = synthetic_scene()
    est = GlobalMotionEstimator()
    est.update(frame)
    est.reset()
    assert not est.update(frame).valid
