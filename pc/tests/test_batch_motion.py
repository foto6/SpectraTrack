import cv2
import numpy as np

from spectratrack.motion import GlobalMotionEstimator


def test_batch_camera_motion_prerequisite_affine_translation():
    rng = np.random.default_rng(123)
    first = np.zeros((240, 320, 3), dtype=np.uint8)
    for x, y in rng.integers([20, 20], [300, 220], size=(120, 2)):
        cv2.circle(first, (int(x), int(y)), 2, (255, 255, 255), -1)
    dx, dy = 9.0, -5.0
    matrix = np.array([[1.0, 0.0, dx], [0.0, 1.0, dy]], dtype=np.float32)
    second = cv2.warpAffine(first, matrix, (320, 240))

    motion = GlobalMotionEstimator()
    assert not motion.update(first).valid
    result = motion.update(second)
    assert result.valid
    assert abs(result.dx - dx) < 2.0
    assert abs(result.dy - dy) < 2.0
