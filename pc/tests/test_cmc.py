import cv2
import numpy as np

from spectratrack.cmc import CameraMotionEstimator


def synthetic_points(width=640, height=480):
    rng = np.random.default_rng(12345)
    image = np.zeros((height, width, 3), dtype=np.uint8)
    for x, y in rng.integers([20, 20], [width - 20, height - 20], size=(140, 2)):
        cv2.circle(image, (int(x), int(y)), 3, (255, 255, 255), -1)
    return image


def test_cmc_recovers_translation():
    first = synthetic_points()
    affine = np.array([[1.0, 0.0, 12.0], [0.0, 1.0, -7.0]], dtype=np.float32)
    second = cv2.warpAffine(first, affine, (first.shape[1], first.shape[0]))

    cmc = CameraMotionEstimator(min_features=12)
    assert not cmc.update(first).valid
    estimate = cmc.update(second)

    assert estimate.valid
    assert abs(estimate.dx - 12.0) < 2.0
    assert abs(estimate.dy + 7.0) < 2.0
    assert abs(estimate.rotation_deg) < 1.0
    assert estimate.inlier_ratio > 0.6
