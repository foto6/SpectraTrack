import numpy as np

from spectratrack.kalman import KalmanBox


def test_kalman_camera_affine_moves_prediction():
    kf = KalmanBox((100, 100, 200, 200))
    affine = np.array([[1.0, 0.0, 25.0], [0.0, 1.0, -10.0]], dtype=np.float64)
    box = kf.predict(camera_affine=affine)
    cx = (box[0] + box[2]) * 0.5
    cy = (box[1] + box[3]) * 0.5
    assert abs(cx - 175.0) < 1e-6
    assert abs(cy - 140.0) < 1e-6


def test_kalman_measurement_update_moves_toward_detection():
    kf = KalmanBox((0, 0, 100, 100))
    kf.predict()
    before = kf.bbox
    after = kf.update((20, 0, 120, 100))
    assert after[0] > before[0]
    assert after[0] < 20.0
