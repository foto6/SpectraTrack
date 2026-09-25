import numpy as np

from spectratrack.camera_calibration import CameraCalibration, Undistorter


def sample_calibration():
    return CameraCalibration(
        1000,
        500,
        np.array([[500.0, 0.0, 500.0], [0.0, 500.0, 250.0], [0.0, 0.0, 1.0]]),
        np.zeros(5),
        rms=0.2,
        used_images=12,
        mean_reprojection_error_px=0.1,
    )


def test_calibration_fov_and_scaling():
    c = sample_calibration()
    assert abs(c.hfov_deg - 90.0) < 1e-9
    scaled = c.scaled_camera_matrix(2000, 1000)
    assert scaled[0, 0] == 1000.0
    assert scaled[0, 2] == 1000.0
    assert scaled[1, 1] == 1000.0
    assert scaled[1, 2] == 500.0


def test_calibration_json_roundtrip(tmp_path):
    c = sample_calibration()
    p = tmp_path / "camera.json"
    c.save(p)
    loaded = CameraCalibration.load(p)
    np.testing.assert_allclose(loaded.camera_matrix, c.camera_matrix)
    np.testing.assert_allclose(loaded.dist_coeffs, c.dist_coeffs)
    assert loaded.used_images == 12


def test_zero_distortion_undistorter_preserves_shape():
    c = sample_calibration()
    frame = np.zeros((250, 500, 3), dtype=np.uint8)
    frame[50:120, 80:200] = 255
    out = Undistorter(c).apply(frame)
    assert out.shape == frame.shape
    assert out.dtype == frame.dtype
