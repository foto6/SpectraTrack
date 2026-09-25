import numpy as np

from spectratrack.calibration import CameraGeometry
from spectratrack.enhance import ENHANCE_MODES, apply_enhancement, next_enhancement_mode


def test_lowlight_mode_lifts_dark_frame_without_shape_change():
    frame = np.full((120, 160, 3), 28, dtype=np.uint8)
    out = apply_enhancement(frame, "lowlight")
    assert out.shape == frame.shape
    assert out.dtype == np.uint8
    assert float(out.mean()) > float(frame.mean())


def test_all_enhancement_modes_preserve_shape():
    rng = np.random.default_rng(7)
    frame = rng.integers(0, 256, size=(80, 120, 3), dtype=np.uint8)
    for mode in ENHANCE_MODES:
        out = apply_enhancement(frame, mode)
        assert out.shape == frame.shape
        assert out.dtype == np.uint8


def test_enhancement_cycle_wraps():
    mode = "off"
    for expected in ("visibility", "lowlight", "detail", "off"):
        mode = next_enhancement_mode(mode)
        assert mode == expected


def test_camera_geometry_center_and_edges():
    geom = CameraGeometry(90.0)
    assert abs(geom.bearing_offset_deg(500, 1000)) < 1e-9
    left = geom.bearing_offset_deg(0, 1000)
    right = geom.bearing_offset_deg(1000, 1000)
    assert left is not None and right is not None
    assert -46.0 < left < -44.0
    assert 44.0 < right < 46.0


def test_camera_geometry_angular_rate_sign():
    geom = CameraGeometry(70.0)
    rate = geom.angular_rate_deg_s(600, 4.0, 30.0, 1000)
    assert rate is not None
    assert rate > 0
