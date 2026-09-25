from pathlib import Path

import pytest

from spectratrack.calibration import CameraCalibration


def test_center_is_zero_angle():
    c = CameraCalibration(1920, 1080, 90.0)
    yaw, pitch = c.angular_offset_deg(960, 540)
    assert yaw == pytest.approx(0.0)
    assert pitch == pytest.approx(0.0)


def test_edges_match_half_fov():
    c = CameraCalibration(1000, 500, 80.0, 40.0)
    yaw, pitch = c.angular_offset_deg(1000, 0)
    assert yaw == pytest.approx(40.0)
    assert pitch == pytest.approx(20.0)


def test_angular_size():
    c = CameraCalibration(1000, 500, 80.0, 40.0)
    aw, ah = c.angular_size_deg(250, 125)
    assert aw == pytest.approx(20.0)
    assert ah == pytest.approx(10.0)


def test_json_roundtrip(tmp_path: Path):
    path = tmp_path / "camera.json"
    c = CameraCalibration(1920, 1080, 84.0, name="test")
    c.to_json(path)
    loaded = CameraCalibration.from_json(path)
    assert loaded.width == 1920
    assert loaded.height == 1080
    assert loaded.hfov_deg == pytest.approx(84.0)
    assert loaded.name == "test"
