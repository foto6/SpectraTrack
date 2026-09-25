import numpy as np
import pytest

from spectratrack.stereo import StereoDepthEstimator, StereoRigCalibration


def test_stereo_geometry():
    cal = StereoRigCalibration(640, 480, fx_px=800.0, baseline_m=0.12)
    assert cal.distance_from_disparity(24.0) == pytest.approx(4.0)


def test_invalid_disparity_rejected():
    cal = StereoRigCalibration(640, 480, fx_px=800.0, baseline_m=0.12)
    with pytest.raises(ValueError):
        cal.distance_from_disparity(0)


def test_bbox_range_uses_median_disparity():
    cal = StereoRigCalibration(100, 80, fx_px=500.0, baseline_m=0.10, num_disparities=16)
    est = StereoDepthEstimator(cal)
    disparity = np.full((80, 100), np.nan, dtype=np.float32)
    disparity[20:60, 30:70] = 10.0
    result = est.range_for_bbox(disparity, (20, 10, 80, 70))
    assert result is not None
    assert result.disparity_px == pytest.approx(10.0)
    assert result.distance_m == pytest.approx(5.0)


def test_bbox_range_requires_valid_fraction():
    cal = StereoRigCalibration(100, 80, fx_px=500.0, baseline_m=0.10, num_disparities=16)
    est = StereoDepthEstimator(cal)
    disparity = np.full((80, 100), np.nan, dtype=np.float32)
    disparity[40, 50] = 10.0
    assert est.range_for_bbox(disparity, (20, 10, 80, 70), min_valid_fraction=0.2) is None


def test_resolution_mismatch_rejected():
    cal = StereoRigCalibration(100, 80, fx_px=500.0, baseline_m=0.10, num_disparities=16)
    est = StereoDepthEstimator(cal)
    left = np.zeros((80, 100, 3), dtype=np.uint8)
    right = np.zeros((60, 100, 3), dtype=np.uint8)
    with pytest.raises(ValueError):
        est.compute_disparity(left, right)
