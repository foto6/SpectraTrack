import numpy as np
import pytest

from spectratrack.enhance import DISPLAY_MODES, adaptive_analysis_frame, apply_display_mode, assess_frame_quality


@pytest.mark.parametrize("mode", DISPLAY_MODES)
def test_display_modes_preserve_shape_and_dtype(mode):
    frame = np.zeros((120, 160, 3), dtype=np.uint8)
    frame[:, 80:] = 180
    out = apply_display_mode(frame, mode)
    assert out.shape == frame.shape
    assert out.dtype == np.uint8


def test_unknown_display_mode_rejected():
    frame = np.zeros((20, 20, 3), dtype=np.uint8)
    with pytest.raises(ValueError):
        apply_display_mode(frame, "magic")


def test_quality_metrics_are_normalized_and_darkness_detected():
    frame = np.full((240, 320, 3), 10, dtype=np.uint8)
    quality = assess_frame_quality(frame)
    assert set(quality) == {"blur", "darkness", "compression", "low_resolution", "noise"}
    assert all(0.0 <= value <= 1.0 for value in quality.values())
    assert quality["darkness"] > 0.8
    assert quality["low_resolution"] > 0.5


def test_adaptive_analysis_preserves_shape_dtype_and_selects_low_light():
    frame = np.full((240, 320, 3), 12, dtype=np.uint8)
    out, quality, operations = adaptive_analysis_frame(frame)
    assert out.shape == frame.shape
    assert out.dtype == frame.dtype
    assert quality["darkness"] > 0.8
    assert "low_light" in operations


def test_well_lit_textured_frame_is_not_forced_through_processing():
    yy, xx = np.indices((720, 960))
    texture = (80 + ((xx + yy) % 32) * 4).clip(0, 255).astype(np.uint8)
    frame = np.dstack([texture, texture, texture])
    out, quality, operations = adaptive_analysis_frame(frame)
    assert operations == ()
    assert out is frame
    assert quality["darkness"] == 0.0
