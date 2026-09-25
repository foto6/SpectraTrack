import numpy as np
import pytest

from spectratrack.enhance import DISPLAY_MODES, apply_display_mode


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
