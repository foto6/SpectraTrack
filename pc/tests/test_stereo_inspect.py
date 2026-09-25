import numpy as np

from spectratrack.stereo_inspect import colorize_disparity


def test_colorize_disparity_shape():
    d = np.full((40, 60), np.nan, dtype=np.float32)
    d[:, 10:50] = 12.0
    out = colorize_disparity(d)
    assert out.shape == (40, 60, 3)
    assert out.dtype == np.uint8


def test_colorize_empty_disparity():
    d = np.full((10, 20), np.nan, dtype=np.float32)
    out = colorize_disparity(d)
    assert out.shape == (10, 20, 3)
    assert int(out.sum()) == 0
