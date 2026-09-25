import numpy as np

from spectratrack.mask import TargetMaskCache, grabcut_mask
from spectratrack.types import Track


def test_tiny_box_returns_no_mask():
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    assert grabcut_mask(frame, (10, 10, 15, 15)) is None


def test_grabcut_mask_has_frame_shape():
    frame = np.zeros((120, 160, 3), dtype=np.uint8)
    frame[35:90, 55:110] = 230
    mask = grabcut_mask(frame, (45, 25, 120, 100), iterations=1)
    assert mask is not None
    assert mask.shape == frame.shape[:2]
    assert mask.dtype == np.uint8


def test_mask_cache_clears_without_target():
    frame = np.zeros((120, 160, 3), dtype=np.uint8)
    track = Track(1, (40, 30, 100, 90), 0.9, 0, "obj", confirmed=True)
    cache = TargetMaskCache(update_every=5)
    cache.update(frame, track, 1)
    cache.update(frame, None, 2)
    assert cache.mask is None
    assert cache.track_id is None
