import cv2
import numpy as np

from spectratrack.frame_quality import measure_frame_quality


def test_dark_frame_detected():
    frame = np.zeros((120, 160, 3), dtype=np.uint8)
    q = measure_frame_quality(frame)
    assert q.low_light
    assert q.dark_fraction == 1.0


def test_checkerboard_is_sharp():
    gray = np.indices((160, 160)).sum(axis=0) % 2
    gray = (gray * 255).astype(np.uint8)
    frame = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    q = measure_frame_quality(frame)
    assert not q.blurred
    assert q.contrast > 100


def test_blurred_gradient_has_low_sharpness():
    x = np.linspace(30, 220, 320, dtype=np.uint8)
    gray = np.tile(x, (180, 1))
    gray = cv2.GaussianBlur(gray, (0, 0), 6.0)
    frame = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    q = measure_frame_quality(frame)
    assert q.sharpness < 55
