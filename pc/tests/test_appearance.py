import numpy as np
import pytest

from spectratrack.appearance import appearance_descriptor
from spectratrack.tracker import MultiObjectTracker, appearance_similarity
from spectratrack.types import Detection


def test_appearance_descriptor_distinguishes_colors():
    frame = np.zeros((120, 240, 3), dtype=np.uint8)
    frame[:, :120] = (0, 0, 255)
    frame[:, 120:] = (255, 0, 0)
    red = appearance_descriptor(frame, (0, 0, 120, 120))
    blue = appearance_descriptor(frame, (120, 0, 240, 120))
    assert red is not None and blue is not None
    assert appearance_similarity(red, red) == pytest.approx(1.0)
    assert appearance_similarity(red, blue) < 0.5


def test_appearance_helps_ambiguous_same_class_association():
    a = tuple([1.0] + [0.0] * 7)
    b = tuple([0.0, 1.0] + [0.0] * 6)
    tracker = MultiObjectTracker(min_hits=1, max_center_ratio=3.0)
    first = tracker.update([
        Detection((0, 0, 100, 100), 0.9, 0, "obj", a),
        Detection((200, 0, 300, 100), 0.9, 0, "obj", b),
    ])
    aid = first[0].track_id
    bid = first[1].track_id

    # Both candidates are deliberately centered nearly together; appearance should
    # keep each descriptor tied to its existing identity.
    tracks = tracker.update([
        Detection((95, 0, 195, 100), 0.9, 0, "obj", b),
        Detection((105, 0, 205, 100), 0.9, 0, "obj", a),
    ])
    by_id = {t.track_id: t for t in tracks}
    assert appearance_similarity(by_id[aid].appearance, a) > 0.9
    assert appearance_similarity(by_id[bid].appearance, b) > 0.9
