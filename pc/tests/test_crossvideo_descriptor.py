import cv2
import numpy as np

from spectratrack.appearance import crossvideo_descriptor
from spectratrack.crossvideo import TrackletSummary, tracklet_similarity


def test_crossvideo_descriptor_is_stable_for_same_crop():
    frame = np.zeros((160, 220, 3), dtype=np.uint8)
    cv2.rectangle(frame, (40, 30), (180, 140), (20, 40, 220), -1)
    cv2.line(frame, (40, 30), (180, 140), (255, 255, 255), 4)
    box = (40.0, 30.0, 180.0, 140.0)
    a = crossvideo_descriptor(frame, box)
    b = crossvideo_descriptor(frame.copy(), box)
    assert a is not None and b is not None
    assert len(a) > 100
    assert sum(x * y for x, y in zip(a, b)) > 0.999


def test_crossvideo_descriptor_separates_distinct_color_structure():
    red = np.zeros((160, 220, 3), dtype=np.uint8)
    blue = np.zeros_like(red)
    cv2.rectangle(red, (40, 30), (180, 140), (10, 10, 240), -1)
    cv2.rectangle(blue, (40, 30), (180, 140), (240, 10, 10), -1)
    box = (40.0, 30.0, 180.0, 140.0)
    a = crossvideo_descriptor(red, box)
    b = crossvideo_descriptor(blue, box)
    assert a is not None and b is not None
    similarity = sum(x * y for x, y in zip(a, b))
    assert similarity < 0.90


def _summary(video, descriptor, gallery):
    return TrackletSummary(
        video=video,
        local_track_id=1,
        class_id=2,
        label="car",
        first_frame=1,
        last_frame=10,
        observations=5,
        mean_score=0.9,
        mean_quality=0.8,
        best_frame=5,
        fps=30.0,
        descriptor=descriptor,
        gallery=gallery,
    )


def test_gallery_can_match_viewpoint_when_mean_is_weaker():
    front = (1.0, 0.0, 0.0)
    side = (0.0, 1.0, 0.0)
    mixed = (0.70710678, 0.70710678, 0.0)
    left = _summary("a.mp4", mixed, (front, side))
    right = _summary("b.mp4", front, (front,))
    score = tracklet_similarity(left, right)
    assert score is not None
    assert score > 0.85
