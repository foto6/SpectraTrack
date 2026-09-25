import numpy as np

from spectratrack.tracker import MultiObjectTracker, TrackerConfig
from spectratrack.types import Detection


def d(x, y, score=0.9, cls=0, w=80, h=80):
    return Detection((x, y, x + w, y + h), score, cls, "obj")


def test_low_confidence_detection_recovers_confirmed_track():
    tracker = MultiObjectTracker(config=TrackerConfig(high_conf=0.5, low_conf=0.1, new_track_conf=0.6, min_hits=2))
    first = tracker.update([d(20, 30, 0.95)])[0].track_id
    tracks = tracker.update([d(25, 30, 0.92)])
    assert tracks[0].confirmed
    tracks = tracker.update([d(31, 30, 0.22)])
    assert len(tracks) == 1
    assert tracks[0].track_id == first
    assert tracks[0].missed == 0


def test_low_confidence_box_does_not_spawn_new_track():
    tracker = MultiObjectTracker(config=TrackerConfig(high_conf=0.5, low_conf=0.1, new_track_conf=0.6))
    assert tracker.update([d(10, 10, 0.2)]) == []


def test_camera_affine_keeps_stationary_world_target_id():
    tracker = MultiObjectTracker(config=TrackerConfig(min_hits=2))
    first = tracker.update([d(100, 120, 0.95)])[0].track_id
    affine = np.array([[1.0, 0.0, 40.0], [0.0, 1.0, -6.0]], dtype=np.float64)
    tracks = tracker.update([d(140, 114, 0.94)], camera_affine=affine)
    assert len(tracks) == 1
    assert tracks[0].track_id == first
    assert tracks[0].confirmed


def test_tentative_false_positive_expires_quickly():
    tracker = MultiObjectTracker(config=TrackerConfig(tentative_max_missed=1, min_hits=3))
    tracker.update([d(50, 50, 0.95)])
    assert len(tracker.update([])) == 1
    assert tracker.update([]) == []


def test_same_class_crossing_with_vertical_separation_keeps_ids():
    tracker = MultiObjectTracker(config=TrackerConfig(min_hits=2, max_center_ratio=2.8))
    tracks = tracker.update([d(80, 90), d(300, 150)])
    id_top = min(tracks, key=lambda t: t.center[1]).track_id
    id_bottom = max(tracks, key=lambda t: t.center[1]).track_id

    sequence = [
        [(115, 90), (265, 150)],
        [(150, 90), (230, 150)],
        [(185, 90), (195, 150)],
        [(220, 90), (160, 150)],
    ]
    for (ax, ay), (bx, by) in sequence:
        tracks = tracker.update([d(ax, ay), d(bx, by)])

    top = min(tracks, key=lambda t: t.center[1])
    bottom = max(tracks, key=lambda t: t.center[1])
    assert top.track_id == id_top
    assert bottom.track_id == id_bottom
