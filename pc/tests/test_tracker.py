from spectratrack.tracker import MultiObjectTracker, bbox_iou
from spectratrack.types import Detection


def d(x, y, cls=0):
    return Detection((x, y, x + 100, y + 100), 0.9, cls, "obj")


def test_iou_identity():
    assert bbox_iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0


def test_stable_id_for_motion():
    tracker = MultiObjectTracker(max_missed=3)
    ids = []
    for i in range(10):
        tracks = tracker.update([d(i * 7, 40)])
        ids.append(tracks[0].track_id)
    assert len(set(ids)) == 1


def test_short_dropout_keeps_confirmed_track():
    tracker = MultiObjectTracker(max_missed=4)
    first = tracker.update([d(20, 20)])[0].track_id
    confirmed = tracker.update([d(27, 20)])[0]
    assert confirmed.track_id == first
    assert confirmed.confirmed
    tracker.update([])
    tracker.update([])
    resumed = tracker.update([d(42, 20)])[0].track_id
    assert resumed == first


def test_class_mismatch_does_not_reuse_track():
    tracker = MultiObjectTracker(max_missed=4)
    first = tracker.update([d(20, 20, cls=0)])[0].track_id
    tracks = tracker.update([d(22, 20, cls=2)])
    ids = {t.track_id for t in tracks}
    assert first in ids
    assert len(ids) == 2
