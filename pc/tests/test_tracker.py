from spectratrack.tracker import MultiObjectTracker, bbox_iou
from spectratrack.types import Detection


def d(x, y, cls=0, score=0.9):
    return Detection((x, y, x + 100, y + 100), score, cls, "obj")


def test_iou_identity():
    assert bbox_iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0


def test_stable_id_for_motion():
    tracker = MultiObjectTracker(max_missed=3, min_hits=1)
    ids = []
    for i in range(10):
        tracks = tracker.update([d(i * 7, 40)])
        ids.append(tracks[0].track_id)
    assert len(set(ids)) == 1


def test_short_dropout_keeps_track():
    tracker = MultiObjectTracker(max_missed=4, min_hits=1)
    first = tracker.update([d(20, 20)])[0].track_id
    tracker.update([])
    tracker.update([])
    resumed = tracker.update([d(35, 20)])[0].track_id
    assert resumed == first


def test_class_mismatch_does_not_reuse_track():
    tracker = MultiObjectTracker(max_missed=4, min_hits=1)
    first = tracker.update([d(20, 20, cls=0)])[0].track_id
    tracks = tracker.update([d(22, 20, cls=2)])
    ids = {t.track_id for t in tracks}
    assert first in ids
    assert len(ids) == 2


def test_low_confidence_does_not_spawn_track():
    tracker = MultiObjectTracker(high_conf=0.45, low_conf=0.12, min_hits=1)
    assert tracker.update([d(10, 10, score=0.20)]) == []


def test_low_confidence_can_extend_existing_track():
    tracker = MultiObjectTracker(high_conf=0.45, low_conf=0.12, min_hits=1)
    tid = tracker.update([d(10, 10, score=0.90)])[0].track_id
    tracks = tracker.update([d(14, 10, score=0.20)])
    assert len(tracks) == 1
    assert tracks[0].track_id == tid
    assert tracks[0].missed == 0


def test_camera_motion_is_removed_from_target_velocity():
    tracker = MultiObjectTracker(min_hits=1)
    tr = tracker.update([d(100, 100)])[0]
    assert tr.track_id == 1
    # Whole image pans +20px; a static target should have near-zero residual motion.
    tr = tracker.update([d(120, 100)], camera_motion=(20.0, 0.0))[0]
    assert abs(tr.vx) < 1e-6
    assert abs(tr.vy) < 1e-6


def test_affine_translation_compensation():
    tracker = MultiObjectTracker(min_hits=1)
    tracker.update([d(100, 100)])
    affine = (1.0, 0.0, 25.0, 0.0, 1.0, -8.0)
    tr = tracker.update([d(125, 92)], camera_transform=affine)[0]
    assert abs(tr.vx) < 1e-6
    assert abs(tr.vy) < 1e-6


def test_affine_rotation_preserves_static_target_identity():
    tracker = MultiObjectTracker(min_hits=1, max_center_ratio=3.0)
    first = tracker.update([d(100, 100)])[0].track_id
    # 90-degree rotation around origin maps center (150,150) -> (-150,150).
    affine = (0.0, -1.0, 0.0, 1.0, 0.0, 0.0)
    tracks = tracker.update([Detection((-200, 100, -100, 200), 0.9, 0, "obj")], camera_transform=affine)
    assert any(t.track_id == first and t.missed == 0 for t in tracks)
