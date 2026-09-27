from spectratrack.tracker import MultiObjectTracker, bbox_iou
from spectratrack.types import Detection


def d(x, y, cls=0, score=0.9, appearance=None):
    return Detection((x, y, x + 100, y + 100), score, cls, "obj", appearance)


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


def test_two_same_class_targets_cross_without_immediate_id_swap():
    tracker = MultiObjectTracker(min_hits=1, max_center_ratio=2.5)
    initial = tracker.update([d(100, 100), d(400, 100)])
    left_id = min(initial, key=lambda t: t.center[0]).track_id
    right_id = max(initial, key=lambda t: t.center[0]).track_id

    # Approach each other, but do not make positions fully ambiguous in a single frame.
    for xa, xb in [(140, 360), (180, 320), (220, 280)]:
        tracks = tracker.update([d(xa, 100), d(xb, 100)])
        by_id = {t.track_id: t for t in tracks}
        assert left_id in by_id and right_id in by_id

    tracks = tracker.update([d(260, 100), d(240, 100)])
    by_id = {t.track_id: t for t in tracks}
    assert left_id in by_id and right_id in by_id
    assert by_id[left_id].center[0] > by_id[right_id].center[0]


def test_long_dropout_reactivates_confirmed_track_with_multiple_cues():
    appearance = tuple([1.0] + [0.0] * 7)
    tracker = MultiObjectTracker(max_missed=2, min_hits=1, reactivation_window=6)
    first = tracker.update([d(50, 50, appearance=appearance)])[0].track_id

    tracker.update([])
    tracker.update([])
    assert tracker.update([]) == []

    resumed = tracker.update([d(80, 50, appearance=appearance)])[0]
    assert resumed.track_id == first
    assert resumed.recoveries == 1
    assert resumed.missed == 0


def test_long_dropout_does_not_reactivate_different_appearance():
    appearance_a = tuple([1.0] + [0.0] * 7)
    appearance_b = tuple([0.0, 1.0] + [0.0] * 6)
    tracker = MultiObjectTracker(max_missed=1, min_hits=1, reactivation_window=6)
    first = tracker.update([d(50, 50, appearance=appearance_a)])[0].track_id

    tracker.update([])
    assert tracker.update([]) == []

    resumed = tracker.update([d(55, 50, appearance=appearance_b)])[0]
    assert resumed.track_id != first


def test_recovery_counter_increments_after_occlusion():
    tracker = MultiObjectTracker(min_hits=1, max_missed=5)
    tid = tracker.update([d(50, 50)])[0].track_id
    tracker.update([])
    tracker.update([])
    tr = tracker.update([d(55, 50)])[0]
    assert tr.track_id == tid
    assert tr.recoveries == 1
    assert tr.lifecycle == "TRACKED"


def test_predicted_lifecycle_during_short_loss():
    tracker = MultiObjectTracker(min_hits=1, max_missed=5)
    tracker.update([d(50, 50)])
    tr = tracker.update([])[0]
    assert tr.confirmed
    assert tr.missed == 1
    assert tr.lifecycle == "PREDICTED"
    assert 0.0 <= tr.quality <= 1.0


def test_large_camera_pan_with_affine_keeps_id():
    tracker = MultiObjectTracker(min_hits=1)
    tid = tracker.update([d(300, 200)])[0].track_id
    affine = (1.0, 0.0, 180.0, 0.0, 1.0, 40.0)
    tr = tracker.update([d(480, 240)], camera_transform=affine)[0]
    assert tr.track_id == tid
    assert tr.missed == 0
    assert abs(tr.vx) < 1e-6
    assert abs(tr.vy) < 1e-6


def test_tentative_track_expires_quickly():
    tracker = MultiObjectTracker(min_hits=3, max_missed=10)
    tr = tracker.update([d(10, 10)])[0]
    assert not tr.confirmed
    tracker.update([])
    tracker.update([])
    tracks = tracker.update([])
    assert all(t.track_id != tr.track_id for t in tracks)


def test_predict_only_does_not_increment_missed():
    tracker = MultiObjectTracker(min_hits=1)
    tr = tracker.update([d(100, 100)])[0]
    assert tr.missed == 0
    tr = tracker.predict_only(camera_motion=(5.0, 0.0))[0]
    assert tr.track_id == 1
    assert tr.missed == 0
    assert tr.age == 2
    assert tr.center[0] == 155.0


def test_long_dropout_without_appearance_does_not_reactivate():
    tracker = MultiObjectTracker(max_missed=1, min_hits=1, reactivation_window=6)
    first = tracker.update([d(50, 50)])[0].track_id

    tracker.update([])
    assert tracker.update([]) == []

    resumed = tracker.update([d(55, 50)])[0]
    assert resumed.track_id != first


def test_dormant_track_expires_after_reactivation_window():
    appearance = tuple([1.0] + [0.0] * 7)
    tracker = MultiObjectTracker(max_missed=1, min_hits=1, reactivation_window=2)
    first = tracker.update([d(50, 50, appearance=appearance)])[0].track_id

    tracker.update([])
    assert tracker.update([]) == []
    tracker.update([])
    tracker.update([])
    tracker.update([])

    resumed = tracker.update([d(55, 50, appearance=appearance)])[0]
    assert resumed.track_id != first


def test_reset_clears_dormant_tracks():
    appearance = tuple([1.0] + [0.0] * 7)
    tracker = MultiObjectTracker(max_missed=1, min_hits=1, reactivation_window=6)
    first = tracker.update([d(50, 50, appearance=appearance)])[0].track_id

    tracker.update([])
    assert tracker.update([]) == []
    tracker.reset()

    resumed = tracker.update([d(55, 50, appearance=appearance)])[0]
    assert resumed.track_id == 1
    assert resumed.recoveries == 0
    assert first == 1


def test_dormant_reactivation_tracks_camera_motion_across_skipped_frames():
    appearance = tuple([1.0] + [0.0] * 7)
    tracker = MultiObjectTracker(
        max_missed=1,
        min_hits=1,
        reactivation_window=6,
        reactivation_max_center_ratio=4.0,
    )
    first = tracker.update([d(50, 50, appearance=appearance)])[0].track_id

    affine_200 = (1.0, 0.0, 200.0, 0.0, 1.0, 0.0)
    tracker.update([], camera_transform=affine_200)
    assert tracker.update([], camera_transform=affine_200) == []

    affine_400 = (1.0, 0.0, 400.0, 0.0, 1.0, 0.0)
    tracker.predict_only(camera_transform=affine_400)
    resumed = tracker.update(
        [d(1250, 50, appearance=appearance)],
        camera_transform=affine_400,
    )[0]

    assert resumed.track_id == first
    assert resumed.recoveries == 1
