from spectratrack.scheduler import AdaptiveDetectorScheduler
from spectratrack.types import Track


def tr(quality_score=0.9, missed=0):
    t = Track(1, (0, 0, 100, 100), 0.95, 0, "obj", confirmed=True, last_detection_score=0.95)
    t.hits = 10
    t.missed = missed
    # Quality depends on last detection score/hits/missed. Lower last score when requested.
    t.last_detection_score = quality_score
    return t


def test_no_tracks_runs_detector():
    s = AdaptiveDetectorScheduler("speed")
    s.mark_detector_ran()
    assert s.decide([]).run_detector


def test_missing_track_forces_detector():
    s = AdaptiveDetectorScheduler("speed")
    s.mark_detector_ran()
    assert s.decide([tr(missed=1)]).run_detector


def test_high_motion_forces_detector():
    s = AdaptiveDetectorScheduler("speed")
    s.mark_detector_ran()
    assert s.decide([tr()], camera_motion=(30.0, 0.0)).run_detector


def test_stable_tracks_can_extend_interval():
    s = AdaptiveDetectorScheduler("balanced", max_interval=4)
    s.mark_detector_ran()
    d0 = s.decide([tr()])
    assert not d0.run_detector
    assert d0.interval >= 2
    s.mark_skipped()
    d1 = s.decide([tr()])
    # Depending on quality threshold, balanced high-quality can extend to 3.
    if not d1.run_detector:
        s.mark_skipped()
        assert s.decide([tr()]).run_detector


def test_max_interval_is_hard_limit():
    s = AdaptiveDetectorScheduler("speed", max_interval=2)
    s.frames_since_detection = 2
    assert s.decide([tr()]).run_detector
