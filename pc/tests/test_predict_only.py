from spectratrack.tracker import MultiObjectTracker, TrackerConfig
from spectratrack.types import Detection


def d(x, score=0.95):
    return Detection((x, 20, x + 80, 100), score, 0, "obj")


def test_scheduled_skip_does_not_increment_missed():
    tracker = MultiObjectTracker(config=TrackerConfig(min_hits=2))
    tid = tracker.update([d(20)])[0].track_id
    confirmed = tracker.update([d(28)])[0]
    assert confirmed.confirmed

    predicted = tracker.update([], detector_ran=False)[0]
    assert predicted.track_id == tid
    assert predicted.missed == 0
    assert predicted.predicted_only
    assert predicted.state == "PREDICT"

    observed = tracker.update([d(44)], detector_ran=True)[0]
    assert observed.track_id == tid
    assert observed.missed == 0
    assert not observed.predicted_only
