from spectratrack.metrics import StageTimer


def test_timer_statistics():
    t = StageTimer(max_samples=4)
    for value in [1, 2, 3, 4, 10]:
        t.add("detect", value)
    assert t.latest("detect") == 10
    assert t.average("detect") == 4.75
    assert t.p95("detect") == 10


def test_unknown_stage_is_zero():
    t = StageTimer()
    assert t.latest("missing") == 0.0
    assert t.average("missing") == 0.0
    assert t.p95("missing") == 0.0
