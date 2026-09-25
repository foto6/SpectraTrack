from spectratrack.scheduler import DetectionScheduler


def test_fixed_interval_schedule():
    s = DetectionScheduler(fixed_interval=3, adaptive=False, max_interval=5)
    ran = [i for i in range(1, 9) if s.should_detect(i)]
    assert ran == [1, 4, 7]


def test_adaptive_scheduler_increases_interval_for_slow_detector():
    s = DetectionScheduler(adaptive=True, target_fps=30.0, max_interval=4)
    assert s.interval == 1
    s.observe_detector_ms(80.0)
    assert s.interval >= 3


def test_adaptive_scheduler_decreases_only_after_hysteresis():
    s = DetectionScheduler(adaptive=True, target_fps=30.0, max_interval=4)
    s.observe_detector_ms(100.0)
    assert s.interval == 4
    for _ in range(10):
        s.observe_detector_ms(1.0)
    assert s.interval >= 3
    for _ in range(40):
        s.observe_detector_ms(1.0)
    assert s.interval < 4
