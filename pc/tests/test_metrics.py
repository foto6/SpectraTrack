from spectratrack.metrics import ProcessCpuSampler, StageTimer


def test_timer_statistics():
    t = StageTimer(max_samples=4)
    for value in [1, 2, 3, 4, 10]:
        t.add("detect", value)
    assert t.latest("detect") == 10
    assert t.average("detect") == 4.75
    assert t.run_average("detect") == 4.0
    assert t.count("detect") == 5
    assert t.maximum("detect") == 10
    assert t.p95("detect") == 10
    assert t.summary("detect") == {
        "samples": 5,
        "avg": 4.0,
        "p95_recent": 10.0,
        "max": 10.0,
    }


def test_unknown_stage_is_zero():
    t = StageTimer()
    assert t.latest("missing") == 0.0
    assert t.average("missing") == 0.0
    assert t.run_average("missing") == 0.0
    assert t.count("missing") == 0
    assert t.maximum("missing") == 0.0
    assert t.p95("missing") == 0.0


def test_cpu_summary_is_explicit_when_no_sample_exists():
    sampler = ProcessCpuSampler()
    summary = sampler.summary()
    assert summary["samples"] == 0
    assert summary["avg_pct"] == 0.0
    assert summary["max_pct"] == 0.0
    assert summary["logical_cpus"] >= 1
    assert summary["scale"] == "percent_of_total_logical_cpu_capacity"


def test_cpu_sampler_normalizes_to_total_capacity(monkeypatch):
    sampler = ProcessCpuSampler()
    sampler.logical_cpus = 4
    sampler._last_wall = 10.0
    sampler._last_cpu = 5.0

    monkeypatch.setattr("spectratrack.metrics.time.perf_counter", lambda: 12.0)
    monkeypatch.setattr("spectratrack.metrics.time.process_time", lambda: 9.0)

    assert sampler.sample(force=True) == 50.0
    assert sampler.summary()["avg_pct"] == 50.0
