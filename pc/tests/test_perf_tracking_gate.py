from spectratrack.perf_tracking_gate import (
    LOCKED_CANDIDATES,
    evaluate_quality_gates,
    pareto_survivor,
)


def _metrics(**updates):
    metrics = {
        "tracking_recall": 0.90,
        "id_switches": 100,
        "fragmentations": 100,
        "mean_recovery_latency_frames": 5.0,
    }
    metrics.update(updates)
    return metrics


def test_candidate_set_is_prelocked_to_two_a4_points():
    assert [candidate["id"] for candidate in LOCKED_CANDIDATES] == [
        "detect-every-3-max-calls-1",
        "detect-every-5-max-calls-3",
    ]


def test_missing_quality_fails_closed_even_with_cost_reduction():
    gate = evaluate_quality_gates(
        _metrics(),
        None,
        discovery_latency_frames=None,
        discovery_bound_frames=24,
    )
    assert gate["decision"] == "STOP_INSUFFICIENT_QUALITY_EVIDENCE"
    assert pareto_survivor(quality_gate=gate, cost_reduced=True) is False


def test_recall_loss_over_quarter_percentage_point_rejects():
    gate = evaluate_quality_gates(
        _metrics(),
        _metrics(tracking_recall=0.8974),
        discovery_latency_frames=10,
        discovery_bound_frames=24,
    )
    assert gate["gates"]["tracking_recall_loss_max_0_25pp"] is False
    assert gate["decision"] == "REJECT_TRACKING_GATE"


def test_fragmentation_increase_over_five_percent_rejects():
    gate = evaluate_quality_gates(
        _metrics(),
        _metrics(fragmentations=106),
        discovery_latency_frames=10,
        discovery_bound_frames=24,
    )
    assert gate["gates"]["fragmentation_increase_max_5pct"] is False


def test_any_id_switch_increase_rejects():
    gate = evaluate_quality_gates(
        _metrics(),
        _metrics(id_switches=101),
        discovery_latency_frames=10,
        discovery_bound_frames=24,
    )
    assert gate["gates"]["no_id_switch_increase"] is False


def test_recovery_or_discovery_latency_bound_rejects():
    recovery = evaluate_quality_gates(
        _metrics(),
        _metrics(mean_recovery_latency_frames=6.01),
        discovery_latency_frames=10,
        discovery_bound_frames=24,
    )
    discovery = evaluate_quality_gates(
        _metrics(),
        _metrics(),
        discovery_latency_frames=25,
        discovery_bound_frames=24,
    )
    assert recovery["gates"]["recovery_latency_increase_max_1_frame"] is False
    assert discovery["gates"]["discovery_latency_within_periodic_bound"] is False


def test_all_quality_gates_plus_cost_are_required_for_survival():
    gate = evaluate_quality_gates(
        _metrics(),
        _metrics(
            tracking_recall=0.899,
            id_switches=99,
            fragmentations=104,
            mean_recovery_latency_frames=5.5,
        ),
        discovery_latency_frames=20,
        discovery_bound_frames=24,
    )
    assert gate["passed"] is True
    assert pareto_survivor(quality_gate=gate, cost_reduced=False) is False
    assert pareto_survivor(quality_gate=gate, cost_reduced=True) is True
