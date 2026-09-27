from __future__ import annotations

from typing import Any

RECALL_LOSS_MAX_PP = 0.25
FRAGMENTATION_INCREASE_MAX_FRACTION = 0.05
RECOVERY_LATENCY_INCREASE_MAX_FRAMES = 1.0

LOCKED_CANDIDATES = (
    {
        "id": "detect-every-3-max-calls-1",
        "detect_every": 3,
        "max_calls_per_detector_frame": 1,
        "global_period": 8,
        "periodic_global_bound_frames": 24,
    },
    {
        "id": "detect-every-5-max-calls-3",
        "detect_every": 5,
        "max_calls_per_detector_frame": 3,
        "global_period": 5,
        "periodic_global_bound_frames": 25,
    },
)


def _fragmentation_increase_fraction(control: int, candidate: int) -> float:
    if control == 0:
        return 0.0 if candidate == 0 else float("inf")
    return (candidate - control) / control


def evaluate_quality_gates(
    control: dict[str, Any],
    candidate: dict[str, Any] | None,
    *,
    discovery_latency_frames: float | None,
    discovery_bound_frames: float,
) -> dict[str, Any]:
    """Apply the locked tracking-quality gates without collapsing them to a score."""
    if candidate is None:
        return {
            "quality_evidence_present": False,
            "passed": False,
            "decision": "STOP_INSUFFICIENT_QUALITY_EVIDENCE",
            "gates": {
                "tracking_recall_loss_max_0_25pp": None,
                "fragmentation_increase_max_5pct": None,
                "no_id_switch_increase": None,
                "recovery_latency_increase_max_1_frame": None,
                "discovery_latency_within_periodic_bound": None,
            },
        }

    recall_loss_pp = max(
        0.0,
        float(control["tracking_recall"]) - float(candidate["tracking_recall"]),
    ) * 100.0
    fragmentation_increase = _fragmentation_increase_fraction(
        int(control["fragmentations"]),
        int(candidate["fragmentations"]),
    )
    recovery_increase = (
        float(candidate["mean_recovery_latency_frames"])
        - float(control["mean_recovery_latency_frames"])
    )
    gates = {
        "tracking_recall_loss_max_0_25pp": recall_loss_pp <= RECALL_LOSS_MAX_PP + 1e-12,
        "fragmentation_increase_max_5pct": (
            fragmentation_increase <= FRAGMENTATION_INCREASE_MAX_FRACTION + 1e-12
        ),
        "no_id_switch_increase": int(candidate["id_switches"]) <= int(control["id_switches"]),
        "recovery_latency_increase_max_1_frame": (
            recovery_increase <= RECOVERY_LATENCY_INCREASE_MAX_FRAMES + 1e-12
        ),
        "discovery_latency_within_periodic_bound": (
            discovery_latency_frames is not None
            and discovery_latency_frames <= discovery_bound_frames + 1e-12
        ),
    }
    passed = all(gates.values())
    return {
        "quality_evidence_present": True,
        "passed": passed,
        "decision": "PARETO_ELIGIBLE" if passed else "REJECT_TRACKING_GATE",
        "gates": gates,
        "deltas": {
            "tracking_recall_loss_percentage_points": recall_loss_pp,
            "fragmentation_increase_fraction": fragmentation_increase,
            "id_switches": int(candidate["id_switches"]) - int(control["id_switches"]),
            "mean_recovery_latency_frames": recovery_increase,
        },
    }


def pareto_survivor(*, quality_gate: dict[str, Any], cost_reduced: bool) -> bool:
    """Cost reduction alone is never sufficient for advancement."""
    return bool(cost_reduced and quality_gate.get("passed") is True)
