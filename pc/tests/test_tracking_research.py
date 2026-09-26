import json

from spectratrack.tracking_research import (
    CurrentTrackerRunner,
    GlobalAssignmentCurrentRunner,
    ReferenceStyleTracker,
    _aggregate,
    bbox_stability_probe,
    current_failure_audit,
    evaluate_tracking,
    mine_current_failure_windows,
    run_loaded_replay,
    run_synthetic_bakeoff,
    synthetic_scenarios,
)


def _scenario(name):
    return next(item for item in synthetic_scenarios() if item.name == name)


def _run(scenario, runner):
    outputs = {}
    for frame in scenario.replay.frames:
        outputs[frame.frame] = runner.step(frame)
    return evaluate_tracking(scenario, outputs)


def test_required_failure_probe_catalog_is_complete():
    names = {item.name for item in synthetic_scenarios()}
    assert names == {
        "crossing",
        "partial_occlusion",
        "full_occlusion",
        "short_dropout",
        "long_dropout",
        "dormant_reactivation",
        "appearance_mismatch",
        "camera_pan",
        "camera_transform",
        "detector_skipped_frames",
        "weak_detection_sequence",
        "false_weak_detections",
        "nearby_same_class",
        "changing_bbox_scale",
        "sudden_direction_change",
    }


def test_existing_low_confidence_rescue_is_measured_not_reimplemented():
    metrics = _run(_scenario("weak_detection_sequence"), CurrentTrackerRunner())
    assert metrics["id_switches"] == 0
    assert metrics["fragmentations"] == 0
    assert metrics["tracking_recall"] == 1.0

    false_metrics = _run(_scenario("false_weak_detections"), CurrentTrackerRunner())
    assert false_metrics["false_track_creations"] == 0


def test_dormant_reactivation_requires_appearance():
    no_appearance = _run(_scenario("long_dropout"), CurrentTrackerRunner())
    with_appearance = _run(_scenario("dormant_reactivation"), CurrentTrackerRunner())
    mismatch = _run(_scenario("appearance_mismatch"), CurrentTrackerRunner())

    assert no_appearance["id_switches"] >= 1
    assert with_appearance["id_switches"] == 0
    assert with_appearance["recovered_same_id"] >= 1
    assert mismatch["id_switches"] >= 1


def test_detector_skipped_frames_do_not_consume_miss_budget():
    metrics = _run(_scenario("detector_skipped_frames"), CurrentTrackerRunner())
    assert metrics["id_switches"] == 0
    assert metrics["fragmentations"] == 0
    assert metrics["tracking_recall"] == 1.0


def test_current_audit_records_high_low_and_reactivation_decisions():
    weak = current_failure_audit(_scenario("weak_detection_sequence"))
    stages = {event.get("stage") for event in weak["events"] if event.get("event") == "association"}
    assert {"high", "low_or_unmatched"} <= stages

    dormant = current_failure_audit(_scenario("dormant_reactivation"))
    reactivation = [event for event in dormant["events"] if event.get("event") == "reactivation"]
    assert reactivation
    assert any(event["matches"] for event in reactivation)


def test_failure_miner_surfaces_switch_and_reactivation_context():
    nearby = mine_current_failure_windows(_scenario("nearby_same_class"))
    assert nearby["summary"]["id_switch_events"] >= 2
    assert nearby["failure_windows"]
    assert any(item["category"] == "id_switch" for item in nearby["failure_windows"])

    dormant = mine_current_failure_windows(_scenario("dormant_reactivation"))
    assert dormant["summary"]["reactivation_matches"] >= 1
    assert dormant["reactivations"]


def test_all_candidates_consume_the_same_replay_fingerprint():
    report = run_synthetic_bakeoff(performance_repeats=1)
    assert report["detection_replay_schema"] == "spectratrack-detection-replay-v1"
    assert len(report["scenario_replay_sha256"]) == 15
    for candidate in report["candidates"].values():
        assert set(candidate["scenarios"]) == set(report["scenario_replay_sha256"])
    assert report["detector_policy_runs"] == 0
    assert report["onnx_inference_calls"] == 0


def test_loaded_replay_is_deterministic_and_does_not_run_detector():
    replay = _scenario("camera_pan").replay
    first = run_loaded_replay(replay, "current")
    second = run_loaded_replay(replay, "current")

    assert first["replay_canonical_sha256"] == second["replay_canonical_sha256"]
    assert first["frames"] == second["frames"]
    assert first["detector_policy_runs"] == 0
    assert first["onnx_inference_calls"] == 0


def test_reference_style_candidates_use_strong_only_creation():
    scenario = _scenario("false_weak_detections")
    for mode in ("byte", "botsort", "ocsort"):
        metrics = _run(scenario, ReferenceStyleTracker(mode))
        assert metrics["false_track_creations"] == 0
        assert metrics["matched_gt"] == 0


def test_global_assignment_probe_exposes_current_greedy_conflict():
    scenario = _scenario("nearby_same_class")
    current = _run(scenario, CurrentTrackerRunner())
    isolated_global = _run(scenario, GlobalAssignmentCurrentRunner())
    byte_style = _run(scenario, ReferenceStyleTracker("byte"))

    assert current["id_switches"] >= 2
    assert isolated_global["id_switches"] == 0
    assert byte_style["id_switches"] == 0
    assert isolated_global["mean_uninterrupted_track_length"] > current["mean_uninterrupted_track_length"]


def test_bbox_smoothing_reports_jitter_and_lag_separately():
    metrics = bbox_stability_probe()
    assert set(metrics) == {"raw", "bounded_ema", "motion_state"}
    assert metrics["bounded_ema"]["center_jitter_px"] < metrics["raw"]["center_jitter_px"]
    assert metrics["motion_state"]["center_jitter_px"] < metrics["raw"]["center_jitter_px"]
    assert metrics["bounded_ema"]["response_lag_px"] > metrics["raw"]["response_lag_px"]
    assert metrics["motion_state"]["response_lag_px"] > metrics["raw"]["response_lag_px"]
    for row in metrics.values():
        assert 0.0 <= row["temporal_iou"] <= 1.0
        assert 0.0 <= row["mean_iou_to_truth"] <= 1.0


def test_aggregate_means_are_weighted_by_actual_segments_and_recoveries():
    rows = [
        {
            "id_switches": 0,
            "fragmentations": 0,
            "tracking_recall": 1.0,
            "false_track_creations": 0,
            "mean_uninterrupted_track_length": 10.0,
            "uninterrupted_segments": 1,
            "mean_recovery_latency_frames": 2.0,
            "recovery_events": 1,
            "recovered_same_id": 1,
            "wrong_recovery": 0,
            "matched_gt": 10,
            "gt": 10,
        },
        {
            "id_switches": 0,
            "fragmentations": 0,
            "tracking_recall": 1.0,
            "false_track_creations": 0,
            "mean_uninterrupted_track_length": 2.0,
            "uninterrupted_segments": 3,
            "mean_recovery_latency_frames": 6.0,
            "recovery_events": 3,
            "recovered_same_id": 0,
            "wrong_recovery": 0,
            "matched_gt": 6,
            "gt": 6,
        },
    ]
    aggregate = _aggregate(rows)
    assert aggregate["mean_uninterrupted_track_length"] == 4.0
    assert aggregate["mean_recovery_latency_frames"] == 5.0
    assert aggregate["uninterrupted_segments"] == 4
    assert aggregate["recovery_events"] == 4


def test_research_report_quality_fields_are_json_serializable():
    report = run_synthetic_bakeoff(performance_repeats=1)
    encoded = json.dumps(report, sort_keys=True)
    assert '"current"' in encoded
    assert '"bbox_stability"' in encoded
