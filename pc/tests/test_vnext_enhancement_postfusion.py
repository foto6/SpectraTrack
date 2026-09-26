import pytest

from spectratrack.research.strict_enhancement_summary import build_summary


def _qa_result(
    *,
    matched,
    recall,
    precision,
    fp,
    fn,
    wall=10.0,
    calls=100,
    model_sha="model-sha",
    gt_sha="gt-sha",
    center=0.10,
    width=0.20,
    height=0.30,
    area=0.40,
    temporal_iou=0.70,
):
    return {
        "schema_version": 1,
        "run_name": "run",
        "ground_truth_sha256": gt_sha,
        "model": {"sha256": model_sha},
        "evaluation": {"label": "person", "match_iou": 0.5},
        "performance": {
            "wall_seconds": wall,
            "onnx_inference_calls": calls,
            "processing_seconds_per_source_second": wall / 5.0,
        },
        "metrics": {
            "recall": recall,
            "precision": precision,
            "false_positives": fp,
            "false_negatives": fn,
            "matched_ground_truth": list(matched),
            "bbox_stability": {
                "detection": {
                    "pair_count": 5,
                    "normalized_center_jitter_mean": center,
                    "width_log_jitter_mean": width,
                    "height_log_jitter_mean": height,
                    "area_log_jitter_mean": area,
                    "temporal_iou_mean": temporal_iou,
                }
            },
        },
    }


def _profile(
    *,
    operation="bilateral",
    gate="weak-person",
    cap=1,
    raw=True,
    gt_sha="gt-sha",
    model_sha="model-sha",
    raw_calls=20,
    extra_calls=4,
):
    return {
        "schema": "spectratrack-vnext-enhancement-profile-v1",
        "source_commit": "deadbeef",
        "ground_truth": {
            "sha256": gt_sha,
            "corpus_revision": "nightowls-public-r1",
        },
        "model": {"sha256": model_sha},
        "settings": {
            "selective_gate": gate,
            "max_enhanced_rois_per_frame": cap,
            "raw_corroboration_required": raw,
        },
        "operations": {
            operation: {
                "cost": {
                    "raw_probe_calls": raw_calls,
                    "enhanced_inference_calls": extra_calls,
                    "operation_preprocessing_ms": 12.0,
                    "enhanced_inference_ms": 80.0,
                    "attributed_total_ms": 150.0,
                }
            }
        },
    }


def test_build_summary_reports_round2_quality_per_compute():
    baseline = _qa_result(
        matched={"p1", "p2"},
        recall=0.50,
        precision=0.80,
        fp=2,
        fn=2,
        center=0.20,
        temporal_iou=0.60,
    )
    candidate = _qa_result(
        matched={"p1", "p3", "p4"},
        recall=0.75,
        precision=0.75,
        fp=3,
        fn=1,
        wall=12.5,
        calls=104,
        center=0.15,
        temporal_iou=0.68,
    )

    summary = build_summary(
        baseline,
        candidate,
        _profile(extra_calls=4),
        operation="bilateral",
    )

    quality = summary["postfusion_quality"]
    compute = summary["compute"]
    assert quality["recovered_gt"] == ["p3", "p4"]
    assert quality["lost_gt"] == ["p2"]
    assert quality["recovered_gt_count"] == 2
    assert quality["lost_gt_count"] == 1
    assert quality["fp_delta"] == 1
    assert quality["recall_delta"] == pytest.approx(0.25)
    assert quality["precision_delta"] == pytest.approx(-0.05)

    assert compute["raw_onnx_calls"] == 20
    assert compute["extra_enhancement_onnx_calls"] == 4
    assert compute["recovered_gt_per_extra_call"] == pytest.approx(0.5)
    assert compute["fp_cost_per_recovered_gt"] == pytest.approx(0.5)
    assert compute["net_fp_delta_per_recovered_gt"] == pytest.approx(0.5)

    stability = summary["bbox_stability"]["delta"]
    assert stability["normalized_center_jitter_mean"] == pytest.approx(-0.05)
    assert stability["temporal_iou_mean"] == pytest.approx(0.08)

    performance = summary["postfusion_run_performance"]
    assert performance["wall_seconds_delta"] == pytest.approx(2.5)
    assert performance["postfusion_run_onnx_call_delta"] == pytest.approx(4.0)


def test_summary_rejects_non_strict_profile():
    baseline = _qa_result(
        matched={"p1"}, recall=1.0, precision=1.0, fp=0, fn=0
    )
    candidate = _qa_result(
        matched={"p1"}, recall=1.0, precision=1.0, fp=0, fn=0
    )

    with pytest.raises(ValueError, match="selective_gate=weak-person"):
        build_summary(
            baseline,
            candidate,
            _profile(gate="quality"),
            operation="bilateral",
        )

    with pytest.raises(ValueError, match="max_enhanced_rois_per_frame=1"):
        build_summary(
            baseline,
            candidate,
            _profile(cap=2),
            operation="bilateral",
        )

    with pytest.raises(ValueError, match="raw corroboration"):
        build_summary(
            baseline,
            candidate,
            _profile(raw=False),
            operation="bilateral",
        )


def test_summary_rejects_incomparable_postfusion_results():
    baseline = _qa_result(
        matched={"p1"}, recall=1.0, precision=1.0, fp=0, fn=0
    )
    candidate = _qa_result(
        matched={"p1"},
        recall=1.0,
        precision=1.0,
        fp=0,
        fn=0,
        gt_sha="other-gt",
    )

    with pytest.raises(ValueError, match="ground-truth"):
        build_summary(
            baseline,
            candidate,
            _profile(),
            operation="bilateral",
        )


def test_summary_rejects_profile_provenance_mismatch():
    baseline = _qa_result(
        matched={"p1"}, recall=1.0, precision=1.0, fp=0, fn=0
    )
    candidate = _qa_result(
        matched={"p1"}, recall=1.0, precision=1.0, fp=0, fn=0
    )

    with pytest.raises(ValueError, match="profile/post-fusion ground-truth"):
        build_summary(
            baseline,
            candidate,
            _profile(gt_sha="other-gt"),
            operation="bilateral",
        )

    with pytest.raises(ValueError, match="profile/post-fusion model"):
        build_summary(
            baseline,
            candidate,
            _profile(model_sha="other-model"),
            operation="bilateral",
        )


def test_zero_recovery_does_not_invent_efficiency_ratio():
    baseline = _qa_result(
        matched={"p1"}, recall=1.0, precision=1.0, fp=0, fn=0
    )
    candidate = _qa_result(
        matched={"p1"}, recall=1.0, precision=0.5, fp=1, fn=0
    )

    summary = build_summary(
        baseline,
        candidate,
        _profile(extra_calls=3),
        operation="bilateral",
    )

    assert summary["compute"]["recovered_gt_per_extra_call"] == 0.0
    assert summary["compute"]["fp_cost_per_recovered_gt"] is None
    assert summary["compute"]["net_fp_delta_per_recovered_gt"] is None
