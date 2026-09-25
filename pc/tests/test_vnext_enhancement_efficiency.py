import json

import numpy as np
import pytest

from spectratrack.enhance import assess_frame_quality
from spectratrack.research.enhancement_efficiency import (
    ROI_SCHEMA,
    _jitter,
    _match_truth,
    apply_operation,
    current_operation_set,
    load_roi_manifest,
    operation_gate,
    quality_snapshot,
    selective_gate,
)
from spectratrack.types import Detection


def test_quality_snapshot_matches_current_router_formulas():
    yy, xx = np.indices((720, 960))
    texture = (30 + ((xx * 3 + yy * 5) % 180)).astype(np.uint8)
    frame = np.dstack([texture, texture, texture])

    current = assess_frame_quality(frame)
    snapshot = quality_snapshot(
        frame,
        max_side=640,
        compression_full_resolution=True,
    )

    assert snapshot.values == pytest.approx(current, abs=1e-7)
    assert set(snapshot.stage_ms) == {
        "grayscale",
        "reduce",
        "darkness",
        "blur",
        "noise",
        "compression",
        "low_resolution",
    }
    assert all(value >= 0.0 for value in snapshot.stage_ms.values())


def test_cached_current_adaptive_matches_current_operation_sequence():
    yy, xx = np.indices((240, 320))
    texture = (8 + ((xx + yy) % 20)).astype(np.uint8)
    frame = np.dstack([texture, texture, texture])
    quality = assess_frame_quality(frame)

    current, _ = apply_operation("current_adaptive", frame, quality)
    cached, _ = apply_operation("current_adaptive_cached", frame, quality)

    assert np.array_equal(current, cached)


def test_operation_gate_keeps_low_light_ops_dark_only():
    dark = np.full((120, 160, 3), 10, dtype=np.uint8)
    quality = assess_frame_quality(dark)

    assert operation_gate("gamma", dark, quality)
    assert operation_gate("clahe", dark, quality)
    assert operation_gate("gamma_clahe", dark, quality)


def test_current_operation_set_exposes_combination_members():
    dark = np.full((120, 160, 3), 10, dtype=np.uint8)
    quality = assess_frame_quality(dark)

    operations = current_operation_set(dark, quality)

    assert "gamma" in operations
    assert "clahe" in operations


def test_selective_gate_uses_external_evidence_without_own_scheduler():
    quality = {
        "darkness": 0.0,
        "blur": 0.0,
        "compression": 0.0,
        "low_resolution": 0.0,
        "noise": 0.0,
    }
    signals = {"weak_person": True, "known_track": False, "uncertainty": False}

    assert selective_gate("weak-person", quality, signals)
    assert selective_gate("quality-or-evidence", quality, signals)
    assert not selective_gate("quality", quality, signals)


def test_roi_manifest_requires_detector_owned_regions(tmp_path):
    path = tmp_path / "regions.jsonl"
    rows = [
        {
            "type": "metadata",
            "schema": ROI_SCHEMA,
            "source": "a1-or-a4",
        },
        {
            "type": "roi",
            "video": "clip.mp4",
            "frame": 7,
            "roi_id": "clip-7-r0",
            "bbox": [10, 20, 110, 220],
            "signals": {
                "weak_person": True,
                "known_track": False,
                "uncertainty": True,
            },
        },
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    metadata, records = load_roi_manifest(path)

    assert metadata["source"] == "a1-or-a4"
    assert len(records) == 1
    assert records[0].bbox == (10, 20, 110, 220)
    assert records[0].signals["weak_person"]


def test_roi_manifest_rejects_duplicate_roi_ids(tmp_path):
    path = tmp_path / "regions.jsonl"
    meta = {"type": "metadata", "schema": ROI_SCHEMA}
    roi = {
        "type": "roi",
        "video": "clip.mp4",
        "frame": 1,
        "roi_id": "same",
        "bbox": [0, 0, 50, 50],
    }
    path.write_text(
        "\n".join(json.dumps(row) for row in (meta, roi, roi)) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicate roi_id"):
        load_roi_manifest(path)


def test_match_truth_reports_recovery_candidate_and_prefusion_fp():
    truth = [("clip.mp4#0#p1", (0.0, 0.0, 20.0, 40.0))]
    detections = [
        Detection((0.0, 0.0, 20.0, 40.0), 0.8, 0, "person"),
        Detection((80.0, 80.0, 100.0, 120.0), 0.5, 0, "person"),
    ]

    matches, fp = _match_truth(truth, detections, 0.5)

    assert set(matches) == {"clip.mp4#0#p1"}
    assert fp == 1


def test_jitter_uses_gt_relative_localization_residuals():
    matches = {
        "clip.mp4#0#p1": (
            1.0,
            Detection((0.0, 0.0, 10.0, 20.0), 0.9, 0, "person"),
            (0.0, 0.0, 10.0, 20.0),
        ),
        "clip.mp4#1#p1": (
            0.8,
            Detection((2.0, 0.0, 12.0, 20.0), 0.9, 0, "person"),
            (0.0, 0.0, 10.0, 20.0),
        ),
    }

    jitter = _jitter(matches)

    assert jitter["steps"] == 1
    assert jitter["center_residual_step_mean"] == pytest.approx(0.2)
    assert jitter["size_residual_step_mean"] == pytest.approx(0.0)
