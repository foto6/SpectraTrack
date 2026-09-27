import json

import numpy as np
import pytest

from spectratrack.detector import YoloOnnxDetector
from spectratrack.performance_smoke import (
    amdahl_upper_bound,
    compare_profiles,
    distribution,
    load_selected_image_frames,
    research_cost_model,
)


class _MixedClassSession:
    def run(self, _outputs, _feeds):
        rows = []
        for index in range(10):
            if index % 2 == 0:
                rows.append([24 + index, 24, 12, 20, 0.90, 0.05])
            else:
                rows.append([24 + index, 24, 12, 20, 0.05, 0.90])
        pred = np.asarray(rows, dtype=np.float32).T[None, ...]
        return [pred]


def _detector() -> YoloOnnxDetector:
    detector = YoloOnnxDetector.__new__(YoloOnnxDetector)
    detector.input_w = 64
    detector.input_h = 64
    detector.input_name = "images"
    detector.session = _MixedClassSession()
    detector.labels = ["person", "car"]
    detector.conf_threshold = 0.35
    detector.iou_threshold = 0.45
    detector.class_thresholds = {}
    detector.last_stage_ms = {}
    detector.last_stage_cpu_ms = {}
    detector.last_policy_ms = {}
    detector.last_policy_counts = {}
    detector.last_inference_calls = 0
    detector.force_contiguous_input = False
    return detector


def _normalized(detections):
    return [
        (
            tuple(round(value, 6) for value in item.bbox),
            round(item.score, 6),
            item.class_id,
            item.label,
        )
        for item in detections
    ]


def test_distribution_reports_tail_percentiles():
    result = distribution([1, 2, 3, 4, 100])
    assert result["count"] == 5
    assert result["p50"] == 3
    assert result["p95"] == 100
    assert result["max"] == 100


def test_manifest_selector_resolves_video_frame_and_source_id(tmp_path):
    ground_truth = tmp_path / "gt.jsonl"
    rows = [
        {
            "video": "night/a",
            "frame": 0,
            "source": "frames/a.png",
            "source_frame": 100,
            "source_sequence": "seq-a",
            "objects": [],
        },
        {
            "video": "night/b",
            "frame": 2,
            "source": "frames/b.png",
            "source_frame": 101,
            "source_sequence": "seq-b",
            "objects": [],
        },
    ]
    ground_truth.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    manifest = tmp_path / "smoke.json"
    manifest.write_text(json.dumps({"frame_ids": ["night/a#0", 101]}), encoding="utf-8")

    selected, provenance = load_selected_image_frames(
        ground_truth,
        manifest,
        expected_frames=2,
    )

    assert [item.key for item in selected] == ["night/a#0", "night/b#2"]
    assert provenance["selected_frames"] == 2
    assert len(provenance["selection_manifest_sha256"]) == 64
    assert len(provenance["selection_keys_sha256"]) == 64


def test_manifest_selector_accepts_a5_selection_proof_entries(tmp_path):
    ground_truth = tmp_path / "smoke.jsonl"
    rows = [
        {
            "video": "night/a",
            "frame": 0,
            "source": "frames/a.png",
            "source_frame": 100,
            "source_sequence": "seq-a",
            "objects": [],
        },
        {
            "video": "night/b",
            "frame": 0,
            "source": "frames/b.png",
            "source_frame": 101,
            "source_sequence": "seq-b",
            "objects": [],
        },
    ]
    ground_truth.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    proof = tmp_path / "proof.json"
    proof.write_text(
        json.dumps(
            {
                "entries": [
                    {"source": "frames/b.png", "source_frame": 101},
                    {"source": "frames/a.png", "source_frame": 100},
                ]
            }
        ),
        encoding="utf-8",
    )

    selected, _ = load_selected_image_frames(ground_truth, proof, expected_frames=2)

    assert [item.source_frame if hasattr(item, "source_frame") else item.source for item in selected] == [
        "frames/b.png",
        "frames/a.png",
    ]


def test_manifest_selector_rejects_wrong_exact_count(tmp_path):
    ground_truth = tmp_path / "gt.jsonl"
    ground_truth.write_text(
        json.dumps(
            {
                "video": "night/a",
                "frame": 0,
                "source": "frames/a.png",
                "source_frame": 100,
                "objects": [],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    manifest = tmp_path / "smoke.json"
    manifest.write_text(json.dumps({"frame_ids": ["night/a#0"]}), encoding="utf-8")

    with pytest.raises(ValueError, match="expected exactly 2"):
        load_selected_image_frames(ground_truth, manifest, expected_frames=2)


def test_tile_person_only_postprocess_preserves_people_recall_outputs():
    frame = np.full((64, 128, 3), 180, dtype=np.uint8)
    control = _detector()
    candidate = _detector()

    control_output = control.detect_people_recall(
        frame,
        person_threshold=0.12,
        tile_size=64,
        tile_overlap=0.0,
        enhancement_mode="off",
        tile_person_only_postprocess=False,
    )
    candidate_output = candidate.detect_people_recall(
        frame,
        person_threshold=0.12,
        tile_size=64,
        tile_overlap=0.0,
        enhancement_mode="off",
        tile_person_only_postprocess=True,
    )

    assert _normalized(candidate_output) == _normalized(control_output)
    assert candidate.last_inference_calls == control.last_inference_calls == 3
    assert candidate.last_policy_counts["full_frame_calls"] == 1
    assert candidate.last_policy_counts["raw_roi_calls"] == 2
    assert set(candidate.last_stage_cpu_ms) == {"preprocess", "inference", "postprocess"}
    assert candidate.last_policy_ms["fusion"] >= 0.0


def test_amdahl_upper_bound_rejects_micro_candidate_below_gate():
    total_ms = 132477.6077
    preprocess_plus_postprocess_ms = 1674.9918
    assert amdahl_upper_bound(
        affected_ms=preprocess_plus_postprocess_ms,
        total_ms=total_ms,
    ) == pytest.approx(1.012806, rel=1e-5)


def test_research_cost_model_quantifies_smoke_first_savings():
    result = research_cost_model(
        smoke_wall_seconds=10.0,
        projected_full_seconds=100.0,
        candidate_count=3,
        promoted_count=1,
    )
    assert result["naive_full5000_seconds"] == 300.0
    assert result["smoke_first_seconds"] == 130.0
    assert result["saved_seconds"] == 170.0
    assert result["saved_fraction"] == pytest.approx(170 / 300)


def _profile(candidate, wall_seconds, output_hash):
    return {
        "candidate": candidate,
        "wall_seconds": wall_seconds,
        "output_sha256": output_hash,
        "quality": None,
        "provenance": {
            "model_sha256": "a" * 64,
            "video_sha256": "b" * 64,
        },
        "settings": {
            "input_size": 960,
            "person_conf": 0.12,
            "tile_size": 640,
            "tile_overlap": 0.20,
            "merge_iou": 0.55,
        },
    }


def test_compare_profiles_requires_meaningful_speedup_and_equal_output_hash():
    promoted = compare_profiles(
        _profile("control", 100.0, "same"),
        _profile("contiguous-input", 80.0, "same"),
        minimum_speedup=1.10,
    )
    assert promoted["speedup_x"] == pytest.approx(1.25)
    assert promoted["output_hash_equal"] is True
    assert promoted["promotion"] == "promote_to_full"

    stopped = compare_profiles(
        _profile("control", 100.0, "same"),
        _profile("tile-person-only-postprocess", 95.0, "different"),
        minimum_speedup=1.10,
    )
    assert stopped["promotion"] == "stop_on_smoke"
    assert stopped["output_hash_equal"] is False
    assert len(stopped["reasons"]) == 2
