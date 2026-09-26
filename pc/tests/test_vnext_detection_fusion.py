import json

import numpy as np
import pytest

from spectratrack.research.vnext_detection_fusion import (
    FusionCandidate,
    FusionConfig,
    ResearchFrame,
    collect_prefusion_people_recall,
    conservative_nmm_fusion,
    convert_prefusion_to_replay,
    evidence_aware_fusion,
    synthetic_frames,
    synthetic_report,
    weighted_coordinate_fusion,
    write_prefusion_dump,
)


def test_synthetic_stress_set_exposes_hard_nms_crowd_failure_and_jitter():
    report = synthetic_report(source_commit="test-sha")
    hard = report["methods"]["hard-nms"]
    nmm = report["methods"]["conservative-nmm"]
    weighted = report["methods"]["weighted"]

    assert report["source_commit"] == "test-sha"
    assert report["policy_runs"] == 0
    assert report["inference_calls"] == 0
    assert hard["fn"] == 1
    assert hard["fusion_mistakes"] == 1
    assert nmm["fn"] == 0
    assert nmm["fusion_mistakes"] == 0
    assert weighted["fn"] == 0
    assert weighted["fusion_mistakes"] == 0
    assert nmm["center_jitter_px"] < hard["center_jitter_px"]
    assert weighted["center_jitter_px"] < hard["center_jitter_px"]
    assert weighted["temporal_iou"] > hard["temporal_iou"]
    assert hard["f1"] == pytest.approx(2 * hard["precision"] * hard["recall"] / (hard["precision"] + hard["recall"]))


def test_conservative_candidates_do_not_merge_high_overlap_distinct_people():
    frame = next(item for item in synthetic_frames() if item.video.endswith("crowded_overlap"))
    config = FusionConfig()

    nmm = conservative_nmm_fusion(frame.candidates, config)
    weighted = weighted_coordinate_fusion(frame.candidates, config)

    assert len(nmm) == 2
    assert len(weighted) == 2
    assert all(len(item.members) == 1 for item in nmm)
    assert all(len(item.members) == 1 for item in weighted)


def test_weighted_coordinates_stay_between_duplicate_evidence():
    candidates = [
        FusionCandidate((0.0, 0.0, 20.0, 40.0), 0.9, 0, "person", "full", "full"),
        FusionCandidate((2.0, 2.0, 18.0, 38.0), 0.6, 0, "person", "tile", "tile:0"),
    ]
    result = weighted_coordinate_fusion(candidates, FusionConfig(iou_threshold=0.5))

    assert len(result) == 1
    x1, y1, x2, y2 = result[0].bbox
    assert 0.0 < x1 < 2.0
    assert 0.0 < y1 < 2.0
    assert 18.0 < x2 < 20.0
    assert 38.0 < y2 < 40.0


def test_evidence_aware_fusion_rejects_weak_single_source_but_keeps_corroborated_weak():
    config = FusionConfig(iou_threshold=0.5)
    single = [FusionCandidate((0.0, 0.0, 20.0, 40.0), 0.15, 0, "person", "tile", "tile:0")]
    corroborated = [
        FusionCandidate((0.0, 0.0, 20.0, 40.0), 0.15, 0, "person", "tile", "tile:0"),
        FusionCandidate((1.0, 1.0, 21.0, 41.0), 0.14, 0, "person", "tile", "tile:1"),
    ]

    assert evidence_aware_fusion(single, config) == []
    assert len(evidence_aware_fusion(corroborated, config)) == 1


def test_evidence_aware_fusion_keeps_full_medium_or_any_strong_detection():
    config = FusionConfig()
    full_medium = [FusionCandidate((0.0, 0.0, 20.0, 40.0), 0.21, 0, "person", "full", "full")]
    tile_medium = [FusionCandidate((0.0, 0.0, 20.0, 40.0), 0.21, 0, "person", "tile", "tile:0")]
    strong = [FusionCandidate((0.0, 0.0, 20.0, 40.0), 0.40, 0, "person", "tile", "tile:0")]

    assert len(evidence_aware_fusion(full_medium, config)) == 1
    assert evidence_aware_fusion(tile_medium, config) == []
    assert len(evidence_aware_fusion(strong, config)) == 1


class _FakeDetector:
    def __init__(self):
        self.labels = ["person", "car"]
        self.class_thresholds = {}
        self.last_inference_calls = 0
        self.last_stage_ms = {}

    def _reset_policy_metrics(self):
        self.last_inference_calls = 0
        self.last_stage_ms = {}

    def _detect_once(self, frame, thresholds):
        self.last_inference_calls += 1
        marker = int(frame[0, 0, 0])
        from spectratrack.types import Detection

        if frame.shape[1] > 400:
            return [Detection((100.0, 100.0, 150.0, 220.0), 0.7, 0, "person")]
        if marker == 2:
            return [
                Detection((10.0, 20.0, 30.0, 80.0), 0.8, 0, "person"),
                Detection((40.0, 20.0, 80.0, 80.0), 0.9, 1, "car"),
            ]
        return []


def test_prefusion_collection_preserves_decoder_local_outputs_before_cross_pass_merge():
    frame = np.zeros((400, 800, 3), dtype=np.uint8)
    frame[:, 400:] = 2
    detector = _FakeDetector()

    candidates = collect_prefusion_people_recall(
        detector,
        frame,
        person_threshold=0.12,
        tile_size=400,
        tile_overlap=0.0,
    )

    assert detector.last_inference_calls == 3
    assert [item.source_kind for item in candidates] == ["full", "tile"]
    assert candidates[1].bbox == (410.0, 20.0, 430.0, 80.0)
    assert all(item.label == "person" for item in candidates)


def test_prefusion_can_be_fused_into_canonical_replay_without_extra_detection_fields(tmp_path):
    frame = ResearchFrame(
        video="clip.mp4",
        frame=3,
        timestamp_s=0.12,
        width=640,
        height=360,
        candidates=(
            FusionCandidate((10.0, 20.0, 40.0, 100.0), 0.9, 0, "person", "full", "full"),
            FusionCandidate((11.0, 21.0, 39.0, 99.0), 0.8, 0, "person", "tile", "tile:0"),
        ),
    )
    prefusion = tmp_path / "prefusion.jsonl"
    replay = tmp_path / "replay.jsonl"
    metadata = {
        "source_commit": "abc",
        "video": "clip.mp4",
        "video_sha256": "video-sha",
        "detector": "current-yolo",
        "model_sha256": "model-sha",
        "provider": "CPUExecutionProvider",
        "config": {"person_conf": 0.12},
        "width": 640,
        "height": 360,
    }
    write_prefusion_dump(
        prefusion,
        metadata,
        [frame],
        {"policy_runs": 1, "inference_calls": 2, "wall_time_s": 0.5},
    )
    convert_prefusion_to_replay(prefusion, replay, method="weighted", config=FusionConfig())

    records = [json.loads(line) for line in replay.read_text(encoding="utf-8").splitlines()]
    assert records[0]["schema"] == "spectratrack-detection-replay-v1"
    assert records[1]["video"] == "clip.mp4"
    assert records[1]["frame"] == 3
    assert len(records[1]["detections"]) == 1
    detection = records[1]["detections"][0]
    assert set(detection) == {"bbox", "score", "class_id", "label", "appearance"}
    assert records[0]["config"]["cross_pass_fusion"]["method"] == "weighted"


def test_fusion_config_rejects_invalid_values():
    with pytest.raises(ValueError):
        FusionConfig(iou_threshold=0.0).validate()
    with pytest.raises(ValueError):
        FusionConfig(size_ratio=0.9).validate()
