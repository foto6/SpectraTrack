import json
from pathlib import Path

import pytest

from spectratrack import vnext_qa


def _write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _all_coverage_tags():
    return [name for name in vnext_qa.REQUIRED_GOLDEN_COVERAGE if name != "negative"]


def test_inspect_split_accepts_complete_golden_coverage(monkeypatch, tmp_path: Path):
    video_root = tmp_path / "videos"
    video = video_root / "golden" / "clip.mp4"
    video.parent.mkdir(parents=True)
    video.write_bytes(b"video")
    gt = tmp_path / "golden.jsonl"
    _write_jsonl(
        gt,
        [
            {
                "video": "golden/clip.mp4",
                "frame": 0,
                "tags": _all_coverage_tags(),
                "objects": [
                    {"id": "p1", "label": "person", "bbox": [10, 10, 30, 30]},
                    {"id": "p2", "label": "person", "bbox": [40, 10, 65, 45]},
                ],
            },
            {
                "video": "golden/clip.mp4",
                "frame": 1,
                "tags": ["negative"],
                "objects": [],
            },
        ],
    )
    monkeypatch.setattr(
        vnext_qa,
        "inspect_qa_source",
        lambda _path, fps_override=None: {
            "kind": "video",
            "width": 100,
            "height": 80,
            "fps": 25.0,
            "frame_count": 2,
            "sha256": "a" * 64,
        },
    )
    monkeypatch.setattr(vnext_qa, "sha256_file", lambda _path: "a" * 64)

    report = vnext_qa.inspect_split(gt, video_root, "golden")

    assert report["errors"] == []
    assert report["coverage"]["negative"] == 1
    assert all(report["coverage"][name] > 0 for name in vnext_qa.REQUIRED_GOLDEN_COVERAGE)


def test_inspect_split_rejects_invalid_label_and_out_of_bounds(monkeypatch, tmp_path: Path):
    video_root = tmp_path / "videos"
    video = video_root / "golden" / "clip.mp4"
    video.parent.mkdir(parents=True)
    video.write_bytes(b"video")
    gt = tmp_path / "golden.jsonl"
    _write_jsonl(
        gt,
        [
            {
                "video": "golden/clip.mp4",
                "frame": 0,
                "tags": _all_coverage_tags(),
                "objects": [
                    {"id": "x1", "label": "car", "bbox": [90, 10, 120, 40]}
                ],
            },
            {"video": "golden/clip.mp4", "frame": 1, "tags": ["negative"], "objects": []},
        ],
    )
    monkeypatch.setattr(
        vnext_qa,
        "inspect_qa_source",
        lambda _path, fps_override=None: {
            "kind": "video",
            "width": 100,
            "height": 80,
            "fps": 25.0,
            "frame_count": 2,
            "sha256": "a" * 64,
        },
    )
    monkeypatch.setattr(vnext_qa, "sha256_file", lambda _path: "a" * 64)

    report = vnext_qa.inspect_split(gt, video_root, "golden")

    assert any("invalid label" in error for error in report["errors"])
    assert any("outside 100x80" in error for error in report["errors"])


def test_inspect_corpus_rejects_train_golden_sha_leakage(monkeypatch, tmp_path: Path):
    video_root = tmp_path / "videos"
    for split in ("train", "golden"):
        path = video_root / split / "same.mp4"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(split.encode("utf-8"))

    train_gt = tmp_path / "train.jsonl"
    golden_gt = tmp_path / "golden.jsonl"
    _write_jsonl(
        train_gt,
        [
            {
                "video": "train/same.mp4",
                "frame": 0,
                "tags": [],
                "objects": [{"id": "p1", "label": "person", "bbox": [0, 0, 10, 20]}],
            }
        ],
    )
    _write_jsonl(
        golden_gt,
        [
            {
                "video": "golden/same.mp4",
                "frame": 0,
                "tags": _all_coverage_tags(),
                "objects": [{"id": "p1", "label": "person", "bbox": [0, 0, 10, 20]}],
            },
            {"video": "golden/same.mp4", "frame": 1, "tags": ["negative"], "objects": []},
        ],
    )
    monkeypatch.setattr(
        vnext_qa,
        "inspect_qa_source",
        lambda _path, fps_override=None: {
            "kind": "video",
            "width": 100,
            "height": 80,
            "fps": 25.0,
            "frame_count": 2,
            "sha256": "a" * 64,
        },
    )
    monkeypatch.setattr(vnext_qa, "sha256_file", lambda _path: "f" * 64)

    report = vnext_qa.inspect_corpus(
        video_root=video_root,
        train_ground_truth=train_gt,
        golden_ground_truth=golden_gt,
    )

    assert not report["valid"]
    assert any("train/golden leakage" in error for error in report["errors"])


def _valid_report():
    coverage = {name: 1 for name in vnext_qa.REQUIRED_GOLDEN_COVERAGE}
    return {
        "valid": True,
        "errors": [],
        "train": None,
        "golden": {
            "ground_truth": "golden.jsonl",
            "ground_truth_sha256": "1" * 64,
            "coverage": coverage,
            "videos": [
                {
                    "video": "golden/clip.mp4",
                    "split": "golden",
                    "sha256": "2" * 64,
                    "width": 1920,
                    "height": 1080,
                    "fps": 25.0,
                    "frame_count": 100,
                    "annotated_frames": 10,
                    "valid_person_instances": 9,
                }
            ],
        },
    }


def test_frozen_manifest_requires_human_confirmation_and_detects_tamper(tmp_path: Path):
    with pytest.raises(ValueError, match="human confirmation"):
        vnext_qa.build_frozen_manifest(
            _valid_report(),
            revision="golden-r1",
            reviewer="human",
            human_confirmed=False,
        )

    manifest = vnext_qa.build_frozen_manifest(
        _valid_report(),
        revision="golden-r1",
        reviewer="human",
        human_confirmed=True,
    )
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")

    loaded = vnext_qa.load_frozen_manifest(path)
    assert loaded["revision"] == "golden-r1"

    manifest["revision"] = "tampered"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        vnext_qa.load_frozen_manifest(path)


def _write_replay(path: Path, frame_number=0, width=1920):
    rows = [
        {
            "type": "metadata",
            "schema": vnext_qa.REPLAY_SCHEMA,
            "source_commit": "a" * 40,
            "video": "golden/clip.mp4",
            "video_sha256": "b" * 64,
            "detector": "yolo-test",
            "model_sha256": "c" * 64,
            "provider": "DmlExecutionProvider",
            "config": {"mode": "standard"},
            "width": 1920,
            "height": 1080,
        },
        {
            "type": "frame",
            "video": "golden/clip.mp4",
            "frame": frame_number,
            "timestamp_s": 0.0,
            "width": width,
            "height": 1080,
            "detections": [
                {
                    "bbox": [100.0, 120.0, 180.0, 310.0],
                    "score": 0.82,
                    "class_id": 0,
                    "label": "person",
                    "appearance": None,
                }
            ],
        },
    ]
    _write_jsonl(path, rows)


def test_detection_replay_validation_checks_provenance_and_dimensions(tmp_path: Path):
    replay = tmp_path / "replay.jsonl"
    _write_replay(replay)

    report = vnext_qa.validate_detection_replay(replay)
    assert report["schema"] == vnext_qa.REPLAY_SCHEMA
    assert report["frame_records"] == 1
    assert report["detections"] == 1

    _write_replay(replay, width=1280)
    with pytest.raises(ValueError, match="dimensions differ"):
        vnext_qa.validate_detection_replay(replay)


def _qa_result(match_iou=0.5):
    return {
        "schema_version": 1,
        "run_name": "run",
        "revision": "d" * 40,
        "ground_truth_sha256": "1" * 64,
        "input_videos": [
            {
                "video": "golden/clip.mp4",
                "sha256": "2" * 64,
                "width": 1920,
                "height": 1080,
                "fps": 25.0,
                "frame_count": 100,
                "processed_frames": 100,
            }
        ],
        "model": {
            "path": "model.onnx",
            "sha256": "3" * 64,
            "providers": ["DmlExecutionProvider", "CPUExecutionProvider"],
        },
        "evaluation": {"label": "person", "match_iou": match_iou},
        "settings": {"detector_mode": "standard", "conf": 0.35},
        "performance": {
            "wall_seconds": 20.0,
            "onnx_inference_calls": 100,
            "onnx_calls_per_frame": 1.0,
            "processing_seconds_per_source_second": 5.0,
            "peak_vram_mb": None,
        },
        "metrics": {
            "recall": 0.9,
            "precision": 0.8,
            "false_negatives": 1,
            "false_positives": 2,
            "by_size": {
                "height_lt_24": {"recall": 0.5},
                "height_24_47": {"recall": 0.8},
                "height_48_95": {"recall": 0.9},
                "height_ge_96": {"recall": 1.0},
            },
            "tracking": {
                "recall": 0.85,
                "id_switches": 1,
                "fragmentations": 2,
                "mean_uninterrupted_track_length_annotated_frames": 6.0,
                "mean_recovery_latency_frames": 3.0,
            },
            "bbox_stability": {
                "detection": {
                    "normalized_center_jitter_mean": 0.02,
                    "temporal_iou_mean": 0.9,
                },
                "tracking": {
                    "normalized_center_jitter_mean": 0.01,
                    "temporal_iou_mean": 0.95,
                },
            },
        },
    }


def test_stamp_and_leaderboard_require_same_corpus_and_scoring(tmp_path: Path):
    manifest = vnext_qa.build_frozen_manifest(
        _valid_report(),
        revision="golden-r1",
        reviewer="human",
        human_confirmed=True,
    )
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    result_a = tmp_path / "a.json"
    result_b = tmp_path / "b.json"
    result_a.write_text(json.dumps(_qa_result()), encoding="utf-8")
    second = _qa_result()
    second["settings"] = {"detector_mode": "people-recall", "conf": 0.35}
    result_b.write_text(json.dumps(second), encoding="utf-8")

    stamped_a = vnext_qa.stamp_qa_result(
        manifest_path=manifest_path,
        result_path=result_a,
        role="Baseline",
        experiment="current",
    )
    stamped_b = vnext_qa.stamp_qa_result(
        manifest_path=manifest_path,
        result_path=result_b,
        role="A1",
        experiment="fusion",
    )
    run_a = tmp_path / "run_a.json"
    run_b = tmp_path / "run_b.json"
    run_a.write_text(json.dumps(stamped_a), encoding="utf-8")
    run_b.write_text(json.dumps(stamped_b), encoding="utf-8")

    leaderboard = vnext_qa.build_leaderboard(
        manifest_path=manifest_path,
        run_paths=[run_a, run_b],
    )
    assert [row["role"] for row in leaderboard["rows"]] == ["Baseline", "A1"]
    assert "does not rank" in leaderboard["note"]
    assert leaderboard["rows"][0]["onnx_calls_per_frame"] == pytest.approx(1.0)

    incompatible_result = tmp_path / "incompatible.json"
    incompatible_result.write_text(json.dumps(_qa_result(match_iou=0.4)), encoding="utf-8")
    stamped_bad = vnext_qa.stamp_qa_result(
        manifest_path=manifest_path,
        result_path=incompatible_result,
        role="A2",
        experiment="tracking",
    )
    run_bad = tmp_path / "run_bad.json"
    run_bad.write_text(json.dumps(stamped_bad), encoding="utf-8")

    with pytest.raises(ValueError, match="incompatible scoring settings"):
        vnext_qa.build_leaderboard(
            manifest_path=manifest_path,
            run_paths=[run_a, run_bad],
        )


def test_stamp_requires_full_subject_commit_sha(tmp_path: Path):
    manifest = vnext_qa.build_frozen_manifest(
        _valid_report(),
        revision="golden-r1",
        reviewer="human",
        human_confirmed=True,
    )
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    result = _qa_result()
    result["revision"] = "agent/vnext-detection"
    result_path = tmp_path / "result.json"
    result_path.write_text(json.dumps(result), encoding="utf-8")

    with pytest.raises(ValueError, match="40-hex"):
        vnext_qa.stamp_qa_result(
            manifest_path=manifest_path,
            result_path=result_path,
            role="A1",
            experiment="bad-revision",
        )


def test_leaderboard_rejects_tampered_stamped_payload(tmp_path: Path):
    manifest = vnext_qa.build_frozen_manifest(
        _valid_report(),
        revision="golden-r1",
        reviewer="human",
        human_confirmed=True,
    )
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    result_path = tmp_path / "result.json"
    result_path.write_text(json.dumps(_qa_result()), encoding="utf-8")

    stamped = vnext_qa.stamp_qa_result(
        manifest_path=manifest_path,
        result_path=result_path,
        role="A1",
        experiment="fusion",
    )
    stamped["result"]["settings"]["conf"] = 0.99
    run_path = tmp_path / "tampered.json"
    run_path.write_text(json.dumps(stamped), encoding="utf-8")

    with pytest.raises(ValueError, match="payload hash mismatch"):
        vnext_qa.build_leaderboard(
            manifest_path=manifest_path,
            run_paths=[run_path],
        )


def test_ignored_person_does_not_satisfy_negative_coverage(monkeypatch, tmp_path: Path):
    video_root = tmp_path / "videos"
    video = video_root / "golden" / "clip.mp4"
    video.parent.mkdir(parents=True)
    video.write_bytes(b"video")
    gt = tmp_path / "golden.jsonl"
    _write_jsonl(
        gt,
        [
            {
                "video": "golden/clip.mp4",
                "frame": 0,
                "tags": _all_coverage_tags() + ["negative"],
                "objects": [
                    {
                        "id": "p1",
                        "label": "person",
                        "bbox": [10, 10, 30, 30],
                        "ignore": True,
                    }
                ],
            }
        ],
    )
    monkeypatch.setattr(
        vnext_qa,
        "inspect_qa_source",
        lambda _path, fps_override=None: {
            "kind": "video",
            "width": 100,
            "height": 80,
            "fps": 25.0,
            "frame_count": 1,
            "sha256": "a" * 64,
        },
    )
    monkeypatch.setattr(vnext_qa, "sha256_file", lambda _path: "a" * 64)

    report = vnext_qa.inspect_split(gt, video_root, "golden")

    assert report["coverage"]["negative"] == 0
    assert any("negative" in error for error in report["errors"])


def test_replay_appearance_vector_requires_explicit_schema(tmp_path: Path):
    replay = tmp_path / "appearance.jsonl"
    rows = [
        {
            "type": "metadata",
            "schema": vnext_qa.REPLAY_SCHEMA,
            "source_commit": "a" * 40,
            "video": "golden/clip.mp4",
            "video_sha256": "b" * 64,
            "detector": "yolo-test",
            "model_sha256": "c" * 64,
            "provider": "DmlExecutionProvider",
            "config": {},
            "width": 1920,
            "height": 1080,
        },
        {
            "type": "frame",
            "video": "golden/clip.mp4",
            "frame": 0,
            "timestamp_s": 0.0,
            "width": 1920,
            "height": 1080,
            "detections": [
                {
                    "bbox": [100.0, 120.0, 180.0, 310.0],
                    "score": 0.82,
                    "class_id": 0,
                    "label": "person",
                    "appearance": [0.1, 0.2],
                }
            ],
        },
    ]
    _write_jsonl(replay, rows)

    with pytest.raises(ValueError, match="appearance_schema"):
        vnext_qa.validate_detection_replay(replay)

    rows[0]["config"]["appearance_schema"] = "spectratrack-hsv-appearance-v1"
    _write_jsonl(replay, rows)
    report = vnext_qa.validate_detection_replay(replay)
    assert report["detections"] == 1


def _external_evidence():
    return {
        "schema": vnext_qa.EVIDENCE_SCHEMA,
        "role": "A2",
        "experiment": "tracker-current",
        "scope": "tracker-replay",
        "source_commit": "4" * 40,
        "corpus_revision": "golden-r1",
        "corpus_sha256": None,
        "ground_truth_sha256": "1" * 64,
        "source_artifact_sha256": "5" * 64,
        "model": {
            "id": "current-yolo",
            "sha256": "3" * 64,
            "provider": "DmlExecutionProvider",
        },
        "config": {
            "tracker_candidate": "current",
            "replay_sha256": "6" * 64,
        },
        "evaluation": {"label": "person", "match_iou": 0.5},
        "quality": {
            "track_recall": 0.88,
            "id_switches": 1,
            "fragmentations": 2,
            "mean_track_length_annotated_frames": 8.0,
            "mean_recovery_latency_frames": 2.0,
            "tracking_center_jitter": 0.02,
            "tracking_temporal_iou": 0.91,
        },
        "compute": {
            "onnx_inference_calls": 0,
            "onnx_calls_per_frame": 0.0,
            "wall_seconds": 1.2,
            "processing_seconds_per_source_second": 0.03,
            "peak_vram_mb": None,
        },
        "provenance": {
            "replay_sha256": "6" * 64,
            "input_videos": [
                {
                    "video": "golden/clip.mp4",
                    "sha256": "2" * 64,
                    "width": 1920,
                    "height": 1080,
                }
            ],
        },
    }


def test_external_specialist_evidence_can_join_common_leaderboard(tmp_path: Path):
    manifest = vnext_qa.build_frozen_manifest(
        _valid_report(),
        revision="golden-r1",
        reviewer="human",
        human_confirmed=True,
    )
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    evidence = _external_evidence()
    evidence["corpus_sha256"] = manifest["corpus_sha256"]
    source_artifact = tmp_path / "tracking-result.json"
    source_artifact.write_text('{"schema":"tracking-result"}', encoding="utf-8")
    evidence["source_artifact_sha256"] = vnext_qa.sha256_file(source_artifact)
    evidence_path = tmp_path / "tracking-evidence.json"
    evidence_path.write_text(json.dumps(evidence), encoding="utf-8")

    stamped = vnext_qa.stamp_external_evidence(
        manifest_path=manifest_path,
        evidence_path=evidence_path,
        source_artifact_path=source_artifact,
    )
    run_path = tmp_path / "tracking.stamped.json"
    run_path.write_text(json.dumps(stamped), encoding="utf-8")

    leaderboard = vnext_qa.build_leaderboard(
        manifest_path=manifest_path,
        run_paths=[run_path],
    )
    row = leaderboard["rows"][0]
    assert row["scope"] == "tracker-replay"
    assert row["track_recall"] == pytest.approx(0.88)
    assert row["onnx_inference_calls"] == 0
    assert row["recall"] is None


def test_external_evidence_rejects_hidden_corpus_video_change(tmp_path: Path):
    manifest = vnext_qa.build_frozen_manifest(
        _valid_report(),
        revision="golden-r1",
        reviewer="human",
        human_confirmed=True,
    )
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    evidence = _external_evidence()
    evidence["corpus_sha256"] = manifest["corpus_sha256"]
    evidence["provenance"]["input_videos"][0]["sha256"] = "9" * 64
    source_artifact = tmp_path / "tracking-result.json"
    source_artifact.write_text('{"schema":"tracking-result"}', encoding="utf-8")
    evidence["source_artifact_sha256"] = vnext_qa.sha256_file(source_artifact)
    evidence_path = tmp_path / "bad-evidence.json"
    evidence_path.write_text(json.dumps(evidence), encoding="utf-8")

    with pytest.raises(ValueError, match="different sha256"):
        vnext_qa.stamp_external_evidence(
            manifest_path=manifest_path,
            evidence_path=evidence_path,
            source_artifact_path=source_artifact,
        )


def test_person_specific_coverage_cannot_be_satisfied_by_empty_tagged_frame(
    monkeypatch, tmp_path: Path
):
    video_root = tmp_path / "videos"
    video = video_root / "golden" / "empty.mp4"
    video.parent.mkdir(parents=True)
    video.write_bytes(b"video")
    gt = tmp_path / "golden.jsonl"
    _write_jsonl(
        gt,
        [
            {
                "video": "golden/empty.mp4",
                "frame": 0,
                "tags": list(vnext_qa.REQUIRED_GOLDEN_COVERAGE),
                "objects": [],
            }
        ],
    )
    monkeypatch.setattr(
        vnext_qa,
        "inspect_qa_source",
        lambda _path, fps_override=None: {
            "kind": "video",
            "width": 100,
            "height": 80,
            "fps": 25.0,
            "frame_count": 1,
            "sha256": "a" * 64,
        },
    )
    monkeypatch.setattr(vnext_qa, "sha256_file", lambda _path: "a" * 64)

    report = vnext_qa.inspect_split(gt, video_root, "golden")

    assert report["coverage"]["negative"] == 1
    assert report["coverage"]["night_dark"] == 1
    assert report["coverage"]["tiny_person"] == 0
    assert report["coverage"]["crossing_people"] == 0
    assert any("tiny_person" in error for error in report["errors"])

