import hashlib
import json
from pathlib import Path

import pytest

from spectratrack.tracking_contiguous_pack import (
    EXPECTED_WINDOWS,
    PACK_SCHEMA,
    _frame_ids_from_payload,
    build_contiguous_pack,
    validate_contiguous_pack,
)


VIDEO = "golden/public/mot17/MOT17-04"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _line(value) -> str:
    return json.dumps(value, sort_keys=True) + "\n"


def _fixture(tmp_path: Path):
    replay = tmp_path / "source.replay.jsonl"
    gt = tmp_path / "source.gt.jsonl"
    observations = tmp_path / "source.observations.jsonl"
    selection = tmp_path / "mot17-04-smoke400-v1.json"

    replay_metadata = {
        "type": "metadata",
        "schema": "spectratrack-detection-replay-v1",
        "source_commit": "a1-source",
        "video": VIDEO,
        "video_sha256": "video-sha",
        "detector": "current-yolo-onnx-prefusion",
        "model_sha256": "model-sha",
        "provider": "DmlExecutionProvider,CPUExecutionProvider",
        "config": {},
        "width": 1920,
        "height": 1080,
    }
    observation_metadata = {
        "type": "metadata",
        "schema": "spectratrack-detection-prefusion-v1",
        "source_commit": "a1-source",
        "video": VIDEO,
        "video_sha256": "video-sha",
        "detector": "current-yolo-onnx-prefusion",
        "model_sha256": "model-sha",
        "provider": "DmlExecutionProvider,CPUExecutionProvider",
        "config": {},
        "width": 1920,
        "height": 1080,
    }

    with replay.open("w", encoding="utf-8") as handle:
        handle.write(_line(replay_metadata))
        for frame in range(600):
            handle.write(
                _line(
                    {
                        "type": "frame",
                        "video": VIDEO,
                        "frame": frame,
                        "timestamp_s": frame / 25.0,
                        "width": 1920,
                        "height": 1080,
                        "detections": [],
                    }
                )
            )

    with gt.open("w", encoding="utf-8") as handle:
        for frame in range(600):
            handle.write(
                _line(
                    {
                        "video": VIDEO,
                        "frame": frame,
                        "tags": [],
                        "objects": [],
                    }
                )
            )

    with observations.open("w", encoding="utf-8") as handle:
        handle.write(_line(observation_metadata))
        for frame in range(600):
            handle.write(
                _line(
                    {
                        "type": "frame",
                        "video": VIDEO,
                        "frame": frame,
                        "timestamp_s": frame / 25.0,
                        "width": 1920,
                        "height": 1080,
                        "candidates": [],
                    }
                )
            )
        handle.write(
            _line(
                {
                    "type": "summary",
                    "policy_runs": 600,
                    "inference_calls": 5400,
                }
            )
        )

    windows = []
    for ordinal, (start, end) in enumerate(EXPECTED_WINDOWS):
        windows.append(
            {
                "ordinal": ordinal,
                "start_position": start,
                "end_position": end,
                "frame_ids": list(range(start, end + 1)),
            }
        )
    selection.write_text(
        json.dumps(
            {
                "schema": "spectratrack-tracking-smoke-manifest-v1",
                "video": VIDEO,
                "source_replay_file_sha256": _sha(replay),
                "source_replay_canonical_sha256": "canonical-replay-sha",
                "ground_truth_sha256": _sha(gt),
                "source_frame_count": 600,
                "selected_frame_count": 400,
                "windows": windows,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return selection, replay, gt, observations


def test_build_pack_emits_exact_frozen_contiguous_windows_only(tmp_path):
    selection, replay, gt, observations = _fixture(tmp_path)
    output = tmp_path / "pack"
    manifest = build_contiguous_pack(
        selection_manifest_path=selection,
        replay_path=replay,
        ground_truth_path=gt,
        observations_path=observations,
        output_dir=output,
    )

    assert manifest["schema"] == PACK_SCHEMA
    assert manifest["selection"]["windows"] == [[0, 199], [400, 599]]
    assert manifest["selection"]["selected_frame_count"] == 400
    assert manifest["selection"]["excluded_middle_window"] == [200, 399]
    assert manifest["selection"]["nightowls_tracking_gt_used"] is False

    for window, (start, end) in zip(manifest["windows"], EXPECTED_WINDOWS):
        expected = tuple(range(start, end + 1))
        for kind in ("replay", "ground_truth", "observations"):
            path = output / window["payloads"][kind]["path"]
            assert _frame_ids_from_payload(path, expected_kind=kind) == expected
            assert not any(200 <= frame <= 399 for frame in expected)

    validate_contiguous_pack(output)


def test_build_pack_bytes_and_hashes_are_stable(tmp_path):
    selection, replay, gt, observations = _fixture(tmp_path)
    left = tmp_path / "left"
    right = tmp_path / "right"

    first = build_contiguous_pack(
        selection_manifest_path=selection,
        replay_path=replay,
        ground_truth_path=gt,
        observations_path=observations,
        output_dir=left,
    )
    second = build_contiguous_pack(
        selection_manifest_path=selection,
        replay_path=replay,
        ground_truth_path=gt,
        observations_path=observations,
        output_dir=right,
    )

    assert first["pack_digest"] == second["pack_digest"]
    assert sorted(path.name for path in left.iterdir()) == sorted(path.name for path in right.iterdir())
    for left_path in left.iterdir():
        assert left_path.read_bytes() == (right / left_path.name).read_bytes()


def test_pack_validation_fails_closed_on_payload_tamper(tmp_path):
    selection, replay, gt, observations = _fixture(tmp_path)
    output = tmp_path / "pack"
    build_contiguous_pack(
        selection_manifest_path=selection,
        replay_path=replay,
        ground_truth_path=gt,
        observations_path=observations,
        output_dir=output,
    )
    target = output / "mot17-04-000-199.gt.jsonl"
    target.write_bytes(target.read_bytes() + _line({"video": VIDEO, "frame": 250, "objects": []}).encode())

    with pytest.raises(ValueError, match="SHA mismatch"):
        validate_contiguous_pack(output)


def test_builder_rejects_source_hash_drift(tmp_path):
    selection, replay, gt, observations = _fixture(tmp_path)
    replay.write_bytes(replay.read_bytes() + b"\n")

    with pytest.raises(ValueError, match="source replay file SHA-256"):
        build_contiguous_pack(
            selection_manifest_path=selection,
            replay_path=replay,
            ground_truth_path=gt,
            observations_path=observations,
            output_dir=tmp_path / "pack",
        )


def test_builder_rejects_any_change_to_frozen_window_identity(tmp_path):
    selection, replay, gt, observations = _fixture(tmp_path)
    manifest = json.loads(selection.read_text(encoding="utf-8"))
    manifest["windows"][1]["frame_ids"][0] = 399
    selection.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="frame IDs changed"):
        build_contiguous_pack(
            selection_manifest_path=selection,
            replay_path=replay,
            ground_truth_path=gt,
            observations_path=observations,
            output_dir=tmp_path / "pack",
        )



def test_checked_in_canonical_pack_validates_when_present():
    pack = (
        Path(__file__).resolve().parents[1]
        / "benchmarks"
        / "vnext"
        / "tracking"
        / "canonical"
        / "mot17-04-contiguous-smoke400-r1"
    )
    if not (pack / "manifest.json").is_file():
        pytest.skip("canonical contiguous pack is materialized only from pinned target-PC A1 bytes")
    manifest = validate_contiguous_pack(pack)
    assert manifest["schema"] == PACK_SCHEMA
    assert manifest["selection"]["windows"] == [[0, 199], [400, 599]]
    assert manifest["selection"]["selected_frame_count"] == 400
    assert manifest["control_tracker"] == "current MultiObjectTracker"
    assert manifest["tracker_policy_changed"] is False
