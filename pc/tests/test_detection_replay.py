import json

import pytest

from spectratrack.detection_replay import (
    DETECTION_REPLAY_SCHEMA,
    DetectionReplay,
    ReplayDetection,
    ReplayFrame,
    ReplayMetadata,
    canonical_replay_bytes,
    load_detection_replay,
    parse_detection_replay,
    write_detection_replay,
)


def sample_replay() -> DetectionReplay:
    return DetectionReplay(
        metadata=ReplayMetadata(
            source_commit="abc123",
            video="clip.mp4",
            video_sha256="video-sha",
            detector="synthetic",
            model_sha256=None,
            provider="CPUExecutionProvider",
            config={"conf": 0.12},
            width=640,
            height=360,
        ),
        frames=(
            ReplayFrame(
                video="clip.mp4",
                frame=0,
                timestamp_s=0.0,
                width=640,
                height=360,
                detections=(
                    ReplayDetection((10.0, 20.0, 40.0, 80.0), 0.9, 0, "person", (1.0, 0.0)),
                ),
            ),
            ReplayFrame(
                video="clip.mp4",
                frame=1,
                timestamp_s=1.0 / 30.0,
                width=640,
                height=360,
                detections=(),
                detector_ran=False,
                camera_motion=(4.0, -1.0),
            ),
        ),
    )


def test_round_trip_is_canonical_and_deterministic(tmp_path):
    replay = sample_replay()
    target = tmp_path / "replay.jsonl"
    written_sha = write_detection_replay(replay, target)

    first = load_detection_replay(target)
    second = load_detection_replay(target)

    assert first == second
    assert first.source_sha256 == written_sha
    assert first.canonical_sha256() == second.canonical_sha256()
    assert canonical_replay_bytes(first) == target.read_bytes()


def test_fresh_detections_do_not_share_mutable_detection_objects():
    frame = sample_replay().frames[0]
    first = frame.fresh_detections()
    second = frame.fresh_detections()

    assert first == second
    assert first[0] is not second[0]
    first[0].bbox = (0.0, 0.0, 1.0, 1.0)
    assert second[0].bbox == (10.0, 20.0, 40.0, 80.0)


def test_parser_accepts_canonical_minimum_without_research_extensions():
    text = "\n".join(
        [
            json.dumps(
                {
                    "type": "metadata",
                    "schema": DETECTION_REPLAY_SCHEMA,
                    "source_commit": "abc",
                    "video": "clip.mp4",
                    "video_sha256": None,
                    "detector": "yolo",
                    "model_sha256": None,
                    "provider": "DmlExecutionProvider",
                    "config": {},
                    "width": 1920,
                    "height": 1080,
                }
            ),
            json.dumps(
                {
                    "type": "frame",
                    "video": "clip.mp4",
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
                            "appearance": None,
                        }
                    ],
                }
            ),
        ]
    )
    replay = parse_detection_replay(text)
    assert replay.frames[0].detector_ran is True
    assert replay.frames[0].camera_motion == (0.0, 0.0)
    assert replay.frames[0].camera_transform is None


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (lambda rows: rows[0].update(schema="wrong"), "unsupported detection replay schema"),
        (lambda rows: rows[1].update(video="other.mp4"), "frame video must match metadata video"),
        (lambda rows: rows[1].update(frame=-1), "frame must be a non-negative integer"),
        (
            lambda rows: rows[1]["detections"][0].update(bbox=[10, 10, 5, 20]),
            "positive width and height",
        ),
        (
            lambda rows: rows[1]["detections"][0].update(score=1.5),
            "score must be in",
        ),
    ],
)
def test_invalid_replay_is_rejected(mutator, message):
    replay = sample_replay()
    rows = [replay.metadata.to_json(), replay.frames[0].to_json()]
    mutator(rows)
    text = "\n".join(json.dumps(row) for row in rows)
    with pytest.raises(ValueError, match=message):
        parse_detection_replay(text)


def test_frame_order_must_be_strictly_increasing():
    replay = sample_replay()
    rows = [
        replay.metadata.to_json(),
        replay.frames[0].to_json(),
        replay.frames[0].to_json(),
    ]
    with pytest.raises(ValueError, match="strictly increasing"):
        parse_detection_replay("\n".join(json.dumps(row) for row in rows))
