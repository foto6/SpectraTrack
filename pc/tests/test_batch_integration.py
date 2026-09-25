from pathlib import Path

import cv2
import numpy as np
import pytest

from spectratrack.batch import analyze_video
from spectratrack.types import Detection


class FakeDetector:
    def detect(self, _frame):
        return [Detection((90.0, 60.0, 190.0, 170.0), 0.92, 2, "car")]


def _write_test_video(path: Path, frame_count: int = 12) -> None:
    width, height = 320, 240
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        20.0,
        (width, height),
    )
    assert writer.isOpened()
    rng = np.random.default_rng(7)
    background = rng.integers(0, 60, size=(height, width, 3), dtype=np.uint8)
    for index in range(frame_count):
        frame = background.copy()
        cv2.rectangle(frame, (90, 60), (190, 170), (20, 30, 220), -1)
        cv2.line(frame, (95, 65), (185, 165), (245, 245, 245), 3)
        cv2.putText(frame, str(index), (110, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        writer.write(frame)
    writer.release()


def test_analyze_video_end_to_end_without_neural_model(tmp_path):
    video = tmp_path / "sample.avi"
    previews = tmp_path / "previews"
    _write_test_video(video)

    tracklets = analyze_video(
        video,
        FakeDetector(),
        detect_every=1,
        class_filter=set(),
        min_observations=3,
        preview_dir=previews,
        video_id="nested/sample.avi",
        progress_every=0,
    )

    assert len(tracklets) == 1
    tracklet = tracklets[0]
    assert tracklet.video == "nested/sample.avi"
    assert tracklet.label == "car"
    assert tracklet.observations >= 3
    assert tracklet.descriptor is not None
    assert len(tracklet.gallery) >= 1
    assert tracklet.preview_path is not None
    assert Path(tracklet.preview_path).is_file()



class FakeRecallDetector:
    def detect(self, _frame):
        raise AssertionError("standard detector path must not be used in people-recall mode")

    def detect_people_recall(self, _frame, **_kwargs):
        return [Detection((90.0, 60.0, 190.0, 170.0), 0.92, 0, "person")]


def test_analyze_video_dispatches_people_recall_mode(tmp_path):
    video = tmp_path / "recall.avi"
    _write_test_video(video)

    tracklets = analyze_video(
        video,
        FakeRecallDetector(),
        detect_every=1,
        class_filter=set(),
        min_observations=3,
        progress_every=0,
        detector_mode="people-recall",
        person_conf=0.12,
        person_tile_size=640,
        person_tile_overlap=0.2,
        person_merge_iou=0.55,
    )

    assert len(tracklets) == 1
    assert tracklets[0].label == "person"


def test_analyze_video_rejects_unknown_detector_mode(tmp_path):
    with pytest.raises(ValueError, match="detector_mode"):
        analyze_video(
            tmp_path / "unused.avi",
            FakeDetector(),
            detect_every=1,
            class_filter=set(),
            min_observations=1,
            progress_every=0,
            detector_mode="typo",
        )
