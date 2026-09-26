import json
import cv2
import numpy as np

from spectratrack.qa_benchmark import GroundTruthFrame, GroundTruthObject
from spectratrack.research.vnext_detection_corpus import (
    attach_ground_truth,
    collect_corpus_video_frames,
    evaluate_methods,
    export_canonical_replays,
)
from spectratrack.research.vnext_detection_fusion import (
    FusionCandidate,
    FusionConfig,
    ResearchFrame,
    write_prefusion_dump,
)
from spectratrack.types import Detection


class _FakeDetector:
    def __init__(self):
        self.labels = ["person"]
        self.class_thresholds = {}
        self.last_inference_calls = 0
        self.last_stage_ms = {}

    def _reset_policy_metrics(self):
        self.last_inference_calls = 0
        self.last_stage_ms = {}

    def _detect_once(self, frame, thresholds):
        self.last_inference_calls += 1
        self.last_stage_ms["inference_ms"] = self.last_stage_ms.get("inference_ms", 0.0) + 1.0
        return [Detection((1.0, 1.0, 8.0, 12.0), 0.40, 0, "person")]


def _annotation(video: str = "clip", frame: int = 0) -> GroundTruthFrame:
    return GroundTruthFrame(
        video,
        frame,
        ("public",),
        (GroundTruthObject("p1", "person", (1.0, 1.0, 8.0, 12.0)),),
    )


def test_collect_corpus_video_frames_resolves_mot_image_sequence(tmp_path):
    image_dir = tmp_path / "MOT17-04-FRCNN" / "img1"
    image_dir.mkdir(parents=True)
    assert cv2.imwrite(str(image_dir / "000001.jpg"), np.zeros((20, 20, 3), dtype=np.uint8))

    annotation = _annotation()
    records = {
        ("clip", 0): {
            "video": "clip",
            "frame": 0,
            "source": "MOT17-04-FRCNN/img1",
            "source_frame": 1,
            "source_fps": 25.0,
        }
    }
    frames, summary = collect_corpus_video_frames(
        _FakeDetector(),
        [annotation],
        records,
        video_root=tmp_path,
        person_threshold=0.12,
        tile_size=20,
        tile_overlap=0.0,
    )

    assert len(frames) == 1
    assert frames[0].ground_truth == annotation.objects
    assert frames[0].tags == annotation.tags
    assert frames[0].timestamp_s == 0.0
    assert len(frames[0].candidates) == 2
    assert summary["policy_runs"] == 1
    assert summary["inference_calls"] == 2
    assert summary["source_modes"] == ["image_sequence"]


def test_collect_corpus_video_frames_resolves_direct_crowdhuman_image(tmp_path):
    image_dir = tmp_path / "Images"
    image_dir.mkdir(parents=True)
    assert cv2.imwrite(str(image_dir / "crowd.jpg"), np.zeros((20, 20, 3), dtype=np.uint8))

    annotation = _annotation("crowd.jpg")
    records = {
        ("crowd.jpg", 0): {
            "video": "crowd.jpg",
            "frame": 0,
            "source": "Images/crowd.jpg",
        }
    }
    frames, summary = collect_corpus_video_frames(
        _FakeDetector(),
        [annotation],
        records,
        video_root=tmp_path,
        person_threshold=0.12,
        tile_size=20,
        tile_overlap=0.0,
    )

    assert len(frames) == 1
    assert summary["source_modes"] == ["image"]


def test_attach_ground_truth_rehydrates_cached_prefusion_frames():
    cached = ResearchFrame(
        "clip",
        3,
        0.12,
        640,
        360,
        (FusionCandidate((10.0, 10.0, 30.0, 50.0), 0.4, 0, "person", "full", "full"),),
    )
    annotation = GroundTruthFrame(
        "clip",
        3,
        ("held_out",),
        (GroundTruthObject("p1", "person", (10.0, 10.0, 30.0, 50.0)),),
    )

    result = attach_ground_truth([cached], {("clip", 3): annotation})

    assert result[0].ground_truth == annotation.objects
    assert result[0].tags == ("held_out",)


def test_evaluate_methods_reports_aggregate_and_per_video():
    weak_single = ResearchFrame(
        "a",
        0,
        0.0,
        100,
        100,
        (FusionCandidate((10.0, 10.0, 30.0, 60.0), 0.15, 0, "person", "tile", "tile:0"),),
        (GroundTruthObject("p1", "person", (10.0, 10.0, 30.0, 60.0)),),
    )
    strong = ResearchFrame(
        "b",
        0,
        0.0,
        100,
        100,
        (FusionCandidate((40.0, 10.0, 60.0, 60.0), 0.40, 0, "person", "tile", "tile:0"),),
        (GroundTruthObject("p2", "person", (40.0, 10.0, 60.0, 60.0)),),
    )

    aggregate, per_video = evaluate_methods(
        [weak_single, strong],
        methods=["hard-nms", "evidence-aware"],
        config=FusionConfig(),
        match_iou=0.5,
        per_video=True,
    )

    assert aggregate["hard-nms"]["recall"] == 1.0
    assert aggregate["evidence-aware"]["recall"] == 0.5
    assert per_video["a"]["evidence-aware"]["recall"] == 0.0
    assert per_video["b"]["evidence-aware"]["recall"] == 1.0

def test_export_canonical_replays_uses_prefusion_without_inference(tmp_path):
    prefusion_dir = tmp_path / "prefusion"
    replay_dir = tmp_path / "replays"
    prefusion_dir.mkdir()
    frame = ResearchFrame(
        "clip",
        0,
        0.0,
        100,
        80,
        (FusionCandidate((10.0, 10.0, 30.0, 60.0), 0.40, 0, "person", "full", "full"),),
    )
    write_prefusion_dump(
        prefusion_dir / "clip.jsonl",
        {
            "source_commit": "a" * 40,
            "video": "clip",
            "video_sha256": "b" * 64,
            "detector": "current-yolo-onnx-prefusion",
            "model_sha256": "c" * 64,
            "provider": "DmlExecutionProvider,CPUExecutionProvider",
            "config": {"person_conf": 0.12},
            "width": 100,
            "height": 80,
        },
        [frame],
        {"policy_runs": 1, "inference_calls": 9, "wall_time_s": 1.0},
    )

    outputs = export_canonical_replays(
        prefusion_dir=prefusion_dir,
        replay_dir=replay_dir,
        videos=["clip"],
        methods=["hard-nms", "evidence-aware"],
        config=FusionConfig(),
    )

    assert set(outputs["clip"]) == {"hard-nms", "evidence-aware"}
    for method, path_text in outputs["clip"].items():
        path = replay_dir / f"clip.{method}.replay.jsonl"
        assert path_text == str(path)
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        assert rows[0]["schema"] == "spectratrack-detection-replay-v1"
        assert rows[0]["config"]["cross_pass_fusion"]["method"] == method
        assert rows[1]["detections"][0]["appearance"] is None

