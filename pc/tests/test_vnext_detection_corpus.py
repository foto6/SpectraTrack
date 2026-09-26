import json
from types import SimpleNamespace
import cv2
import numpy as np

from spectratrack.qa_benchmark import GroundTruthFrame, GroundTruthObject
import spectratrack.research.vnext_detection_corpus as corpus_module
from spectratrack.research.vnext_detection_corpus import (
    attach_ground_truth,
    collect_corpus_video_frames,
    _per_video_performance,
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


def test_per_video_performance_distinguishes_new_and_reused_prefusion():
    summary = {
        "policy_runs": 3,
        "inference_calls": 27,
        "wall_time_s": 4.5,
        "stage_ms": {"inference": 4000.0},
        "source_modes": ["image_sequence"],
    }

    fresh = _per_video_performance(summary, reused=False)
    cached = _per_video_performance(summary, reused=True)

    assert fresh["inference_calls"] == 27
    assert fresh["new_inference_calls"] == 27
    assert fresh["reused_inference_calls"] == 0
    assert fresh["new_detector_wall_s"] == 4.5
    assert fresh["evidence_source"] == "new_inference"

    assert cached["inference_calls"] == 27
    assert cached["new_inference_calls"] == 0
    assert cached["reused_inference_calls"] == 27
    assert cached["reused_detector_wall_s"] == 4.5
    assert cached["evidence_source"] == "reused_prefusion"


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

def test_resume_marks_new_prefusion_as_fresh_then_reuses_on_second_run(tmp_path, monkeypatch):
    model = tmp_path / "model.onnx"
    model.write_bytes(b"model")
    gt = tmp_path / "gt.jsonl"
    gt.write_text("{}\n", encoding="utf-8")
    prefusion_dir = tmp_path / "prefusion"
    annotation = _annotation("clip", 0)
    calls = {"collect": 0}

    class FakeRunnerDetector:
        def __init__(self, *_args, **_kwargs):
            self.providers = ["CPUExecutionProvider"]
            self.input_w = 960
            self.input_h = 960

    def fake_collect(*_args, **_kwargs):
        calls["collect"] += 1
        frame = ResearchFrame(
            "clip",
            0,
            0.0,
            100,
            80,
            (FusionCandidate((1.0, 1.0, 8.0, 12.0), 0.4, 0, "person", "full", "full"),),
            annotation.objects,
            annotation.tags,
        )
        return [frame], {
            "policy_runs": 1,
            "inference_calls": 3,
            "wall_time_s": 2.5,
            "stage_ms": {"inference": 2000.0},
            "source_modes": ["image_sequence"],
        }

    monkeypatch.setattr(corpus_module, "YoloOnnxDetector", FakeRunnerDetector)
    monkeypatch.setattr(corpus_module, "load_ground_truth", lambda _path: [annotation])
    monkeypatch.setattr(corpus_module, "_load_benchmark_source_records", lambda _path: {})
    monkeypatch.setattr(corpus_module, "ground_truth_sha256", lambda _path: "g" * 64)
    monkeypatch.setattr(corpus_module, "collect_corpus_video_frames", fake_collect)

    args = SimpleNamespace(
        model=str(model),
        ground_truth=str(gt),
        video_root=str(tmp_path),
        output=str(tmp_path / "unused.json"),
        source_commit="a" * 40,
        corpus_revision="mot17-public-r1",
        include_video=[],
        prefusion_dir=str(prefusion_dir),
        replay_dir=None,
        resume=True,
        per_video=True,
        frame_limit_per_video=0,
        input_size=960,
        conf=0.35,
        decoder_iou=0.45,
        person_conf=0.12,
        tile_size=640,
        tile_overlap=0.20,
        match_iou=0.50,
        fusion_iou=0.55,
        center_ratio=0.20,
        size_ratio=1.80,
        score_power=1.0,
        full_weight=1.0,
        tile_weight=1.0,
        evidence_weak_score=0.12,
        evidence_solo_score=0.20,
        evidence_strong_score=0.35,
        evidence_min_sources=2,
        method=["hard-nms"],
        cpu=True,
    )

    first = corpus_module.run_corpus_benchmark(args)
    assert calls["collect"] == 1
    assert first["performance"]["per_video"]["clip"]["evidence_source"] == "new_inference"
    assert first["performance"]["new_inference_calls"] == 3
    assert first["performance"]["reused_inference_calls"] == 0
    assert first["performance"]["new_detector_wall_s"] == 2.5

    second = corpus_module.run_corpus_benchmark(args)
    assert calls["collect"] == 1
    assert second["performance"]["per_video"]["clip"]["evidence_source"] == "reused_prefusion"
    assert second["performance"]["new_inference_calls"] == 0
    assert second["performance"]["reused_inference_calls"] == 3
    assert second["performance"]["reused_detector_wall_s"] == 2.5


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

