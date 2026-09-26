import json
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

import spectratrack.research.strict_enhancement_evidence as evidence
from spectratrack.research.apply_enhancement_alternates import apply_alternates
from spectratrack.research.strict_weak_manifest import build_manifest


def _write_prefusion(path):
    rows = [
        {
            "type": "metadata",
            "schema": "spectratrack-detection-prefusion-v1",
            "source_commit": "a1",
            "model_sha256": "model-sha",
            "video_sha256": "video-sha",
            "config": {"person_conf": 0.12},
        },
        {
            "type": "frame",
            "video": "golden/public/test/seq",
            "frame": 0,
            "timestamp_s": 0.0,
            "width": 64,
            "height": 64,
            "candidates": [
                {
                    "bbox": [10.0, 10.0, 20.0, 40.0],
                    "score": 0.20,
                    "class_id": 0,
                    "label": "person",
                    "source_kind": "tile",
                    "source_id": "tile:0",
                    "source_region": [0, 0, 64, 64],
                },
                {
                    "bbox": [10.0, 10.0, 20.0, 40.0],
                    "score": 0.34,
                    "class_id": 0,
                    "label": "person",
                    "source_kind": "tile",
                    "source_id": "tile:1",
                    "source_region": [0, 0, 64, 64],
                },
                {
                    "bbox": [30.0, 10.0, 40.0, 40.0],
                    "score": 0.18,
                    "class_id": 0,
                    "label": "person",
                    "source_kind": "tile",
                    "source_id": "tile:1",
                    "source_region": [0, 0, 64, 64],
                },
                {
                    "bbox": [10.0, 10.0, 20.0, 40.0],
                    "score": 0.30,
                    "class_id": 0,
                    "label": "person",
                    "source_kind": "full",
                    "source_id": "full",
                    "source_region": [0, 0, 64, 64],
                },
                {
                    "bbox": [45.0, 10.0, 55.0, 40.0],
                    "score": 0.50,
                    "class_id": 0,
                    "label": "person",
                    "source_kind": "tile",
                    "source_id": "tile:2",
                    "source_region": [0, 0, 64, 64],
                },
            ],
        },
        {"type": "summary", "frames": 1},
    ]
    path.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )


def _write_source_map(path):
    path.write_text(
        json.dumps(
            {
                "video": "golden/public/test/seq",
                "frame": 0,
                "source": "images/frame0.png",
                "objects": [],
            }
        )
        + "\n",
        encoding="utf-8",
    )


def test_strict_manifest_selects_highest_weak_tile_and_preserves_source(tmp_path):
    prefusion = tmp_path / "prefusion.jsonl"
    source_map = tmp_path / "gt.jsonl"
    output = tmp_path / "roi.jsonl"
    _write_prefusion(prefusion)
    _write_source_map(source_map)

    result = build_manifest(
        prefusion,
        output,
        source_map_path=source_map,
    )

    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert result["selected_frames"] == 1
    assert len(rows) == 2

    metadata, roi = rows
    assert metadata["selection"]["ground_truth_used_for_selection"] is False
    assert roi["source_kind"] == "tile"
    assert roi["source_id"] == "tile:1"
    assert roi["source"] == "images/frame0.png"
    assert roi["signals"] == {"weak_person": True}
    assert roi["trigger"]["score"] == pytest.approx(0.34)
    assert [item["score"] for item in roi["raw_support"]] == pytest.approx([0.34, 0.18])


def test_strict_manifest_does_not_use_full_or_strong_candidate_as_trigger(tmp_path):
    prefusion = tmp_path / "prefusion.jsonl"
    output = tmp_path / "roi.jsonl"
    _write_prefusion(prefusion)

    build_manifest(prefusion, output)

    roi = json.loads(output.read_text(encoding="utf-8").splitlines()[1])
    assert roi["source_id"] == "tile:1"
    assert all(0.12 <= item["score"] < 0.35 for item in roi["raw_support"])


def test_strict_evidence_reuses_frozen_raw_support_without_raw_onnx_call(
    tmp_path,
    monkeypatch,
):
    prefusion = tmp_path / "prefusion.jsonl"
    source_map = tmp_path / "gt.jsonl"
    roi_manifest = tmp_path / "roi.jsonl"
    _write_prefusion(prefusion)
    _write_source_map(source_map)
    build_manifest(prefusion, roi_manifest, source_map_path=source_map)

    source_root = tmp_path / "dataset"
    image_dir = source_root / "images"
    image_dir.mkdir(parents=True)
    frame = np.full((64, 64, 3), 10, dtype=np.uint8)
    assert cv2.imwrite(str(image_dir / "frame0.png"), frame)

    model = tmp_path / "model.onnx"
    model.write_bytes(b"fake-model")

    class FakeDetector:
        def __init__(self, *_args, **_kwargs):
            self.providers = ["FakeExecutionProvider"]
            self.input_w = 960
            self.input_h = 960
            self.class_thresholds = {}
            self.last_stage_ms = {}
            self.last_inference_calls = 0

        def _reset_policy_metrics(self):
            self.last_stage_ms = {}
            self.last_inference_calls = 0

        def _detect_once(self, _image, thresholds):
            assert thresholds["person"] == pytest.approx(0.12)
            self.last_inference_calls += 1
            self.last_stage_ms["inference"] = self.last_stage_ms.get("inference", 0.0) + 2.0
            from spectratrack.types import Detection

            return [Detection((10.0, 10.0, 20.0, 40.0), 0.55, 0, "person")]

    monkeypatch.setattr("spectratrack.detector.YoloOnnxDetector", FakeDetector)

    output = tmp_path / "alternates.jsonl"
    args = SimpleNamespace(
        roi_manifest=str(roi_manifest),
        source_root=str(source_root),
        model=str(model),
        operation="current_adaptive_cached",
        output=str(output),
        source_commit="a3",
        input_size=960,
        conf=0.35,
        nms_iou=0.45,
        cpu=True,
    )
    result = evidence.run_strict_evidence(args)

    assert result["additional_raw_onnx_calls"] == 0
    assert result["extra_enhancement_onnx_calls"] == 1
    assert result["accepted_alternate_measurements"] == 1

    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    alternate = next(row for row in rows if row["type"] == "alternate")
    assert alternate["source_kind"] == "tile"
    assert alternate["source_id"] == "tile:1"
    assert alternate["semantics"]["independent_evidence_increment"] == 0
    assert alternate["semantics"]["raw_corroborated"] is True
    assert alternate["raw_support"][0]["score"] == pytest.approx(0.34)


def test_strict_evidence_rejects_manifest_without_frozen_raw_support(tmp_path):
    manifest = tmp_path / "roi.jsonl"
    rows = [
        {
            "type": "metadata",
            "schema": "spectratrack-vnext-enhancement-roi-v1",
        },
        {
            "type": "roi",
            "video": "x",
            "frame": 0,
            "roi_id": "r0",
            "bbox": [0, 0, 10, 10],
            "signals": {"weak_person": True},
            "source_kind": "tile",
            "source_id": "tile:0",
            "source": "frame.png",
        },
    ]
    manifest.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )
    _metadata, records = evidence.load_roi_manifest(manifest)

    with pytest.raises(ValueError, match="frozen raw_support"):
        evidence._validate_strict_records(records)


def test_apply_alternates_replaces_same_source_measurement_without_new_evidence_source(
    tmp_path,
    monkeypatch,
):
    prefusion = tmp_path / "prefusion.jsonl"
    source_map = tmp_path / "gt.jsonl"
    roi_manifest = tmp_path / "roi.jsonl"
    _write_prefusion(prefusion)
    _write_source_map(source_map)
    build_manifest(prefusion, roi_manifest, source_map_path=source_map)

    source_root = tmp_path / "dataset"
    image_dir = source_root / "images"
    image_dir.mkdir(parents=True)
    frame = np.full((64, 64, 3), 10, dtype=np.uint8)
    assert cv2.imwrite(str(image_dir / "frame0.png"), frame)
    model = tmp_path / "model.onnx"
    model.write_bytes(b"fake-model")

    class FakeDetector:
        def __init__(self, *_args, **_kwargs):
            self.providers = ["FakeExecutionProvider"]
            self.input_w = 960
            self.input_h = 960
            self.class_thresholds = {}
            self.last_stage_ms = {}
            self.last_inference_calls = 0

        def _reset_policy_metrics(self):
            self.last_stage_ms = {}
            self.last_inference_calls = 0

        def _detect_once(self, _image, _thresholds):
            self.last_inference_calls += 1
            self.last_stage_ms["inference"] = self.last_stage_ms.get("inference", 0.0) + 2.0
            from spectratrack.types import Detection

            return [Detection((10.0, 10.0, 20.0, 40.0), 0.55, 0, "person")]

    monkeypatch.setattr("spectratrack.detector.YoloOnnxDetector", FakeDetector)
    evidence_path = tmp_path / "alternates.jsonl"
    evidence.run_strict_evidence(
        SimpleNamespace(
            roi_manifest=str(roi_manifest),
            source_root=str(source_root),
            model=str(model),
            operation="current_adaptive_cached",
            output=str(evidence_path),
            source_commit="a3",
            input_size=960,
            conf=0.35,
            nms_iou=0.45,
            cpu=True,
        )
    )

    augmented = tmp_path / "augmented-prefusion.jsonl"
    result = apply_alternates(prefusion, evidence_path, augmented)

    assert result["alternate_measurements_replaced"] == 1
    assert result["independent_sources_added"] == 0

    rows = [json.loads(line) for line in augmented.read_text(encoding="utf-8").splitlines()]
    frame_row = next(row for row in rows if row["type"] == "frame")
    tile1 = [
        candidate
        for candidate in frame_row["candidates"]
        if candidate["source_id"] == "tile:1"
    ]
    assert len(tile1) == 2
    assert sorted(candidate["score"] for candidate in tile1) == pytest.approx([0.18, 0.55])
    assert {candidate["source_id"] for candidate in frame_row["candidates"]} == {
        "tile:0",
        "tile:1",
        "tile:2",
        "full",
    }

    metadata = next(row for row in rows if row["type"] == "metadata")
    assert metadata["a3_enhancement"]["semantics"].startswith("same-source")
    summary = next(row for row in rows if row["type"] == "summary")
    assert summary["a3_enhancement"]["independent_sources_added"] == 0
