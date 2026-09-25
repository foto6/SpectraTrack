from types import SimpleNamespace

import numpy as np
import pytest

from spectratrack.research.vnext_detector_backend_lab import (
    BACKEND_SPECS,
    RtDetrv2OnnxAdapter,
    _half_pixel_resize_rgb,
    _rf_preprocess,
    decode_rfdetr_person,
)


def test_backend_specs_cover_required_vnext_candidates():
    assert set(BACKEND_SPECS) == {"current-yolo", "rf-detr", "rt-detrv2"}
    assert "UNVERIFIED" in BACKEND_SPECS["rf-detr"].directml_status
    assert "UNVERIFIED" in BACKEND_SPECS["rt-detrv2"].directml_status


def test_half_pixel_resize_preserves_constant_rgb_and_shape():
    image = np.full((3, 5, 3), 42.0, dtype=np.float32)
    resized = _half_pixel_resize_rgb(image, 7, 9)

    assert resized.shape == (7, 9, 3)
    assert resized.dtype == np.float32
    assert np.allclose(resized, 42.0)


def test_rf_preprocess_is_nchw_imagenet_normalized():
    frame = np.zeros((2, 2, 3), dtype=np.uint8)
    blob = _rf_preprocess(frame, 4, 4)

    assert blob.shape == (1, 3, 4, 4)
    assert blob.dtype == np.float32
    expected = -np.asarray([0.485, 0.456, 0.406]) / np.asarray([0.229, 0.224, 0.225])
    assert np.allclose(blob[0, :, 0, 0], expected, atol=1e-5)


def test_decode_rfdetr_person_uses_sigmoid_and_normalized_cxcywh():
    boxes = np.asarray([[0.5, 0.5, 1.0, 1.0]], dtype=np.float32)
    logits = np.asarray([[8.0, -8.0]], dtype=np.float32)

    detections = decode_rfdetr_person(
        boxes,
        logits,
        width=100,
        height=50,
        threshold=0.3,
        person_class_id=0,
        background_class_id=1,
    )

    assert len(detections) == 1
    assert detections[0].label == "person"
    assert detections[0].bbox == pytest.approx((0.0, 0.0, 100.0, 50.0))
    assert detections[0].score > 0.99


def test_decode_rfdetr_rejects_person_as_background():
    with pytest.raises(ValueError, match="background"):
        decode_rfdetr_person(
            np.asarray([[0.5, 0.5, 1.0, 1.0]], dtype=np.float32),
            np.asarray([[8.0, -8.0]], dtype=np.float32),
            width=100,
            height=50,
            threshold=0.3,
            person_class_id=0,
            background_class_id=0,
        )


class _FakeRtSession:
    def __init__(self):
        self.feed = None

    def get_outputs(self):
        return [
            SimpleNamespace(name="labels", shape=[1, 2]),
            SimpleNamespace(name="boxes", shape=[1, 2, 4]),
            SimpleNamespace(name="scores", shape=[1, 2]),
        ]

    def get_providers(self):
        return ["CPUExecutionProvider"]

    def run(self, _names, feed):
        self.feed = feed
        return [
            np.asarray([[0, 2]], dtype=np.int64),
            np.asarray([[[1.0, 2.0, 20.0, 40.0], [5.0, 5.0, 15.0, 15.0]]], dtype=np.float32),
            np.asarray([[0.8, 0.9]], dtype=np.float32),
        ]


def test_rtdetrv2_adapter_feeds_original_size_and_filters_person():
    adapter = RtDetrv2OnnxAdapter.__new__(RtDetrv2OnnxAdapter)
    adapter.model_path = "fake.onnx"
    adapter.threshold = 0.3
    adapter.person_class_id = 0
    adapter.input_h = 64
    adapter.input_w = 64
    adapter.session = _FakeRtSession()

    frame = np.zeros((50, 100, 3), dtype=np.uint8)
    result = adapter.detect(frame)

    assert len(result.detections) == 1
    assert result.detections[0].bbox == (1.0, 2.0, 20.0, 40.0)
    assert adapter.session.feed["images"].shape == (1, 3, 64, 64)
    assert adapter.session.feed["orig_target_sizes"].tolist() == [[100, 50]]
    assert result.inference_calls == 1
