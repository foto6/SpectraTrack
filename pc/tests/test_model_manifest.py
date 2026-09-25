import hashlib

import pytest

from spectratrack.model_manifest import ModelManifest


def test_manifest_roundtrip_and_verify(tmp_path):
    model = tmp_path / "detector.onnx"
    model.write_bytes(b"known-model")
    digest = hashlib.sha256(b"known-model").hexdigest()
    manifest = ModelManifest(
        name="test-model",
        sha256=digest,
        source="unit-test",
        license="test-only",
        input_size=640,
        output_layout="xywh + class scores",
    )
    path = tmp_path / "manifest.json"
    manifest.save(path)

    loaded = ModelManifest.load(path)
    assert loaded.name == "test-model"
    assert loaded.input_size == 640
    assert loaded.verify(model) == digest


def test_manifest_detects_changed_model(tmp_path):
    model = tmp_path / "detector.onnx"
    model.write_bytes(b"first")
    digest = hashlib.sha256(b"first").hexdigest()
    manifest = ModelManifest("m", digest)
    model.write_bytes(b"changed")
    with pytest.raises(ValueError):
        manifest.verify(model)


def test_manifest_rejects_bad_input_size():
    with pytest.raises(ValueError):
        ModelManifest("m", "0" * 64, input_size=0)
