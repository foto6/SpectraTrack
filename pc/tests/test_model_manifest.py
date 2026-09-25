from pathlib import Path

from spectratrack.integrity import sha256_file
from spectratrack.model_manifest import ModelManifest


def test_manifest_roundtrip_and_verify(tmp_path: Path):
    model = tmp_path / "model.onnx"
    model.write_bytes(b"fake-model-for-hash-test")
    manifest = ModelManifest("test", sha256_file(model), input_size=640, source="unit-test")
    path = tmp_path / "manifest.json"
    manifest.save(path)
    loaded = ModelManifest.load(path)
    assert loaded.name == "test"
    assert loaded.input_size == 640
    assert loaded.verify(model) == sha256_file(model)
