import hashlib

import pytest

from spectratrack.model_security import normalize_sha256, sha256_file, verify_sha256


def test_sha256_roundtrip(tmp_path):
    p = tmp_path / "model.onnx"
    p.write_bytes(b"example-model-bytes")
    expected = hashlib.sha256(b"example-model-bytes").hexdigest()
    assert sha256_file(p) == expected
    assert verify_sha256(p, expected) == expected
    assert verify_sha256(p, "sha256:" + expected) == expected


def test_bad_digest_rejected(tmp_path):
    p = tmp_path / "model.onnx"
    p.write_bytes(b"x")
    with pytest.raises(ValueError):
        verify_sha256(p, "0" * 64)


def test_invalid_digest_format_rejected():
    with pytest.raises(ValueError):
        normalize_sha256("not-a-hash")
