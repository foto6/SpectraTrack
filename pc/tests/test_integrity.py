from pathlib import Path

import pytest

from spectratrack.integrity import sha256_file, verify_sha256


def test_sha256_known_value(tmp_path: Path):
    p = tmp_path / "x.bin"
    p.write_bytes(b"abc")
    expected = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    assert sha256_file(p) == expected
    assert verify_sha256(p, expected) == expected
    assert verify_sha256(p, "sha256:" + expected) == expected


def test_sha256_mismatch(tmp_path: Path):
    p = tmp_path / "x.bin"
    p.write_bytes(b"abc")
    with pytest.raises(ValueError):
        verify_sha256(p, "0" * 64)
