from __future__ import annotations

import hashlib
from pathlib import Path


def sha256_file(path: str | Path) -> str:
    p = Path(path)
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def normalize_sha256(value: str) -> str:
    value = value.strip().lower()
    if value.startswith("sha256:"):
        value = value.split(":", 1)[1]
    if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError("Expected a 64-character SHA-256 hex digest")
    return value


def verify_sha256(path: str | Path, expected: str) -> str:
    expected_norm = normalize_sha256(expected)
    actual = sha256_file(path)
    if actual != expected_norm:
        raise ValueError(f"SHA-256 mismatch for {path}: expected {expected_norm}, got {actual}")
    return actual
