from __future__ import annotations

import hashlib
from pathlib import Path


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    p = Path(path)
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_sha256(path: str | Path, expected: str) -> str:
    actual = sha256_file(path)
    normalized = expected.lower().strip().removeprefix("sha256:")
    if actual.lower() != normalized:
        raise ValueError(f"SHA-256 mismatch for {path}: expected {normalized}, got {actual}")
    return actual
