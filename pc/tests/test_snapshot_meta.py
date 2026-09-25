import json
from pathlib import Path

from spectratrack.snapshot_meta import write_snapshot_metadata
from spectratrack.types import Track


def test_snapshot_sidecar_contains_hash(tmp_path: Path):
    image = tmp_path / "crop.png"
    image.write_bytes(b"hashable-test-bytes")
    tr = Track(
        7, (1, 2, 11, 22), 0.9, 0, "person",
        confirmed=True, last_detection_score=0.9
    )
    sidecar = write_snapshot_metadata(
        image, tr, 12, "a" * 64, "normal",
        None, "video.mp4", "SOURCE_CROP"
    )
    data = json.loads(sidecar.read_text(encoding="utf-8"))
    assert data["classification"] == "SOURCE_CROP"
    assert data["track"]["id"] == 7
    assert len(data["image_sha256"]) == 64


def test_ai_sidecar_links_source(tmp_path: Path):
    source = tmp_path / "source.png"
    enhanced = tmp_path / "enhanced.png"
    source.write_bytes(b"source")
    enhanced.write_bytes(b"enhanced")
    tr = Track(
        1, (0, 0, 10, 10), 0.8, 0, "obj",
        confirmed=True, last_detection_score=0.8
    )
    sidecar = write_snapshot_metadata(
        enhanced, tr, 3, "b" * 64, "clarity",
        None, "cam0", "AI_ENHANCED", source
    )
    data = json.loads(sidecar.read_text(encoding="utf-8"))
    assert data["classification"] == "AI_ENHANCED"
    assert data["derived_from"]["image"] == "source.png"
    assert len(data["derived_from"]["image_sha256"]) == 64
