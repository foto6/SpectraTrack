from pathlib import Path

import pytest

from spectratrack.batch import discover_videos


def test_discover_videos_filters_and_sorts(tmp_path):
    (tmp_path / "b.MP4").write_bytes(b"x")
    (tmp_path / "a.mkv").write_bytes(b"x")
    (tmp_path / "note.txt").write_text("x", encoding="utf-8")
    names = [p.name for p in discover_videos(tmp_path)]
    assert names == ["a.mkv", "b.MP4"]


def test_discover_videos_recursive(tmp_path):
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "clip.mov").write_bytes(b"x")
    assert discover_videos(tmp_path) == []
    assert [p.name for p in discover_videos(tmp_path, recursive=True)] == ["clip.mov"]


def test_discover_videos_requires_directory(tmp_path):
    with pytest.raises(FileNotFoundError):
        discover_videos(Path(tmp_path / "missing"))
