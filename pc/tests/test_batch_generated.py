from spectratrack.batch import discover_videos


def test_generated_only_folder_is_empty_without_include_derived(tmp_path):
    (tmp_path / "clip_SpectraTrack.mp4").write_bytes(b"not-a-video")
    assert discover_videos(tmp_path) == []
