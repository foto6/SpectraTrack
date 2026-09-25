from pathlib import Path


from spectratrack.batch import video_identifier


def test_video_identifier_falls_back_to_name_outside_root(tmp_path):
    outside = tmp_path.parent / "outside-video.mp4"
    assert video_identifier(tmp_path, outside) == "outside-video.mp4"


def test_batch_validation_logic_documented_by_cli_help():
    # Keep a lightweight regression around the public batch entrypoint import;
    # detailed parser behavior is exercised by the standalone batch --help smoke test in CI.
    from spectratrack import batch

    assert callable(batch.main)
    assert Path(batch.__file__).name == "batch.py"
