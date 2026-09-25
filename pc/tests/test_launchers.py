from pathlib import Path


def test_folder_launcher_quotes_review_path_safely():
    text = Path("ANALYZE_VIDEO_FOLDER.bat").read_text(encoding="utf-8")
    assert '--review "%INPUT%\\spectratrack_review.json"' in text
    assert 'set "REVIEW=--review "' not in text


def test_single_video_launcher_exists():
    assert Path("PROCESS_VIDEO.bat").is_file()
