from spectratrack.capture import CaptureConfig, RobustCapture


def test_capture_config_defaults():
    cfg = CaptureConfig()
    assert cfg.backend == "auto"
    assert cfg.buffer_size == 1
    assert cfg.reconnect_attempts == 3


def test_nonexistent_file_fails_without_reconnect(tmp_path):
    source = str(tmp_path / "missing-video.mp4")
    cap = RobustCapture(source, CaptureConfig(reconnect_attempts=5))
    try:
        assert not cap.is_opened()
        result = cap.read()
        assert not result.ok
        assert cap.reconnects == 0
    finally:
        cap.release()


def test_video_file_ignores_camera_backend(monkeypatch, tmp_path):
    calls = []

    class FakeCapture:
        def __init__(self, *args):
            calls.append(args)

        def isOpened(self):
            return True

        def set(self, *_args):
            return True

        def release(self):
            return None

    monkeypatch.setattr("spectratrack.capture.cv2.VideoCapture", FakeCapture)
    source = str(tmp_path / "clip.mp4")
    cap = RobustCapture(source, CaptureConfig(backend="dshow"))
    try:
        assert cap.is_opened()
        assert calls == [(source,)]
    finally:
        cap.release()
