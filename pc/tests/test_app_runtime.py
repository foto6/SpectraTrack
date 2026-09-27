import numpy as np

from spectratrack import app


class _Recorder:
    def __init__(self):
        self.events = []

    def event(self, name, **payload):
        self.events.append((name, payload))


def test_adaptive_people_recall_blocks_runtime_enhancement_toggle_and_keeps_raw_frame(monkeypatch, capsys):
    recorder = _Recorder()
    enhancement = False

    enhancement = app.handle_analysis_enhancement_toggle(
        enhancement,
        detector_mode="people-recall",
        people_recall_enhancement="adaptive",
        recorder=recorder,
    )

    assert enhancement is False
    assert recorder.events == []
    assert "adaptive people-recall requires raw corroboration" in capsys.readouterr().out

    raw_frame = np.zeros((8, 8, 3), dtype=np.uint8)

    def fail_if_enhanced(_frame):
        raise AssertionError("ordinary analysis enhancement must not run")

    monkeypatch.setattr(app, "enhance_visibility", fail_if_enhanced)
    assert app.prepare_analysis_frame(raw_frame, enhancement) is raw_frame


def test_runtime_enhancement_toggle_still_works_outside_adaptive_people_recall():
    recorder = _Recorder()

    enhancement = app.handle_analysis_enhancement_toggle(
        False,
        detector_mode="standard",
        people_recall_enhancement="off",
        recorder=recorder,
    )
    assert enhancement is True
    assert recorder.events == [("enhance", {"enabled": True})]

    enhancement = app.handle_analysis_enhancement_toggle(
        enhancement,
        detector_mode="standard",
        people_recall_enhancement="off",
        recorder=recorder,
    )
    assert enhancement is False
    assert recorder.events[-1] == ("enhance", {"enabled": False})


def test_runtime_enhancement_toggle_still_works_for_people_recall_when_adaptive_is_off():
    recorder = _Recorder()

    enhancement = app.handle_analysis_enhancement_toggle(
        False,
        detector_mode="people-recall",
        people_recall_enhancement="off",
        recorder=recorder,
    )

    assert enhancement is True
    assert recorder.events == [("enhance", {"enabled": True})]
