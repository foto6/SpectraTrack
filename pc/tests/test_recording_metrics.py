import json

from spectratrack.metrics import RollingProfiler
from spectratrack.recording import SessionRecorder
from spectratrack.types import Track


def test_rolling_profiler_summary():
    p = RollingProfiler(window=10)
    for value in (1.0, 2.0, 3.0, 4.0):
        p.add("detect", value)
    summary = p.summary()
    assert summary["detect_avg_ms"] == 2.5
    assert summary["detect_p95_ms"] == 4.0


def test_session_recorder_writes_parseable_jsonl(tmp_path):
    tr = Track(3, (1, 2, 10, 20), 0.9, 0, "person", confirmed=True)
    rec = SessionRecorder(tmp_path, {"test": True})
    rec.write_frame(
        7,
        12.5,
        [tr],
        selected_id=3,
        motion={"valid": True, "dx": 1.0},
        metrics={"detect_avg_ms": 5.0},
    )
    directory = rec.directory
    rec.close()

    meta = json.loads((directory / "meta.json").read_text(encoding="utf-8"))
    row = json.loads((directory / "frames.jsonl").read_text(encoding="utf-8").strip())
    assert meta["test"] is True
    assert row["frame"] == 7
    assert row["selected_id"] == 3
    assert row["tracks"][0]["id"] == 3
    assert row["motion"]["valid"] is True
