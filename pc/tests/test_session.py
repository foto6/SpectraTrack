import json
from pathlib import Path

from spectratrack.session import SessionRecorder
from spectratrack.types import Track


def test_session_jsonl(tmp_path: Path):
    path = tmp_path / "session.jsonl"
    tr = Track(1, (1.0, 2.0, 11.0, 22.0), 0.9, 0, "person", confirmed=True)
    with SessionRecorder(path, {"model": "x.onnx"}) as rec:
        rec.event("hello", answer=42)
        rec.frame(
            1,
            [tr],
            1,
            30.0,
            {"detect": 5.2},
            (1.0, 2.0, 0.01),
            {"valid": True, "inliers": 12, "inlier_ratio": 0.8},
        )

    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert [r["type"] for r in rows] == ["header", "event", "frame", "footer"]
    assert rows[2]["tracks"][0]["id"] == 1
    assert rows[2]["selected_id"] == 1
    assert rows[2]["timings_ms"]["detect"] == 5.2
    assert rows[2]["selected_refine"]["valid"] is True
    assert rows[2]["selected_refine"]["inliers"] == 12
