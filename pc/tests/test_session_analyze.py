import json

from spectratrack.session_analyze import analyze_session


def test_session_analysis_summary(tmp_path):
    rows = [
        {
            "frame": 1, "t": 10.0,
            "metrics": {"detect_avg_ms": 12.0},
            "tracks": [{"id": 1, "label": "person", "score": 0.8, "vx": 1.0, "vy": 0.0, "missed": 0, "confirmed": True}],
        },
        {
            "frame": 2, "t": 10.1,
            "metrics": {"detect_avg_ms": 14.0},
            "tracks": [
                {"id": 1, "label": "person", "score": 0.9, "vx": 2.0, "vy": 0.0, "missed": 0, "confirmed": True},
                {"id": 2, "label": "car", "score": 0.7, "vx": 0.0, "vy": 1.0, "missed": 1, "confirmed": False},
            ],
        },
    ]
    p = tmp_path / "frames.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    result = analyze_session(p)
    assert result["frames"] == 2
    assert result["tracks"] == 2
    assert result["max_concurrent_tracks"] == 2
    assert abs(result["duration_s"] - 0.1) < 1e-9
    assert result["per_track"]["1"]["confirmed_seen"] is True
