import json

import numpy as np

from spectratrack.replay import load_session_rows, render_telemetry


def test_load_session_rows_and_render(tmp_path):
    row = {
        "frame": 1,
        "selected_id": 3,
        "motion": {"dx": 2.0, "dy": -1.0},
        "extra": {"detector_ran": True, "detector_interval": 2},
        "tracks": [
            {
                "id": 3,
                "label": "person",
                "score": 0.9,
                "bbox": [20, 30, 80, 100],
                "confirmed": True,
                "missed": 0,
                "predicted_only": False,
            }
        ],
    }
    p = tmp_path / "frames.jsonl"
    p.write_text(json.dumps(row) + "\n", encoding="utf-8")
    rows = load_session_rows(p)
    assert rows[1]["selected_id"] == 3

    frame = np.zeros((140, 180, 3), dtype=np.uint8)
    out = render_telemetry(frame, rows[1])
    assert out.shape == frame.shape
    assert np.any(out != frame)
