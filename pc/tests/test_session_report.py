import json
from pathlib import Path

from spectratrack.session_report import summarize


def test_report_summary(tmp_path: Path):
    path = tmp_path / "s.jsonl"
    rows = [
        {"type":"header","metadata":{"model":"x"}},
        {"type":"frame","t":0.1,"fps":20.0,"timings_ms":{"detect":10.0},"tracks":[{"id":1,"label":"person","missed":0}]},
        {"type":"frame","t":0.2,"fps":30.0,"timings_ms":{"detect":20.0},"tracks":[{"id":1,"label":"person","missed":1}]},
        {"type":"footer","duration_s":0.2},
    ]
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    report = summarize(path)
    assert report["session"]["frames"] == 2
    assert report["fps"]["avg"] == 25.0
    assert report["timings_ms"]["detect"]["avg"] == 15.0
    assert report["track_count"] == 1
    assert report["tracks"][0]["frames_predicted"] == 1
