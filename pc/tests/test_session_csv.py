import csv
import json
from pathlib import Path

from spectratrack.session_csv import export_csv


def test_session_csv_export(tmp_path: Path):
    src = tmp_path / "s.jsonl"
    dst = tmp_path / "tracks.csv"
    rows = [
        {"type":"frame","frame":1,"t":0.1,"fps":30,"selected_id":1,"tracks":[{
            "id":1,"label":"person","class_id":0,"score":0.9,
            "bbox":[1,2,11,22],"center":[6,12],"velocity_px_frame":[1,0],
            "age":2,"hits":2,"missed":0,"confirmed":True,"lifecycle":"TRACKED",
            "quality":0.8,"recoveries":0
        }]}
    ]
    src.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    assert export_csv(src, dst) == 1
    out = list(csv.DictReader(dst.open(encoding="utf-8")))
    assert out[0]["track_id"] == "1"
    assert out[0]["lifecycle"] == "TRACKED"
