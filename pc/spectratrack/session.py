from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import json
import time
from typing import Iterable

from .types import Track


class SessionRecorder:
    def __init__(self, path: str | Path, metadata: dict | None = None) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._f = self.path.open("w", encoding="utf-8")
        self.started_unix = time.time()
        self._write({"type": "header", "version": 1, "started_unix": self.started_unix, "metadata": metadata or {}})

    def _write(self, payload: dict) -> None:
        self._f.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n")
        self._f.flush()

    def frame(
        self,
        frame_index: int,
        tracks: Iterable[Track],
        selected_id: int | None,
        fps: float,
        timings_ms: dict[str, float] | None = None,
        camera_motion: tuple[float, float, float] | None = None,
    ) -> None:
        packed = []
        for tr in tracks:
            x1, y1, x2, y2 = tr.bbox
            cx, cy = tr.center
            packed.append({
                "id": tr.track_id,
                "class_id": tr.class_id,
                "label": tr.label,
                "score": round(float(tr.score), 6),
                "bbox": [round(v, 3) for v in (x1, y1, x2, y2)],
                "center": [round(cx, 3), round(cy, 3)],
                "velocity_px_frame": [round(tr.vx, 3), round(tr.vy, 3)],
                "age": tr.age,
                "hits": tr.hits,
                "missed": tr.missed,
                "confirmed": tr.confirmed,
                "lifecycle": tr.lifecycle,
                "quality": round(tr.quality, 4),
                "recoveries": tr.recoveries,
            })
        self._write({
            "type": "frame",
            "frame": int(frame_index),
            "t": round(time.time() - self.started_unix, 6),
            "fps": round(float(fps), 3),
            "selected_id": selected_id,
            "camera_motion": list(camera_motion) if camera_motion is not None else None,
            "timings_ms": timings_ms or {},
            "tracks": packed,
        })

    def event(self, name: str, **data) -> None:
        self._write({"type": "event", "t": round(time.time() - self.started_unix, 6), "name": name, "data": data})

    def close(self) -> None:
        if not self._f.closed:
            self._write({"type": "footer", "duration_s": round(time.time() - self.started_unix, 6)})
            self._f.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
