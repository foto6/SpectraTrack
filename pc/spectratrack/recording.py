from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .types import Track


class SessionRecorder:
    """Append-only JSONL telemetry for reproducible offline analysis."""

    def __init__(self, root: str | Path, metadata: dict[str, Any] | None = None) -> None:
        root = Path(root)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self.directory = root / f"session-{stamp}"
        suffix = 1
        while self.directory.exists():
            self.directory = root / f"session-{stamp}-{suffix}"
            suffix += 1
        self.directory.mkdir(parents=True, exist_ok=False)
        meta = dict(metadata or {})
        meta["created_utc"] = datetime.now(timezone.utc).isoformat()
        (self.directory / "meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")
        self._fp = (self.directory / "frames.jsonl").open("a", encoding="utf-8")

    def write_frame(
        self,
        frame_index: int,
        monotonic_s: float,
        tracks: list[Track],
        selected_id: int | None,
        motion: dict[str, Any] | None,
        metrics: dict[str, float],
        extra: dict[str, Any] | None = None,
    ) -> None:
        payload = {
            "frame": int(frame_index),
            "t": float(monotonic_s),
            "selected_id": selected_id,
            "motion": motion,
            "metrics": metrics,
            "tracks": [
                {
                    "id": t.track_id,
                    "class_id": t.class_id,
                    "label": t.label,
                    "score": t.score,
                    "bbox": [float(v) for v in t.bbox],
                    "vx": t.vx,
                    "vy": t.vy,
                    "age": t.age,
                    "hits": t.hits,
                    "missed": t.missed,
                    "confirmed": t.confirmed,
                    "predicted_only": t.predicted_only,
                    "association_score": t.association_score,
                }
                for t in tracks
            ],
        }
        if extra:
            payload["extra"] = extra
        self._fp.write(json.dumps(payload, separators=(",", ":")) + "\n")
        self._fp.flush()

    def close(self) -> None:
        if not self._fp.closed:
            self._fp.close()

    def __enter__(self) -> "SessionRecorder":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
