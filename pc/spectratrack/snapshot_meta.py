from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .calibration import CameraCalibration
from .integrity import sha256_file
from .types import Track


def snapshot_metadata(
    image_path: str | Path,
    track: Track,
    frame_index: int,
    model_sha256: str,
    view_mode: str,
    calibration: CameraCalibration | None,
    source: str,
    classification: str,
    derived_from: str | Path | None = None,
) -> dict[str, Any]:
    image = Path(image_path)
    cx, cy = track.center
    payload: dict[str, Any] = {
        "schema": 1,
        "classification": classification,
        "image": image.name,
        "image_sha256": sha256_file(image),
        "model_sha256": model_sha256,
        "source": source,
        "frame": int(frame_index),
        "view_mode": view_mode,
        "track": {
            "id": track.track_id,
            "label": track.label,
            "class_id": track.class_id,
            "score": round(float(track.score), 6),
            "bbox": [round(float(x), 3) for x in track.bbox],
            "center": [round(float(cx), 3), round(float(cy), 3)],
            "lifecycle": track.lifecycle,
            "quality": round(track.quality, 4),
            "recoveries": track.recoveries,
        },
    }
    if calibration is not None:
        payload["calibration"] = {
            "name": calibration.name,
            "width": calibration.width,
            "height": calibration.height,
            "hfov_deg": calibration.hfov_deg,
            "vfov_deg": calibration.vfov_deg,
        }
    if derived_from is not None:
        src = Path(derived_from)
        payload["derived_from"] = {
            "image": src.name,
            "image_sha256": sha256_file(src),
        }
    return payload


def write_snapshot_metadata(
    image_path: str | Path,
    track: Track,
    frame_index: int,
    model_sha256: str,
    view_mode: str,
    calibration: CameraCalibration | None,
    source: str,
    classification: str,
    derived_from: str | Path | None = None,
) -> Path:
    image = Path(image_path)
    payload = snapshot_metadata(
        image,
        track,
        frame_index,
        model_sha256,
        view_mode,
        calibration,
        source,
        classification,
        derived_from,
    )
    out = image.with_suffix(image.suffix + ".json")
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return out
