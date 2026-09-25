from __future__ import annotations

import json
from pathlib import Path
from typing import Any


_ALLOWED: dict[str, type | tuple[type, ...]] = {
    "camera_width": int,
    "camera_height": int,
    "camera_fps": (int, float),
    "camera_backend": str,
    "reconnect_attempts": int,
    "reconnect_delay_ms": int,
    "input_size": int,
    "conf": (int, float),
    "iou": (int, float),
    "classes": str,
    "profile": str,
    "detect_every": int,
    "cpu": bool,
    "enhance": bool,
    "people_recall": bool,
    "person_conf": (int, float),
    "person_probe_conf": (int, float),
    "tile_size": int,
    "tile_overlap": (int, float),
    "view": str,
    "stabilize": bool,
    "no_cmc": bool,
    "no_appearance": bool,
    "calibration": str,
    "session_log": str,
    "record": str,
    "headless": bool,
    "max_frames": int,
}


def load_runtime_config(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    data = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("runtime config must be a JSON object")

    unknown = sorted(set(data) - set(_ALLOWED))
    if unknown:
        raise ValueError(f"unknown runtime config keys: {', '.join(unknown)}")

    out: dict[str, Any] = {}
    for key, value in data.items():
        expected = _ALLOWED[key]
        if isinstance(value, bool) and expected is not bool:
            raise ValueError(f"{key} must be {expected}, not bool")
        if not isinstance(value, expected):
            raise ValueError(f"{key} has invalid type: {type(value).__name__}")
        out[key] = value

    if out.get("camera_backend", "auto") not in {"auto", "dshow", "msmf"}:
        raise ValueError("camera_backend must be auto, dshow or msmf")
    if out.get("profile", "balanced") not in {"quality", "balanced", "speed"}:
        raise ValueError("profile must be quality, balanced or speed")
    if out.get("view", "normal") not in {"normal", "clarity", "lowlight", "edges", "pseudo-thermal"}:
        raise ValueError("invalid view mode")
    if int(out.get("detect_every", 0)) < 0:
        raise ValueError("detect_every cannot be negative")
    if int(out.get("reconnect_attempts", 3)) < 0:
        raise ValueError("reconnect_attempts cannot be negative")
    if int(out.get("reconnect_delay_ms", 250)) < 0:
        raise ValueError("reconnect_delay_ms cannot be negative")
    person_conf = float(out.get("person_conf", 0.18))
    probe_conf = float(out.get("person_probe_conf", 0.08))
    if not 0.0 < probe_conf <= person_conf <= 1.0:
        raise ValueError("require 0 < person_probe_conf <= person_conf <= 1")
    if int(out.get("tile_size", 512)) < 64:
        raise ValueError("tile_size must be at least 64")
    tile_overlap = float(out.get("tile_overlap", 0.20))
    if not 0.0 <= tile_overlap < 0.8:
        raise ValueError("tile_overlap must be in [0, 0.8)")
    return out
