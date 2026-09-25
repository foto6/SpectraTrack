from __future__ import annotations

import json
from pathlib import Path
from typing import Any


PROFILE_NAMES = ("fast", "balanced", "high-quality", "max-recall")
_PROFILE_ALIASES = {
    "speed": "fast",
    "quality": "high-quality",
}
_PROFILE_DETECT_EVERY = {
    "fast": 3,
    "balanced": 2,
    "high-quality": 1,
    "max-recall": 1,
}


def canonical_profile(value: str) -> str:
    profile = _PROFILE_ALIASES.get(value, value)
    if profile not in PROFILE_NAMES:
        raise ValueError(
            "profile must be fast, balanced, high-quality or max-recall "
            "(legacy speed/quality aliases are still supported)"
        )
    return profile


def profile_detect_every(value: str) -> int:
    return _PROFILE_DETECT_EVERY[canonical_profile(value)]


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
    "view": str,
    "stabilize": bool,
    "no_cmc": bool,
    "no_appearance": bool,
    "calibration": str,
    "session_log": str,
    "perf_report": str,
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
    canonical_profile(out.get("profile", "balanced"))
    if out.get("view", "normal") not in {"normal", "clarity", "lowlight", "edges", "pseudo-thermal"}:
        raise ValueError("invalid view mode")
    if int(out.get("detect_every", 0)) < 0:
        raise ValueError("detect_every cannot be negative")
    if int(out.get("reconnect_attempts", 3)) < 0:
        raise ValueError("reconnect_attempts cannot be negative")
    if int(out.get("reconnect_delay_ms", 250)) < 0:
        raise ValueError("reconnect_delay_ms cannot be negative")
    return out
