import json

import pytest

from spectratrack.runtime_config import canonical_profile, load_runtime_config, profile_detect_every


def test_runtime_config_loads_valid_values(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({
        "profile": "speed",
        "detect_every": 3,
        "camera_backend": "auto",
        "cpu": True,
    }), encoding="utf-8")
    cfg = load_runtime_config(path)
    assert cfg["profile"] == "speed"
    assert cfg["detect_every"] == 3
    assert cfg["cpu"] is True


def test_runtime_config_accepts_new_profiles_and_perf_report(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({
        "profile": "max-recall",
        "perf_report": "run-performance.json",
    }), encoding="utf-8")
    cfg = load_runtime_config(path)
    assert cfg["profile"] == "max-recall"
    assert cfg["perf_report"] == "run-performance.json"


def test_profile_aliases_preserve_old_configs():
    assert canonical_profile("speed") == "fast"
    assert canonical_profile("quality") == "high-quality"
    assert profile_detect_every("fast") == 3
    assert profile_detect_every("balanced") == 2
    assert profile_detect_every("high-quality") == 1
    assert profile_detect_every("max-recall") == 1


def test_runtime_config_rejects_unknown_keys(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"magic": 1}), encoding="utf-8")
    with pytest.raises(ValueError):
        load_runtime_config(path)


def test_runtime_config_rejects_invalid_profile(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"profile": "ultra"}), encoding="utf-8")
    with pytest.raises(ValueError):
        load_runtime_config(path)
