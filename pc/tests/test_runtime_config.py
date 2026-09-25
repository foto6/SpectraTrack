import json

import pytest

from spectratrack.runtime_config import load_runtime_config


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
