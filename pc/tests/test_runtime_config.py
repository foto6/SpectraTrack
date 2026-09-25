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


def test_runtime_config_accepts_people_recall_settings(tmp_path):
    path = tmp_path / "people.json"
    path.write_text(json.dumps({
        "people_recall": True,
        "person_conf": 0.18,
        "person_probe_conf": 0.08,
        "tile_size": 512,
        "tile_overlap": 0.2,
    }), encoding="utf-8")
    cfg = load_runtime_config(path)
    assert cfg["people_recall"] is True
    assert cfg["tile_size"] == 512


@pytest.mark.parametrize("payload", [
    {"person_conf": 0.05, "person_probe_conf": 0.08},
    {"tile_size": 32},
    {"tile_overlap": 0.8},
])
def test_runtime_config_rejects_invalid_people_recall_settings(tmp_path, payload):
    path = tmp_path / "bad_people.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError):
        load_runtime_config(path)
