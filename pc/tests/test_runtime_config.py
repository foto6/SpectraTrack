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
    path = tmp_path / "recall.json"
    path.write_text(json.dumps({
        "detector_mode": "people-recall",
        "person_conf": 0.12,
        "person_tile_size": 640,
        "person_tile_overlap": 0.2,
        "person_merge_iou": 0.55,
    }), encoding="utf-8")
    cfg = load_runtime_config(path)
    assert cfg["detector_mode"] == "people-recall"
    assert cfg["person_conf"] == 0.12


def test_runtime_config_rejects_invalid_people_recall_settings(tmp_path):
    path = tmp_path / "bad-recall.json"
    path.write_text(json.dumps({
        "detector_mode": "people-recall",
        "person_tile_overlap": 1.0,
    }), encoding="utf-8")
    with pytest.raises(ValueError):
        load_runtime_config(path)



def test_runtime_config_accepts_adaptive_people_recall_enhancement(tmp_path):
    path = tmp_path / "adaptive-recall.json"
    path.write_text(json.dumps({
        "detector_mode": "people-recall",
        "person_conf": 0.12,
        "people_recall_enhancement": "adaptive",
    }), encoding="utf-8")
    cfg = load_runtime_config(path)
    assert cfg["people_recall_enhancement"] == "adaptive"


def test_runtime_config_rejects_unknown_people_recall_enhancement(tmp_path):
    path = tmp_path / "bad-enhancement.json"
    path.write_text(json.dumps({"people_recall_enhancement": "magic"}), encoding="utf-8")
    with pytest.raises(ValueError):
        load_runtime_config(path)
