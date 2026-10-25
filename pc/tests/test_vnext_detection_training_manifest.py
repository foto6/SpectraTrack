import copy

import pytest

from spectratrack.research.vnext_detection_training_manifest import (
    REQUIRED_FORBIDDEN_EVAL_SETS,
    SCHEMA,
    validate_manifest_data,
)


def _manifest():
    return {
        "schema": SCHEMA,
        "revision": "a1-train-prelock-r1",
        "purpose": "detector_finetune",
        "created_from_commit": "a" * 40,
        "forbidden_evaluation_sets": sorted(REQUIRED_FORBIDDEN_EVAL_SETS),
        "items": [
            {
                "item_id": "positive-night-1",
                "role": "positive",
                "selection_origin": "human",
                "source_dataset": "future-cctv-train",
                "source_revision": "train-r1",
                "source_path": "images/night-1.jpg",
                "source_sha256": "1" * 64,
                "image_sha256": "2" * 64,
                "video": "train/night-1",
                "frame": 0,
                "bbox": [10, 20, 30, 80],
                "label": "person",
                "tags": ["night", "small"],
            },
            {
                "item_id": "hard-negative-1",
                "role": "hard_negative",
                "selection_origin": "teacher_candidate",
                "source_dataset": "future-cctv-train",
                "source_revision": "train-r1",
                "source_path": "images/negative-1.jpg",
                "source_sha256": "3" * 64,
                "image_sha256": "4" * 64,
                "video": "train/negative-1",
                "frame": 0,
                "bbox": [40, 30, 90, 100],
                "label": "background",
                "tags": ["negative", "night"],
            },
        ],
    }


def test_training_manifest_accepts_disjoint_night_positive_and_hard_negative():
    result = validate_manifest_data(_manifest())

    assert result["status"] == "VALID_DISJOINT_TRAINING_MANIFEST"
    assert result["counts_by_role"]["positive"] == 1
    assert result["counts_by_role"]["hard_negative"] == 1


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("source_dataset", "NightOwls"),
        ("source_revision", "nightowls-public-r1"),
        ("source_path", "heldout4200/frame.jpg"),
        ("video", "FULL5000/0001"),
    ],
)
def test_training_manifest_rejects_eval_leakage_tokens(field, value):
    manifest = _manifest()
    manifest["items"][0][field] = value

    with pytest.raises(ValueError, match="leakage"):
        validate_manifest_data(manifest)


def test_training_manifest_requires_prelocked_eval_exclusions():
    manifest = _manifest()
    manifest["forbidden_evaluation_sets"].remove("nightowls-holdout4200")

    with pytest.raises(ValueError, match="prelock"):
        validate_manifest_data(manifest)


def test_training_manifest_rejects_same_source_frame_bbox_across_roles():
    manifest = _manifest()
    duplicate = copy.deepcopy(manifest["items"][0])
    duplicate["item_id"] = "duplicate-as-negative"
    duplicate["role"] = "hard_negative"
    duplicate["label"] = "background"
    duplicate["tags"] = ["negative"]
    manifest["items"].append(duplicate)

    with pytest.raises(ValueError, match="duplicate source/frame/bbox"):
        validate_manifest_data(manifest)


def test_positive_requires_night_small_or_occluded_condition():
    manifest = _manifest()
    manifest["items"][0]["tags"] = ["blur"]

    with pytest.raises(ValueError, match="positive item must include"):
        validate_manifest_data(manifest)
