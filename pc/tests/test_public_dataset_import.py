import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from spectratrack.public_dataset_import import import_crowdhuman, import_mot17
from spectratrack.qa_benchmark import evaluate_frames, load_ground_truth
from spectratrack.vnext_qa import build_frozen_manifest, inspect_corpus


def _write_image(path: Path, width=100, height=80):
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.zeros((height, width, 3), dtype=np.uint8)
    assert cv2.imwrite(str(path), image)


def _read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _mot17_fixture(root: Path):
    sequence = root / "train" / "MOT17-10-FRCNN"
    image_dir = sequence / "img1"
    _write_image(image_dir / "000001.jpg")
    _write_image(image_dir / "000002.jpg")
    (sequence / "gt").mkdir(parents=True)
    (sequence / "seqinfo.ini").write_text(
        "\n".join(
            [
                "[Sequence]",
                "name=MOT17-10-FRCNN",
                "imDir=img1",
                "frameRate=30",
                "seqLength=2",
                "imWidth=100",
                "imHeight=80",
                "imExt=.jpg",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (sequence / "gt" / "gt.txt").write_text(
        "\n".join(
            [
                "1,1,1,1,10,20,1,1,0.90",
                "1,7,20,1,10,20,1,7,1.00",
                "1,9,40,1,10,20,1,3,1.00",
                "2,1,2,1,10,20,1,1,0.60",
                "2,2,60,1,10,20,0,1,1.00",
                "2,3,-20,1,10,20,1,1,0.05",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def test_import_mot17_preserves_identity_ignore_and_source_provenance(tmp_path: Path):
    root = tmp_path / "MOT17"
    _mot17_fixture(root)
    gt = tmp_path / "mot17.jsonl"
    manifest_path = tmp_path / "mot17.import.json"

    manifest = import_mot17(
        dataset_root=root,
        output_ground_truth=gt,
        output_manifest=manifest_path,
        sequences="10",
        importer_source_commit="a" * 40,
        acknowledge_terms=True,
    )

    rows = _read_jsonl(gt)
    assert len(rows) == 2
    assert rows[0]["frame"] == 0
    assert rows[0]["source_frame"] == 1
    assert rows[0]["source_sequence"] == "MOT17-10-FRCNN"
    assert rows[0]["source"] == "train/MOT17-10-FRCNN/img1"
    assert set(rows[0]["tags"]) == {"night_dark", "camera_motion"}
    assert rows[0]["allow_out_of_bounds"] is True

    scored = [obj for obj in rows[0]["objects"] if not obj.get("ignore", False)]
    ignored = [obj for obj in rows[0]["objects"] if obj.get("ignore", False)]
    assert [obj["id"] for obj in scored] == ["MOT17-10:1"]
    assert [obj["id"] for obj in ignored] == ["ignore:MOT17-10:7:7"]
    assert [obj["id"] for obj in rows[1]["objects"]] == ["MOT17-10:1"]
    assert manifest["stats"]["zero_marked_pedestrians_omitted"] == 1
    assert manifest["stats"]["non_intersecting_target_like_omitted"] == 1
    assert manifest["stats"]["other_classes_omitted"] == 1
    assert manifest["importer_source_commit"] == "a" * 40
    assert manifest["dataset"]["split"] == "train"
    assert manifest["source_files"]

    parsed = load_ground_truth(gt)
    assert parsed[0].source_frame == 1
    assert parsed[0].source_sequence == "MOT17-10-FRCNN"
    assert parsed[0].source_fps == pytest.approx(30.0)


def test_public_mot17_import_validates_and_freezes_without_private_review(tmp_path: Path):
    root = tmp_path / "MOT17"
    _mot17_fixture(root)
    gt = tmp_path / "mot17.jsonl"
    import_manifest = tmp_path / "mot17.import.json"
    import_mot17(
        dataset_root=root,
        output_ground_truth=gt,
        output_manifest=import_manifest,
        sequences="10",
        importer_source_commit="b" * 40,
        acknowledge_terms=True,
    )

    report = inspect_corpus(
        video_root=root,
        golden_ground_truth=gt,
        coverage_profile="public-dataset",
        dataset_import_manifests=[import_manifest],
    )
    assert report["valid"]
    assert report["dataset_imports"][0]["dataset"]["name"] == "MOT17"

    frozen = build_frozen_manifest(
        report,
        revision="mot17-public-r1",
        reviewer="MOTChallenge official GT",
        human_confirmed=False,
        public_dataset_confirmed=True,
    )
    assert frozen["human_confirmation"]["kind"] == "official_public_dataset_ground_truth"
    assert frozen["dataset_imports"][0]["importer_source_commit"] == "b" * 40


def _crowdhuman_fixture(root: Path):
    images = root / "Images"
    _write_image(images / "crowd_001.jpg", width=100, height=80)
    annotation = {
        "ID": "crowd_001",
        "gtboxes": [
            {
                "tag": "person",
                "fbox": [-5, 5, 25, 60],
                "vbox": [0, 10, 12, 35],
                "hbox": [2, 5, 8, 10],
                "extra": {"ignore": 0, "box_id": 11, "occ": 1},
            },
            {
                "tag": "mask",
                "fbox": [50, 5, 15, 30],
                "vbox": [52, 7, 10, 20],
                "hbox": [54, 7, 5, 5],
                "extra": {"ignore": 1, "box_id": 12},
            },
        ],
    }
    (root / "annotation_val.odgt").write_text(json.dumps(annotation) + "\n", encoding="utf-8")


def test_import_crowdhuman_uses_full_body_by_default_and_disables_tracking(tmp_path: Path):
    root = tmp_path / "CrowdHuman"
    root.mkdir()
    _crowdhuman_fixture(root)
    gt = tmp_path / "crowdhuman.jsonl"
    manifest_path = tmp_path / "crowdhuman.import.json"

    manifest = import_crowdhuman(
        dataset_root=root,
        annotations="annotation_val.odgt",
        images_dir="Images",
        output_ground_truth=gt,
        output_manifest=manifest_path,
        importer_source_commit="c" * 40,
        acknowledge_terms=True,
    )

    row = _read_jsonl(gt)[0]
    assert row["frame"] == 0
    assert row["source_sequence"] == "CrowdHuman-val"
    assert row["allow_out_of_bounds"] is True
    assert row["objects"][0]["bbox"] == [-5.0, 5.0, 20.0, 65.0]
    assert "id" not in row["objects"][0]
    assert "crowdhuman_occ_1" in row["objects"][0]["attributes"]
    assert row["objects"][1]["ignore"] is True
    assert manifest["conversion_settings"]["bbox_kind"] == "full"
    assert manifest["tracking_supported"] is False

    frames = load_ground_truth(gt)
    metrics = evaluate_frames(frames, {}, {})
    assert metrics["tracking"]["recall"] is None


def test_import_crowdhuman_visible_is_explicit_opt_in(tmp_path: Path):
    root = tmp_path / "CrowdHuman"
    root.mkdir()
    _crowdhuman_fixture(root)
    gt = tmp_path / "crowdhuman-visible.jsonl"

    manifest = import_crowdhuman(
        dataset_root=root,
        annotations="annotation_val.odgt",
        images_dir="Images",
        output_ground_truth=gt,
        output_manifest=tmp_path / "crowdhuman-visible.import.json",
        bbox_kind="visible",
        importer_source_commit="d" * 40,
        acknowledge_terms=True,
    )

    row = _read_jsonl(gt)[0]
    assert row["objects"][0]["bbox"] == [0.0, 10.0, 12.0, 45.0]
    assert manifest["conversion_settings"]["bbox_kind"] == "visible"


def test_import_crowdhuman_rejects_training_annotations(tmp_path: Path):
    root = tmp_path / "CrowdHuman"
    root.mkdir()
    _write_image(root / "Images" / "x.jpg")
    (root / "annotation_train.odgt").write_text(
        json.dumps({"ID": "x", "gtboxes": []}) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="validation split"):
        import_crowdhuman(
            dataset_root=root,
            annotations="annotation_train.odgt",
            images_dir="Images",
            output_ground_truth=tmp_path / "out.jsonl",
            output_manifest=tmp_path / "out.import.json",
            importer_source_commit="e" * 40,
            acknowledge_terms=True,
        )
