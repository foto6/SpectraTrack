import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from spectratrack.public_dataset_import import (
    import_crowdhuman,
    import_dancetrack,
    import_mot17,
    import_nightowls,
)
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
                "1,7,20,1,10,20,0,7,1.00",
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
            {
                "tag": "person",
                "fbox": [150, 90, 20, 30],
                "vbox": [152, 92, 10, 15],
                "hbox": [154, 92, 5, 5],
                "extra": {"ignore": 0, "box_id": 13},
            },
            {
                "tag": "mask",
                "fbox": [-40, 90, 10, 20],
                "vbox": [-38, 92, 5, 10],
                "hbox": [-36, 92, 3, 3],
                "extra": {"ignore": 1, "box_id": 14},
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
    assert len(row["objects"]) == 2
    assert manifest["stats"]["non_intersecting_target_like_omitted"] == 2
    assert manifest["conversion_settings"]["bbox_kind"] == "full"
    assert manifest["conversion_settings"]["non_intersecting_target_like_boxes"] == "omitted"
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


def _dancetrack_fixture(root: Path, *, trailing="1,1,1"):
    sequence = root / "val" / "dancetrack0001"
    image_dir = sequence / "img1"
    _write_image(image_dir / "00000001.jpg")
    _write_image(image_dir / "00000002.jpg")
    (sequence / "gt").mkdir(parents=True)
    (sequence / "seqinfo.ini").write_text(
        "\n".join(
            [
                "[Sequence]",
                "name=dancetrack0001",
                "imDir=img1",
                "frameRate=20",
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
                f"1,4,1,1,10,20,{trailing}",
                f"2,4,2,1,10,20,{trailing}",
                f"2,9,30,10,12,22,{trailing}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def test_import_dancetrack_reuses_mot_geometry_without_mot17_semantics(tmp_path: Path):
    root = tmp_path / "DanceTrack"
    _dancetrack_fixture(root)
    gt = tmp_path / "dancetrack.jsonl"
    manifest_path = tmp_path / "dancetrack.import.json"

    manifest = import_dancetrack(
        dataset_root=root,
        output_ground_truth=gt,
        output_manifest=manifest_path,
        split="val",
        importer_source_commit="f" * 40,
        acknowledge_terms=True,
    )

    rows = _read_jsonl(gt)
    assert len(rows) == 2
    assert rows[0]["video"] == "golden/public/dancetrack-val/dancetrack0001"
    assert rows[0]["frame"] == 0
    assert rows[0]["source_frame"] == 1
    assert rows[0]["source_sequence"] == "dancetrack0001"
    assert rows[0]["source_fps"] == pytest.approx(20.0)
    assert rows[0]["tags"] == []
    assert rows[0]["objects"][0]["id"] == "DanceTrack:dancetrack0001:4"
    assert rows[0]["objects"][0]["bbox"] == [0.0, 0.0, 10.0, 20.0]
    assert rows[0]["objects"][0].get("attributes", []) == []
    assert manifest["dataset"]["split"] == "val"
    assert manifest["tracking_supported"] is True
    assert manifest["conversion_settings"]["trailing_fields"].startswith("required constant")
    assert "scored_classes" not in manifest["conversion_settings"]
    assert "visibility_attributes" not in manifest["conversion_settings"]
    assert manifest["sequences"][0]["gt_rows"] == 3
    assert manifest["importer_source_commit"] == "f" * 40

    parsed = load_ground_truth(gt)
    assert parsed[0].source_sequence == "dancetrack0001"
    assert parsed[0].objects[0].object_id == "DanceTrack:dancetrack0001:4"


def test_import_dancetrack_rejects_nonconstant_trailing_fields(tmp_path: Path):
    root = tmp_path / "DanceTrack"
    _dancetrack_fixture(root, trailing="1,2,1")

    with pytest.raises(ValueError, match="trailing fields"):
        import_dancetrack(
            dataset_root=root,
            output_ground_truth=tmp_path / "bad.jsonl",
            output_manifest=tmp_path / "bad.import.json",
            split="val",
            importer_source_commit="1" * 40,
            acknowledge_terms=True,
        )


def test_import_dancetrack_validates_seqinfo_image_dimensions(tmp_path: Path):
    root = tmp_path / "DanceTrack"
    _dancetrack_fixture(root)
    _write_image(root / "val" / "dancetrack0001" / "img1" / "00000002.jpg", width=99, height=80)

    with pytest.raises(ValueError, match="dimensions"):
        import_dancetrack(
            dataset_root=root,
            output_ground_truth=tmp_path / "bad.jsonl",
            output_manifest=tmp_path / "bad.import.json",
            split="val",
            importer_source_commit="2" * 40,
            acknowledge_terms=True,
        )


def _nightowls_sdk_fixture(root: Path):
    sdk = root / "nightowlsapi"
    (sdk / "python").mkdir(parents=True)
    (sdk / "README.md").write_text(
        "NightOwls API\nLicense: non-commercial research only\n",
        encoding="utf-8",
    )
    (sdk / "python" / "coco.py").write_text("# official COCO-compatible loader fixture\n", encoding="utf-8")
    (sdk / "python" / "eval.py").write_text(
        "annFile = 'nightowls_validation.json'\n",
        encoding="utf-8",
    )
    (sdk / "python" / "eval_MR_multisetup.py").write_text(
        "catIds = [1]  # pedestrian evaluation\n",
        encoding="utf-8",
    )


def _nightowls_fixture(root: Path, *, missing_tracking=False):
    _nightowls_sdk_fixture(root)
    image_dir = root / "nightowls_validation"
    for index in range(4):
        _write_image(image_dir / f"frame_{index:02d}.png", width=1024, height=640)

    images = [
        {
            "id": 100 + index,
            "file_name": f"frame_{index:02d}.png",
            "width": 1024,
            "height": 640,
            "daytime": "night",
            "recordings_id": 7.0,
            "timestamp": 1000 + index * 10,
        }
        for index in range(4)
    ]
    annotations = [
        {
            "id": 1,
            "image_id": 100,
            "category_id": 1,
            "bbox": [10, 20, 20, 70],
            "area": 1400,
            "tracking_id": None if missing_tracking else 55,
            "occluded": False,
            "difficult": False,
            "pose_id": 1,
            "ignore": 0,
            "truncated": False,
        },
        {
            "id": 2,
            "image_id": 100,
            "category_id": 2,
            "bbox": [100, 20, 25, 75],
            "area": 1875,
            "tracking_id": 77,
            "occluded": False,
            "difficult": False,
            "pose_id": 2,
            "ignore": 0,
            "truncated": False,
        },
        {
            "id": 3,
            "image_id": 100,
            "category_id": 4,
            "bbox": [200, 20, 50, 80],
            "area": 4000,
            "tracking_id": 88,
            "occluded": None,
            "difficult": None,
            "pose_id": 5,
            "ignore": 1,
            "truncated": False,
        },
        {
            "id": 4,
            "image_id": 101,
            "category_id": 1,
            "bbox": [12, 21, 20, 70],
            "area": 1400,
            "tracking_id": 55,
            "occluded": True,
            "difficult": True,
            "pose_id": 2,
            "ignore": 0,
            "truncated": True,
        },
        {
            "id": 5,
            "image_id": 102,
            "category_id": 1,
            "bbox": [300, 100, 30, 100],
            "area": 3000,
            "tracking_id": 66,
            "occluded": False,
            "difficult": False,
            "pose_id": 1,
            "ignore": 1,
            "truncated": False,
        },
    ]
    data = {
        "images": images,
        "annotations": annotations,
        "categories": [
            {"id": 1, "name": "pedestrian"},
            {"id": 2, "name": "bicycledriver"},
            {"id": 3, "name": "motorbikedriver"},
            {"id": 4, "name": "ignore"},
        ],
        "poses": [
            {"id": 1, "name": "standing"},
            {"id": 2, "name": "walking"},
            {"id": 5, "name": "unknown"},
        ],
    }
    (root / "nightowls_validation.json").write_text(
        json.dumps(data),
        encoding="utf-8",
    )


def test_import_nightowls_preserves_official_semantics_and_tracking(tmp_path: Path):
    root = tmp_path / "NightOwls"
    root.mkdir()
    _nightowls_fixture(root)
    gt = tmp_path / "nightowls.jsonl"
    import_manifest = tmp_path / "nightowls.import.json"

    manifest = import_nightowls(
        dataset_root=root,
        annotations="nightowls_validation.json",
        images_dir="nightowls_validation",
        sdk_dir="nightowlsapi",
        output_ground_truth=gt,
        output_manifest=import_manifest,
        importer_source_commit="9" * 40,
        acknowledge_terms=True,
    )

    rows = _read_jsonl(gt)
    assert len(rows) == 4
    assert manifest["dataset"]["split"] == "validation"
    assert manifest["tracking_supported"] is True
    assert manifest["tracking_contract"]["repeated_trajectory_observed"] is True
    assert manifest["stats"]["rider_or_other_classes_omitted"] == 1
    assert manifest["stats"]["official_ignore_regions"] == 1

    first = rows[0]
    assert first["video"] == "golden/public/nightowls-val/recording-7"
    assert first["frame"] == 0
    assert first["source_frame"] == 100
    assert first["source_sequence"] == "NightOwls:recording-7"
    assert first["tags"] == ["night_dark"]
    scored = [obj for obj in first["objects"] if not obj.get("ignore", False)]
    ignored = [obj for obj in first["objects"] if obj.get("ignore", False)]
    assert [obj["id"] for obj in scored] == ["NightOwls:7:55"]
    assert len(ignored) == 1
    assert ignored[0]["attributes"] == ["nightowls_official_ignore_region"]
    assert all("bicycledriver" not in obj.get("attributes", []) for obj in first["objects"])

    second_scored = [obj for obj in rows[1]["objects"] if not obj.get("ignore", False)][0]
    assert second_scored["id"] == "NightOwls:7:55"
    assert "nightowls_occluded_true" in second_scored["attributes"]
    assert "nightowls_difficult_true" in second_scored["attributes"]
    assert "nightowls_pose_walking" in second_scored["attributes"]
    assert "nightowls_truncated_true" in second_scored["attributes"]

    parsed = load_ground_truth(gt)
    assert parsed[0].objects[0].object_id == "NightOwls:7:55"

    report = inspect_corpus(
        video_root=root,
        golden_ground_truth=gt,
        coverage_profile="public-dataset",
        dataset_import_manifests=[import_manifest],
    )
    assert report["valid"]
    assert report["golden"]["videos"][0]["source_kind"] == "frame_images"
    assert report["golden"]["videos"][0]["source_count"] == 4


def test_import_nightowls_disables_tracking_when_official_ids_are_incomplete(tmp_path: Path):
    root = tmp_path / "NightOwls"
    root.mkdir()
    _nightowls_fixture(root, missing_tracking=True)
    gt = tmp_path / "nightowls.jsonl"

    manifest = import_nightowls(
        dataset_root=root,
        annotations="nightowls_validation.json",
        images_dir="nightowls_validation",
        sdk_dir="nightowlsapi",
        output_ground_truth=gt,
        output_manifest=tmp_path / "nightowls.import.json",
        importer_source_commit="8" * 40,
        acknowledge_terms=True,
    )

    assert manifest["tracking_supported"] is False
    assert manifest["tracking_contract"]["all_scored_pedestrians_have_valid_tracking_id"] is False
    rows = _read_jsonl(gt)
    scored = [
        obj
        for row in rows
        for obj in row["objects"]
        if not obj.get("ignore", False)
    ]
    assert all("id" not in obj for obj in scored)


def test_import_nightowls_slice_is_deterministic_and_tracking_disabled(tmp_path: Path):
    root = tmp_path / "NightOwls"
    root.mkdir()
    _nightowls_fixture(root)

    manifests = []
    for suffix in ("a", "b"):
        manifest = import_nightowls(
            dataset_root=root,
            annotations="nightowls_validation.json",
            images_dir="nightowls_validation",
            sdk_dir="nightowlsapi",
            output_ground_truth=tmp_path / f"nightowls-{suffix}.jsonl",
            output_manifest=tmp_path / f"nightowls-{suffix}.import.json",
            slice_frames=3,
            slice_seed="round2-fixed-seed",
            importer_source_commit="7" * 40,
            acknowledge_terms=True,
        )
        manifests.append(manifest)

    assert manifests[0]["slice"]["selected_image_ids"] == manifests[1]["slice"]["selected_image_ids"]
    assert manifests[0]["slice"]["selected_image_ids_sha256"] == manifests[1]["slice"]["selected_image_ids_sha256"]
    assert manifests[0]["tracking_supported"] is False
    assert manifests[0]["slice"]["tracking_supported"] is False
    assert len(manifests[0]["slice"]["selected_image_ids"]) == 3


def test_import_nightowls_rejects_wrong_pedestrian_category_contract(tmp_path: Path):
    root = tmp_path / "NightOwls"
    root.mkdir()
    _nightowls_fixture(root)
    path = root / "nightowls_validation.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    data["categories"][0]["name"] = "person"
    path.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(ValueError, match="id=1/name=pedestrian"):
        import_nightowls(
            dataset_root=root,
            annotations="nightowls_validation.json",
            images_dir="nightowls_validation",
            sdk_dir="nightowlsapi",
            output_ground_truth=tmp_path / "bad.jsonl",
            output_manifest=tmp_path / "bad.import.json",
            importer_source_commit="6" * 40,
            acknowledge_terms=True,
        )

