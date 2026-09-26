from __future__ import annotations

import argparse
import configparser
import hashlib
import json
import math
from pathlib import Path
import subprocess
from typing import Any, Iterable

from .integrity import sha256_file
from .qa_benchmark import ground_truth_sha256

IMPORT_SCHEMA = "spectratrack-public-dataset-import-v1"

MOT17_LICENSE = {
    "name": "CC BY-NC-SA 3.0",
    "url": "https://motchallenge.net/",
    "note": "MOTChallenge datasets are non-commercial and require attribution/share-alike.",
}
CROWDHUMAN_TERMS = {
    "name": "CrowdHuman terms of use",
    "url": "https://www.crowdhuman.org/download.html",
    "note": (
        "Non-commercial research/education only; CrowdHuman images must not be redistributed."
    ),
}
DANCETRACK_TERMS = {
    "annotations": "CC BY 4.0",
    "dataset_media": "non-commercial research only",
    "code": "MIT",
    "url": "https://github.com/DanceTrack/DanceTrack",
}
NIGHTOWLS_TERMS = {
    "name": "NightOwls dataset terms",
    "url": "https://www.nightowls-dataset.org/",
    "note": (
        "Non-commercial research/teaching/personal experimentation; citation required; "
        "dataset and modified versions must not be redistributed."
    ),
}

MOT17_TARGET_CLASS = 1
MOT17_IGNORE_CLASSES = {2, 7, 8, 12}
MOT17_CLASS_NAMES = {
    1: "pedestrian",
    2: "person_on_vehicle",
    3: "car",
    4: "bicycle",
    5: "motorbike",
    6: "non_motorized_vehicle",
    7: "static_person",
    8: "distractor",
    9: "occluder",
    10: "occluder_on_ground",
    11: "occluder_full",
    12: "reflection",
}

MOT17_TRAIN_METADATA = {
    "MOT17-02": {
        "description": "People walking around a large square.",
        "tags": (),
    },
    "MOT17-04": {
        "description": "Pedestrian street at night, elevated viewpoint",
        "tags": ("night_dark", "high_angle_cctv"),
    },
    "MOT17-05": {
        "description": "Street scene from a moving platform",
        "tags": ("camera_motion",),
    },
    "MOT17-09": {
        "description": "A pedestrian street scene filmed from a low angle.",
        "tags": ("low_angle",),
    },
    "MOT17-10": {
        "description": "A pedestrian scene filmed at night by a moving camera",
        "tags": ("night_dark", "camera_motion"),
    },
    "MOT17-11": {
        "description": "Forward moving camera in a busy shopping mall",
        "tags": ("camera_motion",),
    },
    "MOT17-13": {
        "description": "Filmed from a bus on a busy intersection",
        "tags": (),
    },
}


def _canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _write_json(path: str | Path, payload: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _is_hex(value: str, length: int) -> bool:
    if len(value) != length:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def _git_head() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=3,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    value = result.stdout.strip()
    return value or None


def _importer_commit(explicit: str | None) -> str:
    value = explicit or _git_head()
    if value is None or not _is_hex(value, 40):
        raise ValueError(
            "Importer source commit must be a full 40-hex SHA; pass --importer-source-commit when git HEAD is unavailable"
        )
    return value


def _relative_path(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError as exc:
        raise ValueError(f"{path} must be inside dataset root {root}") from exc


def _source_file(path: Path, root: Path, role: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        "path": _relative_path(path, root),
        "role": role,
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }


def _visibility_attribute(value: float) -> str:
    if value < 0.25:
        return "visibility_lt_0_25"
    if value < 0.50:
        return "visibility_0_25_0_50"
    if value < 0.75:
        return "visibility_0_50_0_75"
    return "visibility_ge_0_75"


def _xywh_to_xyxy(box: Any, where: str) -> list[float]:
    if not isinstance(box, list) or len(box) != 4:
        raise ValueError(f"{where}: expected [x,y,w,h]")
    try:
        x, y, width, height = (float(item) for item in box)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{where}: bbox must be numeric") from exc
    if not all(math.isfinite(item) for item in (x, y, width, height)):
        raise ValueError(f"{where}: bbox must be finite")
    if width <= 0.0 or height <= 0.0:
        raise ValueError(f"{where}: bbox must have positive width/height")
    return [x, y, x + width, y + height]


def _mot_bbox_to_canonical(x: float, y: float, width: float, height: float) -> list[float]:
    # MOT17 coordinates are 1-based. Preserve the full annotated extent, including boxes
    # that extend beyond the image boundary under the official full-body policy.
    return [x - 1.0, y - 1.0, x - 1.0 + width, y - 1.0 + height]


def _bbox_intersects_image(bbox: list[float], width: int, height: int) -> bool:
    x1, y1, x2, y2 = bbox
    return x2 > 0.0 and y2 > 0.0 and x1 < float(width) and y1 < float(height)


def _read_seqinfo(path: Path) -> dict[str, Any]:
    parser = configparser.ConfigParser()
    with path.open("r", encoding="utf-8-sig") as handle:
        parser.read_file(handle)
    if "Sequence" not in parser:
        raise ValueError(f"{path}: missing [Sequence]")
    section = parser["Sequence"]
    required = ("name", "imDir", "frameRate", "seqLength", "imWidth", "imHeight", "imExt")
    missing = [key for key in required if key not in section]
    if missing:
        raise ValueError(f"{path}: missing seqinfo keys {missing}")
    return {
        "name": section["name"],
        "im_dir": section["imDir"],
        "fps": float(section["frameRate"]),
        "frame_count": int(section["seqLength"]),
        "width": int(section["imWidth"]),
        "height": int(section["imHeight"]),
        "extension": section["imExt"],
    }


def _parse_mot_gt(path: Path) -> dict[int, list[dict[str, Any]]]:
    by_frame: dict[int, list[dict[str, Any]]] = {}
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, raw in enumerate(handle, start=1):
            raw = raw.strip()
            if not raw:
                continue
            fields = [item.strip() for item in raw.split(",")]
            if len(fields) < 9:
                raise ValueError(f"{path}:{line_number}: expected at least 9 CSV fields")
            try:
                frame = int(float(fields[0]))
                object_id = int(float(fields[1]))
                x = float(fields[2])
                y = float(fields[3])
                width = float(fields[4])
                height = float(fields[5])
                valid = float(fields[6])
                class_id = int(float(fields[7]))
                visibility = float(fields[8])
            except ValueError as exc:
                raise ValueError(f"{path}:{line_number}: invalid MOT numeric field") from exc
            if frame <= 0:
                raise ValueError(f"{path}:{line_number}: MOT frame must be 1-based positive")
            if width <= 0.0 or height <= 0.0:
                raise ValueError(f"{path}:{line_number}: bbox must have positive size")
            if not 0.0 <= visibility <= 1.0:
                raise ValueError(f"{path}:{line_number}: visibility must be in [0,1]")
            by_frame.setdefault(frame, []).append(
                {
                    "id": object_id,
                    "bbox": _mot_bbox_to_canonical(x, y, width, height),
                    "valid": valid,
                    "class_id": class_id,
                    "visibility": visibility,
                }
            )
    return by_frame


def _mot_sequence_names(value: str | None) -> list[str]:
    if value is None or not value.strip():
        return sorted(MOT17_TRAIN_METADATA)
    result: list[str] = []
    for item in value.split(","):
        name = item.strip().upper()
        if not name:
            continue
        if name.isdigit():
            name = f"MOT17-{int(name):02d}"
        if name not in MOT17_TRAIN_METADATA:
            raise ValueError(f"Unsupported MOT17 train sequence {name!r}")
        result.append(name)
    if not result:
        raise ValueError("No MOT17 sequences selected")
    return sorted(set(result))


def import_mot17(
    *,
    dataset_root: str | Path,
    output_ground_truth: str | Path,
    output_manifest: str | Path,
    detector_variant: str = "FRCNN",
    sequences: str | None = None,
    logical_prefix: str = "golden/public/mot17",
    importer_source_commit: str | None = None,
    acknowledge_terms: bool = False,
) -> dict[str, Any]:
    if not acknowledge_terms:
        raise ValueError("MOT17 import requires --acknowledge-terms")
    variant = detector_variant.upper()
    if variant not in {"DPM", "FRCNN", "SDP"}:
        raise ValueError("detector_variant must be DPM, FRCNN, or SDP")

    root = Path(dataset_root)
    selected = _mot_sequence_names(sequences)
    rows: list[dict[str, Any]] = []
    source_files: list[dict[str, Any]] = []
    sequence_manifest: list[dict[str, Any]] = []
    counts = {
        "scored_pedestrians": 0,
        "ignored_target_like": 0,
        "zero_marked_pedestrians_omitted": 0,
        "non_intersecting_target_like_omitted": 0,
        "other_classes_omitted": 0,
    }

    for base_name in selected:
        source_sequence = f"{base_name}-{variant}"
        sequence_dir = root / "train" / source_sequence
        seqinfo_path = sequence_dir / "seqinfo.ini"
        gt_path = sequence_dir / "gt" / "gt.txt"
        info = _read_seqinfo(seqinfo_path)
        if info["name"] != source_sequence:
            raise ValueError(
                f"{seqinfo_path}: sequence name {info['name']!r} does not match {source_sequence!r}"
            )
        image_dir = sequence_dir / info["im_dir"]
        if not image_dir.is_dir():
            raise FileNotFoundError(image_dir)
        gt = _parse_mot_gt(gt_path)
        source_rel = _relative_path(image_dir, root)
        logical_video = f"{logical_prefix.rstrip('/')}/{base_name}"
        metadata = MOT17_TRAIN_METADATA[base_name]
        tags = sorted(metadata["tags"])

        source_files.append(_source_file(seqinfo_path, root, "sequence_metadata"))
        source_files.append(_source_file(gt_path, root, "ground_truth"))
        image_entries: list[dict[str, Any]] = []
        for frame_number in range(1, info["frame_count"] + 1):
            image_path = image_dir / f"{frame_number:06d}{info['extension']}"
            entry = _source_file(image_path, root, "image")
            source_files.append(entry)
            image_entries.append(entry)

            objects: list[dict[str, Any]] = []
            for annotation in gt.get(frame_number, []):
                class_id = int(annotation["class_id"])
                valid = float(annotation["valid"])
                visibility = float(annotation["visibility"])
                object_id = int(annotation["id"])
                source_annotation = {
                    "dataset": "MOT17",
                    "original_id": object_id,
                    "class_id": class_id,
                    "class_name": MOT17_CLASS_NAMES.get(class_id, "unknown"),
                    "valid": valid,
                    "visibility": visibility,
                }
                if class_id == MOT17_TARGET_CLASS:
                    if valid <= 0.0:
                        counts["zero_marked_pedestrians_omitted"] += 1
                        continue
                    if not _bbox_intersects_image(annotation["bbox"], info["width"], info["height"]):
                        counts["non_intersecting_target_like_omitted"] += 1
                        continue
                    counts["scored_pedestrians"] += 1
                    objects.append(
                        {
                            "id": f"{base_name}:{object_id}",
                            "label": "person",
                            "bbox": annotation["bbox"],
                            "attributes": [f"mot17_{_visibility_attribute(visibility)}"],
                            "source_annotation": source_annotation,
                        }
                    )
                elif class_id in MOT17_IGNORE_CLASSES:
                    # MOT17 stores target-like distractor/static/reflection regions with
                    # the GT consider/ignore flag set to 0. They must still become
                    # canonical ignore regions so predictions overlapping them are not
                    # counted as ordinary false positives.
                    if not _bbox_intersects_image(annotation["bbox"], info["width"], info["height"]):
                        counts["non_intersecting_target_like_omitted"] += 1
                        continue
                    counts["ignored_target_like"] += 1
                    objects.append(
                        {
                            "id": f"ignore:{base_name}:{class_id}:{object_id}",
                            "label": "person",
                            "bbox": annotation["bbox"],
                            "ignore": True,
                            "attributes": [
                                f"mot17_ignore_{MOT17_CLASS_NAMES.get(class_id, class_id)}"
                            ],
                            "source_annotation": source_annotation,
                        }
                    )
                else:
                    counts["other_classes_omitted"] += 1

            rows.append(
                {
                    "video": logical_video,
                    "frame": frame_number - 1,
                    "source": source_rel,
                    "source_frame": frame_number,
                    "source_sequence": source_sequence,
                    "source_fps": info["fps"],
                    "allow_out_of_bounds": True,
                    "tags": tags,
                    "objects": objects,
                    "source_metadata": {
                        "dataset": "MOT17",
                        "dataset_split": "train",
                        "source_sequence": source_sequence,
                        "original_frame": frame_number,
                    },
                }
            )

        sequence_manifest.append(
            {
                "logical_video": logical_video,
                "base_sequence": base_name,
                "source_sequence": source_sequence,
                "description": metadata["description"],
                "semantic_tags_from_official_metadata": tags,
                "source": source_rel,
                "fps": info["fps"],
                "frame_count": info["frame_count"],
                "width": info["width"],
                "height": info["height"],
                "image_files_sha256": _canonical_sha256(
                    [{"path": item["path"], "sha256": item["sha256"]} for item in image_entries]
                ),
            }
        )

    rows.sort(key=lambda item: (item["video"], item["frame"]))
    _write_jsonl(output_ground_truth, rows)
    gt_sha = ground_truth_sha256(output_ground_truth)
    manifest = {
        "schema": IMPORT_SCHEMA,
        "dataset": {
            "name": "MOT17",
            "version": "MOT17",
            "split": "train",
            "license": MOT17_LICENSE,
        },
        "annotation_origin": "official_public_ground_truth",
        "terms_acknowledged": True,
        "tracking_supported": True,
        "importer_source_commit": _importer_commit(importer_source_commit),
        "selected_sequences": selected,
        "conversion_settings": {
            "detector_variant_for_source_frames": variant,
            "canonical_frame_index": "MOT source frame minus 1",
            "canonical_bbox": "MOT 1-based xywh converted to 0-based xyxy without clipping",
            "scored_classes": [MOT17_TARGET_CLASS],
            "ignored_target_like_classes": sorted(MOT17_IGNORE_CLASSES),
            "zero_marked_pedestrians": "omitted",
            "non_intersecting_target_like_boxes": "omitted",
            "other_classes": "omitted",
            "visibility_attributes": [0.25, 0.50, 0.75],
            "semantic_tags": "only official sequence metadata descriptions",
        },
        "logical_prefix": logical_prefix.rstrip("/"),
        "source_files": source_files,
        "source_files_sha256": _canonical_sha256(
            [{"path": item["path"], "sha256": item["sha256"]} for item in source_files]
        ),
        "sequences": sequence_manifest,
        "stats": {**counts, "frame_records": len(rows)},
        "output": {
            "ground_truth": str(Path(output_ground_truth).name),
            "ground_truth_sha256": gt_sha,
        },
    }
    manifest["import_manifest_sha256"] = _canonical_sha256(manifest)
    _write_json(output_manifest, manifest)
    return manifest


def _selected_subdirectories(root: Path, value: str | None) -> list[str]:
    if not root.is_dir():
        raise FileNotFoundError(root)
    available = sorted(item.name for item in root.iterdir() if item.is_dir())
    if value is None or not value.strip():
        if not available:
            raise ValueError(f"No sequence directories found under {root}")
        return available
    selected = sorted({item.strip() for item in value.split(",") if item.strip()})
    missing = [name for name in selected if name not in available]
    if missing:
        raise ValueError(f"Unknown sequence(s) under {root}: {missing}")
    if not selected:
        raise ValueError("No sequences selected")
    return selected


def _validate_mot_image_sequence(
    image_dir: Path,
    *,
    frame_count: int,
    extension: str,
    width: int,
    height: int,
) -> list[Path]:
    import cv2

    files = sorted(
        item
        for item in image_dir.iterdir()
        if item.is_file() and item.suffix.lower() == extension.lower()
    )
    if len(files) != frame_count:
        raise ValueError(
            f"{image_dir}: seqinfo frame_count={frame_count} but found {len(files)} {extension} images"
        )
    for expected_frame, image_path in enumerate(files, start=1):
        try:
            actual_frame = int(image_path.stem)
        except ValueError as exc:
            raise ValueError(f"{image_path}: expected numeric MOT frame filename") from exc
        if actual_frame != expected_frame:
            raise ValueError(
                f"{image_dir}: expected frame {expected_frame:08d}, found {image_path.name}"
            )
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Cannot read sequence image: {image_path}")
        actual_height, actual_width = image.shape[:2]
        if actual_width != width or actual_height != height:
            raise ValueError(
                f"{image_path}: dimensions {actual_width}x{actual_height} differ from "
                f"seqinfo {width}x{height}"
            )
    return files


def import_dancetrack(
    *,
    dataset_root: str | Path,
    output_ground_truth: str | Path,
    output_manifest: str | Path,
    split: str = "val",
    sequences: str | None = None,
    logical_prefix: str | None = None,
    importer_source_commit: str | None = None,
    acknowledge_terms: bool = False,
) -> dict[str, Any]:
    if not acknowledge_terms:
        raise ValueError("DanceTrack import requires --acknowledge-terms")
    if split not in {"train", "val"}:
        raise ValueError("DanceTrack split must be train or val; test GT is not public")

    root = Path(dataset_root)
    split_root = root / split
    selected = _selected_subdirectories(split_root, sequences)
    prefix = logical_prefix or f"golden/public/dancetrack-{split}"

    rows: list[dict[str, Any]] = []
    source_files: list[dict[str, Any]] = []
    sequence_manifest: list[dict[str, Any]] = []
    total_gt_rows = 0
    non_intersecting_omitted = 0

    for sequence_name in selected:
        sequence_dir = split_root / sequence_name
        seqinfo_path = sequence_dir / "seqinfo.ini"
        gt_path = sequence_dir / "gt" / "gt.txt"
        info = _read_seqinfo(seqinfo_path)
        if info["name"] != sequence_name:
            raise ValueError(
                f"{seqinfo_path}: sequence name {info['name']!r} does not match {sequence_name!r}"
            )
        image_dir = sequence_dir / info["im_dir"]
        if not image_dir.is_dir():
            raise FileNotFoundError(image_dir)
        image_files = _validate_mot_image_sequence(
            image_dir,
            frame_count=info["frame_count"],
            extension=info["extension"],
            width=info["width"],
            height=info["height"],
        )
        gt = _parse_mot_gt(gt_path)
        unexpected_frames = sorted(frame for frame in gt if frame > info["frame_count"])
        if unexpected_frames:
            raise ValueError(
                f"{gt_path}: GT contains frame(s) beyond seqLength: {unexpected_frames[:5]}"
            )

        source_files.append(_source_file(seqinfo_path, root, "sequence_metadata"))
        source_files.append(_source_file(gt_path, root, "ground_truth"))
        image_entries = [_source_file(path, root, "image") for path in image_files]
        source_files.extend(image_entries)

        source_rel = _relative_path(image_dir, root)
        logical_video = f"{prefix.rstrip('/')}/{sequence_name}"
        sequence_gt_rows = 0
        sequence_scored = 0
        for frame_number in range(1, info["frame_count"] + 1):
            objects: list[dict[str, Any]] = []
            for annotation in gt.get(frame_number, []):
                sequence_gt_rows += 1
                total_gt_rows += 1
                # DanceTrack uses MOT geometry/layout, but the final 1,1,1 fields are
                # format constants. They are not MOT17 class/visibility semantics.
                if (
                    float(annotation["valid"]) != 1.0
                    or int(annotation["class_id"]) != 1
                    or float(annotation["visibility"]) != 1.0
                ):
                    raise ValueError(
                        f"{gt_path}: DanceTrack GT trailing fields must be exactly 1,1,1"
                    )
                bbox = annotation["bbox"]
                if not _bbox_intersects_image(bbox, info["width"], info["height"]):
                    non_intersecting_omitted += 1
                    continue
                original_id = int(annotation["id"])
                objects.append(
                    {
                        "id": f"DanceTrack:{sequence_name}:{original_id}",
                        "label": "person",
                        "bbox": bbox,
                        "source_annotation": {
                            "dataset": "DanceTrack",
                            "original_id": original_id,
                        },
                    }
                )
                sequence_scored += 1

            rows.append(
                {
                    "video": logical_video,
                    "frame": frame_number - 1,
                    "source": source_rel,
                    "source_frame": frame_number,
                    "source_sequence": sequence_name,
                    "source_fps": info["fps"],
                    "allow_out_of_bounds": True,
                    "tags": [],
                    "objects": objects,
                    "source_metadata": {
                        "dataset": "DanceTrack",
                        "dataset_split": split,
                        "source_sequence": sequence_name,
                        "original_frame": frame_number,
                    },
                }
            )

        sequence_manifest.append(
            {
                "logical_video": logical_video,
                "source_sequence": sequence_name,
                "source": source_rel,
                "fps": info["fps"],
                "frame_count": info["frame_count"],
                "width": info["width"],
                "height": info["height"],
                "gt_rows": sequence_gt_rows,
                "scored_people": sequence_scored,
                "image_files_sha256": _canonical_sha256(
                    [{"path": item["path"], "sha256": item["sha256"]} for item in image_entries]
                ),
            }
        )

    rows.sort(key=lambda item: (item["video"], item["frame"]))
    _write_jsonl(output_ground_truth, rows)
    gt_sha = ground_truth_sha256(output_ground_truth)
    manifest = {
        "schema": IMPORT_SCHEMA,
        "dataset": {
            "name": "DanceTrack",
            "version": "DanceTrack",
            "split": split,
            "terms": DANCETRACK_TERMS,
        },
        "annotation_origin": "official_public_ground_truth",
        "terms_acknowledged": True,
        "tracking_supported": True,
        "importer_source_commit": _importer_commit(importer_source_commit),
        "selected_sequences": selected,
        "conversion_settings": {
            "layout": "MOT-style",
            "canonical_frame_index": "DanceTrack source frame minus 1",
            "canonical_bbox": "MOT 1-based xywh converted to 0-based xyxy without clipping",
            "stable_object_ids": "namespaced by DanceTrack sequence",
            "trailing_fields": "required constant 1,1,1; no MOT17 semantics inherited",
            "semantic_tags": "none inferred automatically",
            "non_intersecting_boxes": "omitted",
        },
        "logical_prefix": prefix.rstrip("/"),
        "source_files": source_files,
        "source_files_sha256": _canonical_sha256(
            [{"path": item["path"], "sha256": item["sha256"]} for item in source_files]
        ),
        "sequences": sequence_manifest,
        "stats": {
            "frame_records": len(rows),
            "ground_truth_rows": total_gt_rows,
            "non_intersecting_boxes_omitted": non_intersecting_omitted,
        },
        "output": {
            "ground_truth": str(Path(output_ground_truth).name),
            "ground_truth_sha256": gt_sha,
        },
    }
    manifest["import_manifest_sha256"] = _canonical_sha256(manifest)
    _write_json(output_manifest, manifest)
    return manifest


def _find_crowdhuman_image(images_dir: Path, image_id: str) -> Path:
    raw = images_dir / image_id
    if raw.is_file():
        return raw
    for extension in (".jpg", ".jpeg", ".png"):
        candidate = images_dir / f"{image_id}{extension}"
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"CrowdHuman image not found for ID {image_id!r} in {images_dir}")


def _read_odgt(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, raw in enumerate(handle, start=1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                item = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
            if not isinstance(item, dict):
                raise ValueError(f"{path}:{line_number}: expected object")
            if not isinstance(item.get("ID"), str) or not item["ID"]:
                raise ValueError(f"{path}:{line_number}: missing ID")
            if not isinstance(item.get("gtboxes"), list):
                raise ValueError(f"{path}:{line_number}: gtboxes must be a list")
            rows.append(item)
    return rows


def _crowdhuman_visible_ratio(gtbox: dict[str, Any]) -> float | None:
    try:
        fbox = _xywh_to_xyxy(gtbox.get("fbox"), "CrowdHuman fbox")
        vbox = _xywh_to_xyxy(gtbox.get("vbox"), "CrowdHuman vbox")
    except ValueError:
        return None
    full_area = max(0.0, fbox[2] - fbox[0]) * max(0.0, fbox[3] - fbox[1])
    visible_area = max(0.0, vbox[2] - vbox[0]) * max(0.0, vbox[3] - vbox[1])
    if full_area <= 0.0:
        return None
    return min(1.0, max(0.0, visible_area / full_area))


def import_crowdhuman(
    *,
    dataset_root: str | Path,
    annotations: str | Path,
    images_dir: str | Path,
    output_ground_truth: str | Path,
    output_manifest: str | Path,
    bbox_kind: str = "full",
    logical_prefix: str = "golden/public/crowdhuman-val",
    importer_source_commit: str | None = None,
    acknowledge_terms: bool = False,
) -> dict[str, Any]:
    if not acknowledge_terms:
        raise ValueError("CrowdHuman import requires --acknowledge-terms")
    if bbox_kind not in {"full", "visible"}:
        raise ValueError("bbox_kind must be full or visible")

    root = Path(dataset_root)
    annotation_path = Path(annotations)
    if not annotation_path.is_absolute():
        annotation_path = root / annotation_path
    if annotation_path.name.lower() != "annotation_val.odgt":
        raise ValueError("CrowdHuman detector evaluation importer accepts validation split annotation_val.odgt only")
    image_root = Path(images_dir)
    if not image_root.is_absolute():
        image_root = root / image_root
    if not image_root.is_dir():
        raise FileNotFoundError(image_root)

    records = _read_odgt(annotation_path)
    source_files = [_source_file(annotation_path, root, "ground_truth")]
    rows: list[dict[str, Any]] = []
    stats = {
        "images": 0,
        "scored_people": 0,
        "ignored_people_or_masks": 0,
        "non_intersecting_target_like_omitted": 0,
    }
    bbox_key = "fbox" if bbox_kind == "full" else "vbox"

    import cv2

    for record in records:
        image_id = record["ID"]
        image_path = _find_crowdhuman_image(image_root, image_id)
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Cannot read CrowdHuman image: {image_path}")
        height, width = image.shape[:2]
        source_entry = _source_file(image_path, root, "image")
        source_files.append(source_entry)
        source_rel = source_entry["path"]
        objects: list[dict[str, Any]] = []

        for index, gtbox in enumerate(record["gtboxes"]):
            if not isinstance(gtbox, dict):
                raise ValueError(f"CrowdHuman {image_id}: gtboxes[{index}] must be an object")
            tag = gtbox.get("tag")
            if tag not in {"person", "mask"}:
                raise ValueError(f"CrowdHuman {image_id}: unsupported tag {tag!r}")
            bbox = _xywh_to_xyxy(gtbox.get(bbox_key), f"CrowdHuman {image_id} {bbox_key}")
            if not _bbox_intersects_image(bbox, width, height):
                stats["non_intersecting_target_like_omitted"] += 1
                continue
            extra = gtbox.get("extra") if isinstance(gtbox.get("extra"), dict) else {}
            ignore = tag == "mask" or bool(extra.get("ignore", 0))
            if ignore:
                stats["ignored_people_or_masks"] += 1
            else:
                stats["scored_people"] += 1

            visible_ratio = _crowdhuman_visible_ratio(gtbox)
            attributes: list[str] = []
            if visible_ratio is not None and not ignore:
                attributes.append(f"crowdhuman_{_visibility_attribute(visible_ratio)}")
            occ = extra.get("occ")
            if occ is not None and not ignore:
                if isinstance(occ, bool) or not isinstance(occ, int):
                    raise ValueError(f"CrowdHuman {image_id}: extra.occ must be an integer when present")
                attributes.append(f"crowdhuman_occ_{occ}")
            source_annotation = {
                "dataset": "CrowdHuman",
                "tag": tag,
                "box_id": extra.get("box_id"),
                "occ": extra.get("occ"),
                "visible_ratio": visible_ratio,
                "bbox_policy": bbox_key,
                "fbox": gtbox.get("fbox"),
                "vbox": gtbox.get("vbox"),
            }
            obj: dict[str, Any] = {
                "label": "person",
                "bbox": bbox,
                "attributes": attributes,
                "source_annotation": source_annotation,
            }
            if ignore:
                obj["ignore"] = True
            # CrowdHuman validation images are independent samples, not trajectories:
            # deliberately omit canonical stable object IDs so tracking metrics remain disabled.
            objects.append(obj)

        rows.append(
            {
                "video": f"{logical_prefix.rstrip('/')}/{image_path.name}",
                "frame": 0,
                "source": source_rel,
                "source_sequence": "CrowdHuman-val",
                "allow_out_of_bounds": True,
                "tags": [],
                "objects": objects,
                "source_metadata": {
                    "dataset": "CrowdHuman",
                    "dataset_split": "validation",
                    "image_id": image_id,
                    "width": int(width),
                    "height": int(height),
                },
            }
        )
        stats["images"] += 1

    rows.sort(key=lambda item: item["video"])
    _write_jsonl(output_ground_truth, rows)
    gt_sha = ground_truth_sha256(output_ground_truth)
    manifest = {
        "schema": IMPORT_SCHEMA,
        "dataset": {
            "name": "CrowdHuman",
            "version": "CrowdHuman",
            "split": "validation",
            "terms": CROWDHUMAN_TERMS,
        },
        "annotation_origin": "official_public_ground_truth",
        "terms_acknowledged": True,
        "tracking_supported": False,
        "importer_source_commit": _importer_commit(importer_source_commit),
        "selected_sequences": ["validation"],
        "conversion_settings": {
            "bbox_kind": bbox_kind,
            "bbox_source_field": bbox_key,
            "bbox_coordinates": "source xywh converted to xyxy without clipping",
            "non_intersecting_target_like_boxes": "omitted",
            "ignore": "tag=mask OR extra.ignore=1",
            "stable_object_ids": "omitted because validation images are independent samples",
            "visible_ratio_attributes": [0.25, 0.50, 0.75],
            "semantic_tags": "none inferred automatically",
        },
        "logical_prefix": logical_prefix.rstrip("/"),
        "source_files": source_files,
        "source_files_sha256": _canonical_sha256(
            [{"path": item["path"], "sha256": item["sha256"]} for item in source_files]
        ),
        "stats": stats,
        "output": {
            "ground_truth": str(Path(output_ground_truth).name),
            "ground_truth_sha256": gt_sha,
        },
    }
    manifest["import_manifest_sha256"] = _canonical_sha256(manifest)
    _write_json(output_manifest, manifest)
    return manifest


def _normalize_category_name(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("NightOwls category name must be non-empty")
    return value.strip().lower().replace("-", "").replace("_", "").replace(" ", "")


def _stable_scalar_token(value: Any, where: str) -> str:
    if isinstance(value, bool):
        raise ValueError(f"{where}: boolean is not a valid stable scalar id")
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float) and math.isfinite(value) and value.is_integer():
        return str(int(value))
    if isinstance(value, str) and value.strip():
        return value.strip()
    raise ValueError(f"{where}: expected stable integer-like or string value")


def _nightowls_pose_map(data: dict[str, Any]) -> dict[int, str]:
    poses = data.get("poses", [])
    if poses is None:
        return {}
    if not isinstance(poses, list):
        raise ValueError("NightOwls poses must be a list when present")
    result: dict[int, str] = {}
    for index, pose in enumerate(poses, start=1):
        if not isinstance(pose, dict):
            raise ValueError("NightOwls pose entry must be an object")
        pose_id = pose.get("id", index)
        if not isinstance(pose_id, int) or isinstance(pose_id, bool):
            raise ValueError("NightOwls pose id must be an integer")
        name = pose.get("name")
        if not isinstance(name, str) or not name.strip():
            raise ValueError("NightOwls pose name must be non-empty")
        result[pose_id] = name.strip()
    return result


def _nightowls_official_sdk_files(sdk_root: Path) -> list[Path]:
    expected = [
        sdk_root / "README.md",
        sdk_root / "python" / "coco.py",
        sdk_root / "python" / "eval.py",
        sdk_root / "python" / "eval_MR_multisetup.py",
    ]
    missing = [path for path in expected if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"NightOwls official SDK files missing: {missing}")
    readme = expected[0].read_text(encoding="utf-8", errors="replace")
    evaluator = expected[3].read_text(encoding="utf-8", errors="replace")
    if "NightOwls API" not in readme or "non-commercial" not in readme:
        raise ValueError("NightOwls SDK README does not match the official API/license text")
    if "catIds = [1]" not in evaluator:
        raise ValueError("NightOwls SDK evaluator does not expose the expected pedestrian category 1")
    return expected


def _nightowls_size_bin(height: float) -> str:
    if height < 24:
        return "lt24"
    if height < 48:
        return "24_47"
    if height < 96:
        return "48_95"
    return "ge96"


def _nightowls_image_strata(
    image: dict[str, Any],
    annotations: list[dict[str, Any]],
    *,
    pedestrian_category_id: int,
) -> set[str]:
    scored = [
        ann
        for ann in annotations
        if ann.get("category_id") == pedestrian_category_id and not bool(ann.get("ignore", 0))
    ]
    strata = {"positive" if scored else "negative"}
    daytime = image.get("daytime")
    if isinstance(daytime, str) and daytime.strip():
        strata.add(f"daytime:{daytime.strip().lower()}")
    for ann in scored:
        bbox = _xywh_to_xyxy(ann.get("bbox"), "NightOwls slice bbox")
        strata.add(f"size:{_nightowls_size_bin(bbox[3] - bbox[1])}")
        if ann.get("occluded") is True:
            strata.add("occluded:true")
        if ann.get("difficult") is True:
            strata.add("difficult:true")
        pose_id = ann.get("pose_id")
        if isinstance(pose_id, int) and not isinstance(pose_id, bool):
            strata.add(f"pose:{pose_id}")
    return strata


def _deterministic_nightowls_slice(
    images: list[dict[str, Any]],
    annotations_by_image: dict[int, list[dict[str, Any]]],
    *,
    pedestrian_category_id: int,
    limit: int,
    seed: str,
) -> list[dict[str, Any]]:
    if limit <= 0:
        raise ValueError("NightOwls slice frame count must be > 0")
    if limit >= len(images):
        return list(images)

    def rank(image: dict[str, Any]) -> str:
        image_id = image.get("id")
        payload = f"{seed}:{image_id}:{image.get('file_name', '')}".encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    ordered = sorted(images, key=lambda image: (rank(image), int(image["id"])))
    best_for_stratum: dict[str, dict[str, Any]] = {}
    for image in ordered:
        image_id = int(image["id"])
        for stratum in _nightowls_image_strata(
            image,
            annotations_by_image.get(image_id, []),
            pedestrian_category_id=pedestrian_category_id,
        ):
            best_for_stratum.setdefault(stratum, image)

    selected_by_id = {int(image["id"]): image for image in best_for_stratum.values()}
    if len(selected_by_id) > limit:
        raise ValueError(
            f"NightOwls slice size {limit} is too small for deterministic source strata; "
            f"minimum is {len(selected_by_id)}"
        )
    for image in ordered:
        if len(selected_by_id) >= limit:
            break
        selected_by_id.setdefault(int(image["id"]), image)
    return sorted(selected_by_id.values(), key=lambda image: int(image["id"]))


def import_nightowls(
    *,
    dataset_root: str | Path,
    annotations: str | Path,
    images_dir: str | Path,
    sdk_dir: str | Path,
    output_ground_truth: str | Path,
    output_manifest: str | Path,
    logical_prefix: str = "golden/public/nightowls-val",
    slice_frames: int | None = None,
    slice_seed: str = "spectratrack-round2-nightowls-v1",
    importer_source_commit: str | None = None,
    acknowledge_terms: bool = False,
) -> dict[str, Any]:
    if not acknowledge_terms:
        raise ValueError("NightOwls import requires --acknowledge-terms")
    root = Path(dataset_root)
    annotation_path = Path(annotations)
    if not annotation_path.is_absolute():
        annotation_path = root / annotation_path
    if annotation_path.name != "nightowls_validation.json":
        raise ValueError("Round-2 NightOwls importer accepts official nightowls_validation.json only")
    image_root = Path(images_dir)
    if not image_root.is_absolute():
        image_root = root / image_root
    sdk_root = Path(sdk_dir)
    if not sdk_root.is_absolute():
        sdk_root = root / sdk_root
    if not image_root.is_dir():
        raise FileNotFoundError(image_root)

    sdk_files = _nightowls_official_sdk_files(sdk_root)
    try:
        data = json.loads(annotation_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{annotation_path}: invalid JSON") from exc
    if not isinstance(data, dict):
        raise ValueError("NightOwls annotation JSON must be an object")
    images = data.get("images")
    annotations_list = data.get("annotations")
    categories = data.get("categories")
    if not isinstance(images, list) or not isinstance(annotations_list, list) or not isinstance(categories, list):
        raise ValueError("NightOwls JSON requires images, annotations and categories lists")

    category_names: dict[int, str] = {}
    for category in categories:
        if not isinstance(category, dict):
            raise ValueError("NightOwls category must be an object")
        category_id = category.get("id")
        if not isinstance(category_id, int) or isinstance(category_id, bool):
            raise ValueError("NightOwls category id must be an integer")
        category_names[category_id] = _normalize_category_name(category.get("name"))
    if category_names.get(1) != "pedestrian":
        raise ValueError(
            f"NightOwls official pedestrian category must be id=1/name=pedestrian; got {category_names.get(1)!r}"
        )
    ignore_category_ids = {category_id for category_id, name in category_names.items() if name == "ignore"}
    pose_names = _nightowls_pose_map(data)

    images_by_id: dict[int, dict[str, Any]] = {}
    for image in images:
        if not isinstance(image, dict):
            raise ValueError("NightOwls image record must be an object")
        image_id = image.get("id")
        if not isinstance(image_id, int) or isinstance(image_id, bool):
            raise ValueError("NightOwls image id must be an integer")
        if image_id in images_by_id:
            raise ValueError(f"NightOwls duplicate image id {image_id}")
        file_name = image.get("file_name")
        width = image.get("width")
        height = image.get("height")
        if not isinstance(file_name, str) or not file_name:
            raise ValueError(f"NightOwls image {image_id}: file_name must be non-empty")
        if not isinstance(width, int) or width <= 0 or not isinstance(height, int) or height <= 0:
            raise ValueError(f"NightOwls image {image_id}: invalid dimensions")
        _stable_scalar_token(image.get("recordings_id"), f"NightOwls image {image_id} recordings_id")
        timestamp = image.get("timestamp")
        if (
            isinstance(timestamp, bool)
            or not isinstance(timestamp, (int, float))
            or not math.isfinite(float(timestamp))
        ):
            raise ValueError(f"NightOwls image {image_id}: timestamp must be finite")
        images_by_id[image_id] = image

    annotations_by_image: dict[int, list[dict[str, Any]]] = {}
    scored_pedestrian_annotations: list[dict[str, Any]] = []
    trajectory_frames: dict[tuple[str, str], set[int]] = {}
    tracking_fields_valid = True
    for annotation in annotations_list:
        if not isinstance(annotation, dict):
            raise ValueError("NightOwls annotation must be an object")
        image_id = annotation.get("image_id")
        if image_id not in images_by_id:
            raise ValueError(f"NightOwls annotation references unknown image_id {image_id!r}")
        category_id = annotation.get("category_id")
        if category_id not in category_names:
            raise ValueError(f"NightOwls annotation uses unknown category_id {category_id!r}")
        _xywh_to_xyxy(annotation.get("bbox"), f"NightOwls image {image_id} bbox")
        annotations_by_image.setdefault(int(image_id), []).append(annotation)
        if category_id == 1 and not bool(annotation.get("ignore", 0)):
            scored_pedestrian_annotations.append(annotation)
            tracking_id = annotation.get("tracking_id")
            if (
                isinstance(tracking_id, bool)
                or not isinstance(tracking_id, int)
                or tracking_id < 0
            ):
                tracking_fields_valid = False
            else:
                image = images_by_id[int(image_id)]
                recording = _stable_scalar_token(
                    image.get("recordings_id"),
                    f"NightOwls image {image_id} recordings_id",
                )
                key = (recording, str(tracking_id))
                frames = trajectory_frames.setdefault(key, set())
                if int(image_id) in frames:
                    raise ValueError(
                        f"NightOwls duplicate tracking_id {tracking_id} in image {image_id}"
                    )
                frames.add(int(image_id))

    repeated_trajectory = any(len(frame_ids) >= 2 for frame_ids in trajectory_frames.values())
    full_tracking_supported = bool(scored_pedestrian_annotations) and tracking_fields_valid and repeated_trajectory

    selected_images = (
        _deterministic_nightowls_slice(
            images,
            annotations_by_image,
            pedestrian_category_id=1,
            limit=slice_frames,
            seed=slice_seed,
        )
        if slice_frames is not None
        else list(images)
    )
    selected_ids = {int(image["id"]) for image in selected_images}
    tracking_supported = full_tracking_supported and slice_frames is None

    by_recording: dict[str, list[dict[str, Any]]] = {}
    for image in selected_images:
        recording = _stable_scalar_token(
            image.get("recordings_id"),
            f"NightOwls image {image.get('id')} recordings_id",
        )
        by_recording.setdefault(recording, []).append(image)
    for recording_images in by_recording.values():
        recording_images.sort(key=lambda item: (float(item["timestamp"]), int(item["id"])))

    source_files = [_source_file(annotation_path, root, "ground_truth")]
    for sdk_file in sdk_files:
        source_files.append(_source_file(sdk_file, root, "official_sdk"))

    rows: list[dict[str, Any]] = []
    sequence_manifest: list[dict[str, Any]] = []
    stats = {
        "selected_images": len(selected_images),
        "scored_pedestrians": 0,
        "ignored_pedestrians": 0,
        "official_ignore_regions": 0,
        "rider_or_other_classes_omitted": 0,
        "non_intersecting_target_like_omitted": 0,
    }

    import cv2

    for recording, recording_images in sorted(by_recording.items()):
        logical_video = f"{logical_prefix.rstrip('/')}/recording-{recording}"
        sequence_source_entries: list[dict[str, Any]] = []
        for canonical_frame, image in enumerate(recording_images):
            image_id = int(image["id"])
            image_path = image_root / str(image["file_name"])
            source_entry = _source_file(image_path, root, "image")
            source_files.append(source_entry)
            sequence_source_entries.append(source_entry)
            loaded = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if loaded is None:
                raise RuntimeError(f"Cannot read NightOwls image: {image_path}")
            actual_height, actual_width = loaded.shape[:2]
            if actual_width != int(image["width"]) or actual_height != int(image["height"]):
                raise ValueError(
                    f"NightOwls image {image_id}: file dimensions {actual_width}x{actual_height} "
                    f"differ from JSON {image['width']}x{image['height']}"
                )

            objects: list[dict[str, Any]] = []
            for annotation in annotations_by_image.get(image_id, []):
                category_id = int(annotation["category_id"])
                bbox = _xywh_to_xyxy(annotation["bbox"], f"NightOwls image {image_id} bbox")
                if not _bbox_intersects_image(bbox, actual_width, actual_height):
                    if category_id == 1 or category_id in ignore_category_ids:
                        stats["non_intersecting_target_like_omitted"] += 1
                    continue

                source_annotation = {
                    "dataset": "NightOwls",
                    "annotation_id": annotation.get("id"),
                    "category_id": category_id,
                    "category_name": category_names[category_id],
                    "tracking_id": annotation.get("tracking_id"),
                    "occluded": annotation.get("occluded"),
                    "difficult": annotation.get("difficult"),
                    "pose_id": annotation.get("pose_id"),
                    "truncated": annotation.get("truncated"),
                    "ignore": annotation.get("ignore"),
                    "area": annotation.get("area"),
                }

                if category_id == 1:
                    ignored = bool(annotation.get("ignore", 0))
                    attributes: list[str] = []
                    for field_name in ("occluded", "difficult", "truncated"):
                        value = annotation.get(field_name)
                        if value is not None:
                            if not isinstance(value, bool):
                                raise ValueError(
                                    f"NightOwls pedestrian {annotation.get('id')}: "
                                    f"{field_name} must be boolean/null"
                                )
                            attributes.append(f"nightowls_{field_name}_{str(value).lower()}")
                    pose_id = annotation.get("pose_id")
                    if pose_id is not None:
                        if not isinstance(pose_id, int) or isinstance(pose_id, bool):
                            raise ValueError("NightOwls pedestrian pose_id must be integer/null")
                        pose_name = pose_names.get(pose_id)
                        attributes.append(
                            f"nightowls_pose_{_normalize_token_for_attribute(pose_name or str(pose_id))}"
                        )
                    obj: dict[str, Any] = {
                        "label": "person",
                        "bbox": bbox,
                        "attributes": attributes,
                        "source_annotation": source_annotation,
                    }
                    if tracking_supported and not ignored:
                        obj["id"] = (
                            f"NightOwls:{recording}:{int(annotation['tracking_id'])}"
                        )
                    if ignored:
                        obj["ignore"] = True
                        stats["ignored_pedestrians"] += 1
                    else:
                        stats["scored_pedestrians"] += 1
                    objects.append(obj)
                elif category_id in ignore_category_ids:
                    stats["official_ignore_regions"] += 1
                    objects.append(
                        {
                            "label": "person",
                            "bbox": bbox,
                            "ignore": True,
                            "attributes": ["nightowls_official_ignore_region"],
                            "source_annotation": source_annotation,
                        }
                    )
                else:
                    # The official pedestrian evaluator accumulates category 1 only.
                    # Rider classes remain separate and do not become pedestrian targets.
                    stats["rider_or_other_classes_omitted"] += 1

            daytime = image.get("daytime")
            tags = ["night_dark"] if isinstance(daytime, str) and daytime.lower() == "night" else []
            rows.append(
                {
                    "video": logical_video,
                    "frame": canonical_frame,
                    "source": source_entry["path"],
                    "source_frame": image_id,
                    "source_sequence": f"NightOwls:recording-{recording}",
                    "allow_out_of_bounds": True,
                    "tags": tags,
                    "objects": objects,
                    "source_metadata": {
                        "dataset": "NightOwls",
                        "dataset_split": "validation",
                        "recordings_id": image.get("recordings_id"),
                        "image_id": image_id,
                        "timestamp": image.get("timestamp"),
                        "daytime": daytime,
                        "file_name": image.get("file_name"),
                    },
                }
            )

        sequence_manifest.append(
            {
                "logical_video": logical_video,
                "recordings_id": recording,
                "frame_count": len(recording_images),
                "first_timestamp": recording_images[0]["timestamp"],
                "last_timestamp": recording_images[-1]["timestamp"],
                "source_images_sha256": _canonical_sha256(
                    [
                        {"path": item["path"], "sha256": item["sha256"]}
                        for item in sequence_source_entries
                    ]
                ),
            }
        )

    rows.sort(key=lambda item: (item["video"], item["frame"]))
    _write_jsonl(output_ground_truth, rows)
    gt_sha = ground_truth_sha256(output_ground_truth)
    selected_id_list = sorted(selected_ids)
    slice_info = (
        {
            "kind": "deterministic_stratified_round2",
            "frame_limit": slice_frames,
            "seed": slice_seed,
            "selection_inputs": (
                "official image metadata + official pedestrian annotations only; no model output"
            ),
            "selected_image_ids": selected_id_list,
            "selected_image_ids_sha256": _canonical_sha256(selected_id_list),
            "tracking_supported": False,
        }
        if slice_frames is not None
        else None
    )
    manifest = {
        "schema": IMPORT_SCHEMA,
        "dataset": {
            "name": "NightOwls",
            "version": "NightOwls",
            "split": "validation",
            "terms": NIGHTOWLS_TERMS,
        },
        "annotation_origin": "official_public_ground_truth",
        "terms_acknowledged": True,
        "tracking_supported": tracking_supported,
        "tracking_contract": {
            "official_documentation_states_tracking_information": True,
            "all_scored_pedestrians_have_valid_tracking_id": tracking_fields_valid,
            "repeated_trajectory_observed": repeated_trajectory,
            "disabled_for_stratified_slice": slice_frames is not None,
        },
        "importer_source_commit": _importer_commit(importer_source_commit),
        "selected_sequences": sorted(by_recording),
        "conversion_settings": {
            "scored_category": {"id": 1, "name": "pedestrian"},
            "official_ignore_categories": sorted(ignore_category_ids),
            "rider_and_other_categories": "omitted; never relabeled as pedestrian",
            "pedestrian_ignore_flag": "preserved as canonical ignore",
            "bbox_coordinates": "official xywh converted to xyxy without clipping",
            "stable_object_ids": (
                "NightOwls:<recording>:<tracking_id>"
                if tracking_supported
                else "omitted because stable temporal scoring was not validated for this import"
            ),
            "attributes": ["occluded", "difficult", "pose", "truncated"],
            "semantic_tags": "night_dark only when official image daytime metadata equals night",
        },
        "slice": slice_info,
        "logical_prefix": logical_prefix.rstrip("/"),
        "source_files": source_files,
        "source_files_sha256": _canonical_sha256(
            [{"path": item["path"], "sha256": item["sha256"]} for item in source_files]
        ),
        "sequences": sequence_manifest,
        "stats": stats,
        "output": {
            "ground_truth": str(Path(output_ground_truth).name),
            "ground_truth_sha256": gt_sha,
        },
    }
    manifest["import_manifest_sha256"] = _canonical_sha256(manifest)
    _write_json(output_manifest, manifest)
    return manifest


def _normalize_token_for_attribute(value: str) -> str:
    token = value.strip().lower().replace("-", "_").replace(" ", "_").replace("/", "_")
    while "__" in token:
        token = token.replace("__", "_")
    return token or "unknown"


def load_import_manifest(path: str | Path) -> dict[str, Any]:
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    if manifest.get("schema") != IMPORT_SCHEMA:
        raise ValueError(f"{path}: unsupported public dataset import schema")
    expected = manifest.get("import_manifest_sha256")
    unsigned = dict(manifest)
    unsigned.pop("import_manifest_sha256", None)
    actual = _canonical_sha256(unsigned)
    if expected != actual:
        raise ValueError(f"{path}: import manifest hash mismatch")
    commit = manifest.get("importer_source_commit")
    if not isinstance(commit, str) or not _is_hex(commit, 40):
        raise ValueError(f"{path}: importer_source_commit must be full 40-hex SHA")
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SpectraTrack isolated public pedestrian dataset importers")
    sub = parser.add_subparsers(dest="command", required=True)

    mot = sub.add_parser("import-mot17")
    mot.add_argument("--dataset-root", required=True)
    mot.add_argument("--output", required=True)
    mot.add_argument("--manifest", required=True)
    mot.add_argument("--variant", default="FRCNN", choices=("DPM", "FRCNN", "SDP"))
    mot.add_argument("--sequences", help="Comma-separated base sequences, e.g. 02,04,10")
    mot.add_argument("--logical-prefix", default="golden/public/mot17")
    mot.add_argument("--importer-source-commit")
    mot.add_argument("--acknowledge-terms", action="store_true")

    dance = sub.add_parser("import-dancetrack")
    dance.add_argument("--dataset-root", required=True)
    dance.add_argument("--output", required=True)
    dance.add_argument("--manifest", required=True)
    dance.add_argument("--split", choices=("train", "val"), default="val")
    dance.add_argument("--sequences")
    dance.add_argument("--logical-prefix")
    dance.add_argument("--importer-source-commit")
    dance.add_argument("--acknowledge-terms", action="store_true")

    night = sub.add_parser("import-nightowls")
    night.add_argument("--dataset-root", required=True)
    night.add_argument("--annotations", default="nightowls_validation.json")
    night.add_argument("--images-dir", required=True)
    night.add_argument("--sdk-dir", required=True)
    night.add_argument("--output", required=True)
    night.add_argument("--manifest", required=True)
    night.add_argument("--logical-prefix", default="golden/public/nightowls-val")
    night.add_argument("--slice-frames", type=int)
    night.add_argument("--slice-seed", default="spectratrack-round2-nightowls-v1")
    night.add_argument("--importer-source-commit")
    night.add_argument("--acknowledge-terms", action="store_true")

    crowd = sub.add_parser("import-crowdhuman")
    crowd.add_argument("--dataset-root", required=True)
    crowd.add_argument("--annotations", default="annotation_val.odgt")
    crowd.add_argument("--images-dir", required=True)
    crowd.add_argument("--output", required=True)
    crowd.add_argument("--manifest", required=True)
    crowd.add_argument("--bbox-kind", choices=("full", "visible"), default="full")
    crowd.add_argument("--logical-prefix", default="golden/public/crowdhuman-val")
    crowd.add_argument("--importer-source-commit")
    crowd.add_argument("--acknowledge-terms", action="store_true")
    return parser


def main() -> int:
    args = _parser().parse_args()
    try:
        if args.command == "import-mot17":
            result = import_mot17(
                dataset_root=args.dataset_root,
                output_ground_truth=args.output,
                output_manifest=args.manifest,
                detector_variant=args.variant,
                sequences=args.sequences,
                logical_prefix=args.logical_prefix,
                importer_source_commit=args.importer_source_commit,
                acknowledge_terms=args.acknowledge_terms,
            )
        elif args.command == "import-dancetrack":
            result = import_dancetrack(
                dataset_root=args.dataset_root,
                output_ground_truth=args.output,
                output_manifest=args.manifest,
                split=args.split,
                sequences=args.sequences,
                logical_prefix=args.logical_prefix,
                importer_source_commit=args.importer_source_commit,
                acknowledge_terms=args.acknowledge_terms,
            )
        elif args.command == "import-nightowls":
            result = import_nightowls(
                dataset_root=args.dataset_root,
                annotations=args.annotations,
                images_dir=args.images_dir,
                sdk_dir=args.sdk_dir,
                output_ground_truth=args.output,
                output_manifest=args.manifest,
                logical_prefix=args.logical_prefix,
                slice_frames=args.slice_frames,
                slice_seed=args.slice_seed,
                importer_source_commit=args.importer_source_commit,
                acknowledge_terms=args.acknowledge_terms,
            )
        else:
            result = import_crowdhuman(
                dataset_root=args.dataset_root,
                annotations=args.annotations,
                images_dir=args.images_dir,
                output_ground_truth=args.output,
                output_manifest=args.manifest,
                bbox_kind=args.bbox_kind,
                logical_prefix=args.logical_prefix,
                importer_source_commit=args.importer_source_commit,
                acknowledge_terms=args.acknowledge_terms,
            )
        print(
            f"dataset={result['dataset']['name']} split={result['dataset']['split']} "
            f"gt_sha256={result['output']['ground_truth_sha256']} "
            f"import_manifest_sha256={result['import_manifest_sha256']}"
        )
        return 0
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
