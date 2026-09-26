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
