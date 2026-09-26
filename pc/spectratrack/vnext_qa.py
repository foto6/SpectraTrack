from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable

from .integrity import sha256_file
from .qa_benchmark import (
    SCHEMA_VERSION,
    ground_truth_sha256,
    inspect_qa_source,
    load_ground_truth,
)

CORPUS_SCHEMA = "spectratrack-cctv-corpus-v1"
FRAME_BATCH_SCHEMA = "spectratrack-cctv-frame-batch-v1"
REPLAY_SCHEMA = "spectratrack-detection-replay-v1"
STAMPED_RUN_SCHEMA = "spectratrack-vnext-run-v1"
EVIDENCE_SCHEMA = "spectratrack-vnext-evidence-v1"
LEADERBOARD_SCHEMA = "spectratrack-vnext-leaderboard-v1"

REQUIRED_GOLDEN_COVERAGE = (
    "tiny_person",
    "distant_person",
    "normal_person",
    "partial_occlusion",
    "heavy_occlusion",
    "night_dark",
    "motion_blur",
    "compression",
    "high_angle_cctv",
    "crossing_people",
    "camera_motion",
    "negative",
)

_COVERAGE_ALIASES = {
    "tiny": "tiny_person",
    "small": "tiny_person",
    "small_person": "tiny_person",
    "small_people": "tiny_person",
    "distant": "distant_person",
    "normal": "normal_person",
    "partial": "partial_occlusion",
    "partially_occluded": "partial_occlusion",
    "heavy": "heavy_occlusion",
    "heavily_occluded": "heavy_occlusion",
    "dark": "night_dark",
    "night": "night_dark",
    "low_light": "night_dark",
    "blur": "motion_blur",
    "motionblur": "motion_blur",
    "high_angle": "high_angle_cctv",
    "high_angle_camera": "high_angle_cctv",
    "crossing": "crossing_people",
    "people_crossing": "crossing_people",
    "moving_camera": "camera_motion",
    "camera_movement": "camera_motion",
    "negative_frame": "negative",
    "no_people": "negative",
}


def _normalize_token(value: str) -> str:
    token = value.strip().lower().replace("-", "_").replace(" ", "_").replace("/", "_")
    while "__" in token:
        token = token.replace("__", "_")
    return _COVERAGE_ALIASES.get(token, token)


def _canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _json_dump(path: str | Path, payload: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _json_load(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _is_hex(value: Any, length: int) -> bool:
    if not isinstance(value, str) or len(value) != length:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def _coverage_for_frame(frame) -> set[str]:
    frame_tags = {_normalize_token(tag) for tag in frame.tags}
    people = [obj for obj in frame.objects if obj.label == "person"]
    valid_people = [obj for obj in people if not obj.ignore]

    coverage = {
        tag
        for tag in frame_tags
        if tag
        in {
            "night_dark",
            "motion_blur",
            "compression",
            "high_angle_cctv",
            "camera_motion",
        }
    }
    if not people:
        coverage.add("negative")

    person_categories = {
        "tiny_person",
        "distant_person",
        "normal_person",
        "partial_occlusion",
        "heavy_occlusion",
    }
    if valid_people:
        coverage.update(frame_tags & person_categories)
    if len(valid_people) >= 2 and "crossing_people" in frame_tags:
        coverage.add("crossing_people")

    for obj in valid_people:
        object_tags = {_normalize_token(attribute) for attribute in obj.attributes}
        coverage.update(object_tags & person_categories)
        if obj.bbox[3] - obj.bbox[1] < 24:
            coverage.add("tiny_person")
    return coverage


def inspect_split(
    ground_truth_path: str | Path,
    video_root: str | Path,
    split: str,
    *,
    allowed_labels: Iterable[str] = ("person",),
    coverage_profile: str = "private-cctv",
) -> dict[str, Any]:
    if split not in {"train", "golden"}:
        raise ValueError("split must be train or golden")
    if coverage_profile not in {"private-cctv", "public-dataset"}:
        raise ValueError("coverage_profile must be private-cctv or public-dataset")
    frames = load_ground_truth(ground_truth_path)
    root = Path(video_root)
    allowed = set(allowed_labels)
    errors: list[str] = []
    warnings: list[str] = []
    grouped: dict[str, list[Any]] = defaultdict(list)
    coverage: dict[str, int] = defaultdict(int)

    for frame in frames:
        grouped[frame.video].append(frame)
        expected_prefix = f"{split}/"
        if not frame.video.replace("\\", "/").startswith(expected_prefix):
            errors.append(
                f"{frame.video}#{frame.frame}: {split} GT must reference videos under {expected_prefix}"
            )
        for obj in frame.objects:
            if obj.label not in allowed:
                errors.append(f"{frame.video}#{frame.frame}: invalid label {obj.label!r}")
        if split == "golden":
            for category in _coverage_for_frame(frame):
                if category in REQUIRED_GOLDEN_COVERAGE:
                    coverage[category] += 1

    videos: list[dict[str, Any]] = []
    for video, video_frames in sorted(grouped.items()):
        source_names = {frame.source or frame.video for frame in video_frames}
        source_fps_values = {frame.source_fps for frame in video_frames if frame.source_fps is not None}
        if len(source_names) != 1:
            errors.append(f"{video}: multiple physical sources declared: {sorted(source_names)}")
            continue
        if len(source_fps_values) > 1:
            errors.append(f"{video}: conflicting source_fps values")
            continue
        source_name = next(iter(source_names))
        source_path = root / source_name
        if not source_path.exists():
            errors.append(f"missing source: {source_name}")
            continue
        try:
            meta = inspect_qa_source(
                source_path,
                fps_override=next(iter(source_fps_values)) if source_fps_values else None,
            )
        except (FileNotFoundError, RuntimeError) as exc:
            errors.append(str(exc))
            continue

        width = int(meta["width"])
        height = int(meta["height"])
        max_annotated_frame = max(frame.frame for frame in video_frames)
        frame_count = meta.get("frame_count")
        if frame_count is not None and max_annotated_frame >= int(frame_count):
            errors.append(
                f"{video}: annotated frame {max_annotated_frame} is outside frame_count={frame_count}"
            )

        valid_people = 0
        for frame in video_frames:
            for obj in frame.objects:
                x1, y1, x2, y2 = obj.bbox
                outside = x1 < 0.0 or y1 < 0.0 or x2 > width or y2 > height
                if outside and not frame.allow_out_of_bounds:
                    errors.append(
                        f"{video}#{frame.frame}: bbox {obj.bbox} outside {width}x{height}"
                    )
                if frame.allow_out_of_bounds and (
                    x2 <= 0.0 or y2 <= 0.0 or x1 >= width or y1 >= height
                ):
                    errors.append(
                        f"{video}#{frame.frame}: bbox {obj.bbox} does not intersect {width}x{height}"
                    )
                if obj.label == "person" and not obj.ignore:
                    valid_people += 1

        videos.append(
            {
                "video": video,
                "source": source_name,
                "source_kind": meta["kind"],
                "split": split,
                "sha256": meta["sha256"],
                "width": width,
                "height": height,
                "fps": meta.get("fps"),
                "frame_count": frame_count,
                "annotated_frames": len(video_frames),
                "valid_person_instances": valid_people,
            }
        )

    if split == "golden" and coverage_profile == "private-cctv":
        missing_coverage = [name for name in REQUIRED_GOLDEN_COVERAGE if coverage.get(name, 0) == 0]
        if missing_coverage:
            errors.append("golden coverage missing: " + ", ".join(missing_coverage))
        if not any(video["valid_person_instances"] == 0 for video in videos) and coverage.get("negative", 0) == 0:
            warnings.append("golden set has no fully negative video; annotated negative frames are still accepted")

    return {
        "split": split,
        "ground_truth": Path(ground_truth_path).name,
        "ground_truth_sha256": ground_truth_sha256(ground_truth_path),
        "frame_records": len(frames),
        "videos": videos,
        "coverage": {name: int(coverage.get(name, 0)) for name in REQUIRED_GOLDEN_COVERAGE}
        if split == "golden"
        else {},
        "errors": errors,
        "coverage_profile": coverage_profile,
        "warnings": warnings,
    }


def _validate_dataset_import_manifests(
    paths: Iterable[str | Path],
    *,
    video_root: str | Path,
    golden_ground_truth_sha256: str,
    train_ground_truth_sha256: str | None,
) -> list[dict[str, Any]]:
    from .public_dataset_import import load_import_manifest

    root = Path(video_root)
    validated: list[dict[str, Any]] = []
    for path in paths:
        manifest = load_import_manifest(path)
        output_sha = manifest.get("output", {}).get("ground_truth_sha256")
        if output_sha not in {golden_ground_truth_sha256, train_ground_truth_sha256}:
            raise ValueError(
                f"{path}: imported ground-truth SHA does not match current TRAIN/GOLDEN JSONL"
            )
        if not manifest.get("terms_acknowledged"):
            raise ValueError(f"{path}: public dataset terms were not acknowledged at import time")
        for source_file in manifest.get("source_files", []):
            if not isinstance(source_file, dict):
                raise ValueError(f"{path}: source_files entry must be an object")
            relative = source_file.get("path")
            expected_sha = source_file.get("sha256")
            if not isinstance(relative, str) or not relative:
                raise ValueError(f"{path}: source_files path must be non-empty")
            if not isinstance(expected_sha, str) or len(expected_sha) != 64:
                raise ValueError(f"{path}: source_files SHA-256 must be 64-hex")
            source_path = root / relative
            if not source_path.is_file():
                raise FileNotFoundError(source_path)
            actual_sha = sha256_file(source_path)
            if actual_sha != expected_sha:
                raise ValueError(
                    f"{path}: source file hash mismatch for {relative}: "
                    f"recorded={expected_sha} actual={actual_sha}"
                )
        validated.append(manifest)
    return validated


def inspect_corpus(
    *,
    video_root: str | Path,
    golden_ground_truth: str | Path,
    train_ground_truth: str | Path | None = None,
    coverage_profile: str = "private-cctv",
) -> dict[str, Any]:
    train = (
        inspect_split(train_ground_truth, video_root, "train", coverage_profile=coverage_profile)
        if train_ground_truth is not None
        else None
    )
    golden = inspect_split(
        golden_ground_truth,
        video_root,
        "golden",
        coverage_profile=coverage_profile,
    )
    errors: list[str] = []
    warnings: list[str] = []
    if train is not None:
        errors.extend(train["errors"])
        warnings.extend(train["warnings"])
    errors.extend(golden["errors"])
    warnings.extend(golden["warnings"])

    if train is not None:
        train_by_sha = {item["sha256"]: item["video"] for item in train["videos"]}
        for item in golden["videos"]:
            duplicate = train_by_sha.get(item["sha256"])
            if duplicate is not None:
                errors.append(
                    f"train/golden leakage: {duplicate!r} and {item['video']!r} have identical SHA-256"
                )

    return {
        "schema": CORPUS_SCHEMA,
        "video_root": str(Path(video_root)),
        "train": train,
        "golden": golden,
        "errors": errors,
        "coverage_profile": coverage_profile,
        "warnings": warnings,
        "valid": not errors,
    }


def build_frozen_manifest(
    report: dict[str, Any],
    *,
    revision: str,
    reviewer: str,
    human_confirmed: bool,
) -> dict[str, Any]:
    if not report.get("valid"):
        raise ValueError("Cannot freeze invalid corpus: " + "; ".join(report.get("errors", [])))
    if not revision.strip():
        raise ValueError("revision must be non-empty")
    if not reviewer.strip():
        raise ValueError("reviewer must be non-empty")
    if not human_confirmed:
        raise ValueError("Frozen golden corpus requires explicit human confirmation")

    manifest = {
        "schema": CORPUS_SCHEMA,
        "revision": revision.strip(),
        "human_confirmation": {"confirmed": True, "reviewer": reviewer.strip()},
        "allowed_labels": ["person"],
        "splits": {
            "train": None,
            "golden": {
                "ground_truth": report["golden"]["ground_truth"],
                "ground_truth_sha256": report["golden"]["ground_truth_sha256"],
                "videos": report["golden"]["videos"],
            },
        },
        "golden_coverage": report["golden"]["coverage"],
    }
    if report.get("train") is not None:
        manifest["splits"]["train"] = {
            "ground_truth": report["train"]["ground_truth"],
            "ground_truth_sha256": report["train"]["ground_truth_sha256"],
            "videos": report["train"]["videos"],
        }
    manifest["corpus_sha256"] = _canonical_sha256(manifest)
    return manifest


def load_frozen_manifest(path: str | Path) -> dict[str, Any]:
    manifest = _json_load(path)
    if manifest.get("schema") != CORPUS_SCHEMA:
        raise ValueError(f"Unsupported corpus manifest schema in {path}")
    expected = manifest.get("corpus_sha256")
    unsigned = dict(manifest)
    unsigned.pop("corpus_sha256", None)
    actual = _canonical_sha256(unsigned)
    if expected != actual:
        raise ValueError(f"Corpus manifest hash mismatch: recorded={expected!r} actual={actual}")
    if not manifest.get("human_confirmation", {}).get("confirmed"):
        raise ValueError("Frozen corpus is not human-confirmed")
    return manifest


def extract_frames(
    video_path: str | Path,
    *,
    video_id: str,
    output_dir: str | Path,
    frame_indices: Iterable[int] | None = None,
    every_seconds: float | None = None,
    max_frames: int = 200,
) -> dict[str, Any]:
    import cv2

    source = Path(video_path)
    if not source.is_file():
        raise FileNotFoundError(source)
    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {source}")
    try:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_indices is None and every_seconds is None:
            raise ValueError("Choose explicit frame indices or --every-seconds")
        if frame_indices is not None and every_seconds is not None:
            raise ValueError("Use either explicit frame indices or --every-seconds, not both")

        wanted = sorted(set(int(value) for value in frame_indices or []))
        if any(value < 0 for value in wanted):
            raise ValueError("frame indices must be non-negative")
        stride = None
        if every_seconds is not None:
            if not math.isfinite(every_seconds) or every_seconds <= 0.0:
                raise ValueError("every_seconds must be > 0")
            if not math.isfinite(fps) or fps <= 0.0:
                raise ValueError("video FPS unavailable; use explicit frame indices")
            stride = max(1, int(round(fps * every_seconds)))

        extracted: list[dict[str, Any]] = []
        frame_index = 0
        wanted_set = set(wanted)
        last_wanted = max(wanted) if wanted else None
        while len(extracted) < max_frames:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            selected = frame_index in wanted_set if stride is None else frame_index % stride == 0
            if selected:
                filename = f"frame_{frame_index:08d}.png"
                output_path = target_dir / filename
                if not cv2.imwrite(str(output_path), frame):
                    raise RuntimeError(f"Failed to write {output_path}")
                extracted.append({"frame": frame_index, "image": filename})
            frame_index += 1
            if stride is None and last_wanted is not None and frame_index > last_wanted:
                break
    finally:
        cap.release()

    batch = {
        "schema": FRAME_BATCH_SCHEMA,
        "video": video_id,
        "source_sha256": sha256_file(source),
        "width": width,
        "height": height,
        "fps": fps if math.isfinite(fps) and fps > 0.0 else None,
        "frame_count": frame_count if frame_count > 0 else None,
        "frames": extracted,
    }
    _json_dump(target_dir / "frames.json", batch)
    return batch


def _load_jsonl_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            raw = raw.strip()
            if not raw or raw.startswith("#"):
                continue
            try:
                row = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected JSON object")
            rows.append(row)
    return rows


def _write_jsonl_rows(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(rows, key=lambda item: (str(item.get("video", "")), int(item.get("frame", -1))))
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in ordered),
        encoding="utf-8",
    )


def annotate_frame_batch(
    batch_path: str | Path,
    *,
    output_ground_truth: str | Path,
    tags: Iterable[str] = (),
) -> None:
    import cv2

    batch_file = Path(batch_path)
    batch = _json_load(batch_file)
    if batch.get("schema") != FRAME_BATCH_SCHEMA:
        raise ValueError(f"Unsupported frame batch schema in {batch_file}")
    output = Path(output_ground_truth)
    existing = _load_jsonl_rows(output)
    row_map = {(row.get("video"), row.get("frame")): row for row in existing}
    video = batch["video"]
    base_tags = sorted({_normalize_token(tag) for tag in tags if tag.strip()})
    active_id = "p1"
    active_attributes: tuple[str, ...] = ()
    active_ignore = False
    frame_items = list(batch.get("frames", []))
    index = 0

    while 0 <= index < len(frame_items):
        item = frame_items[index]
        frame_number = int(item["frame"])
        image_path = batch_file.parent / item["image"]
        image = cv2.imread(str(image_path))
        if image is None:
            raise RuntimeError(f"Cannot read extracted frame: {image_path}")
        key = (video, frame_number)
        current = row_map.get(key, {"video": video, "frame": frame_number, "tags": base_tags, "objects": []})
        current_tags = {_normalize_token(tag) for tag in current.get("tags", [])}
        current_tags.update(base_tags)
        objects = [dict(obj) for obj in current.get("objects", []) if obj.get("label") == "person"]
        existing_active = next((obj for obj in objects if obj.get("id") == active_id), None)
        if existing_active is None:
            active_attributes = ()
            active_ignore = False
        else:
            active_attributes = tuple(
                sorted(
                    {
                        _normalize_token(str(item))
                        for item in existing_active.get("attributes", [])
                        if str(item).strip()
                    }
                )
            )
            active_ignore = bool(existing_active.get("ignore", False))
        drag_start: list[int] | None = None

        def mouse(event, x, y, _flags, _param):
            nonlocal drag_start, objects
            if event == cv2.EVENT_LBUTTONDOWN:
                drag_start = [x, y]
            elif event == cv2.EVENT_LBUTTONUP and drag_start is not None:
                x1, y1 = drag_start
                drag_start = None
                left, right = sorted((max(0, x1), max(0, x)))
                top, bottom = sorted((max(0, y1), max(0, y)))
                if right - left < 2 or bottom - top < 2:
                    return
                objects = [obj for obj in objects if obj.get("id") != active_id]
                annotated = {
                    "id": active_id,
                    "label": "person",
                    "bbox": [left, top, right, bottom],
                }
                if active_attributes:
                    annotated["attributes"] = list(active_attributes)
                if active_ignore:
                    annotated["ignore"] = True
                objects.append(annotated)

        window = "SpectraTrack human annotation"
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(window, mouse)

        def save_current() -> None:
            tags_now = set(current_tags)
            if objects:
                tags_now.discard("negative")
            else:
                tags_now.add("negative")
            row_map[key] = {
                "video": video,
                "frame": frame_number,
                "tags": sorted(tags_now),
                "objects": sorted(objects, key=lambda obj: str(obj.get("id", ""))),
            }
            _write_jsonl_rows(output, row_map.values())

        while True:
            canvas = image.copy()
            for obj in objects:
                x1, y1, x2, y2 = (int(value) for value in obj["bbox"])
                cv2.rectangle(canvas, (x1, y1), (x2, y2), (255, 255, 255), 2)
                label_text = str(obj.get("id", "person"))
                if obj.get("ignore"):
                    label_text += " IGNORE"
                cv2.putText(
                    canvas,
                    label_text,
                    (x1, max(18, y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )
            help_text = (
                f"frame={frame_number} active={active_id} ignore={active_ignore} | "
                "drag=box i=id a=attrs g=ignore u=undo c=clear n/ENTER=next b=back q=save+quit"
            )
            cv2.putText(canvas, help_text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
            cv2.imshow(window, canvas)
            pressed = cv2.waitKey(30) & 0xFF
            if pressed in (13, ord("n")):
                save_current()
                index += 1
                break
            if pressed == ord("b"):
                save_current()
                index = max(0, index - 1)
                break
            if pressed == ord("u") and objects:
                objects.pop()
            elif pressed == ord("c"):
                objects.clear()
            elif pressed == ord("i"):
                candidate = input("Stable person id for this clip (example p1): ").strip()
                if candidate:
                    active_id = candidate
                    existing_active = next(
                        (obj for obj in objects if obj.get("id") == active_id),
                        None,
                    )
                    if existing_active is None:
                        active_attributes = ()
                        active_ignore = False
                    else:
                        active_attributes = tuple(
                            sorted(
                                {
                                    _normalize_token(str(item))
                                    for item in existing_active.get("attributes", [])
                                    if str(item).strip()
                                }
                            )
                        )
                        active_ignore = bool(existing_active.get("ignore", False))
            elif pressed == ord("a"):
                raw_attributes = input(
                    "Comma-separated attributes for the active person (blank clears): "
                ).strip()
                active_attributes = tuple(
                    sorted(
                        {
                            _normalize_token(item)
                            for item in raw_attributes.split(",")
                            if item.strip()
                        }
                    )
                )
            elif pressed == ord("g"):
                active_ignore = not active_ignore
            elif pressed == ord("q"):
                save_current()
                cv2.destroyAllWindows()
                return
        cv2.destroyWindow(window)
    cv2.destroyAllWindows()


def _validate_bbox(value: Any, width: int, height: int, where: str) -> tuple[float, float, float, float]:
    if not isinstance(value, list) or len(value) != 4:
        raise ValueError(f"{where}: bbox must be [x1,y1,x2,y2]")
    try:
        x1, y1, x2, y2 = (float(item) for item in value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{where}: bbox must be numeric") from exc
    if not all(math.isfinite(item) for item in (x1, y1, x2, y2)):
        raise ValueError(f"{where}: bbox must be finite")
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"{where}: bbox must have positive size")
    if x1 < 0 or y1 < 0 or x2 > width or y2 > height:
        raise ValueError(f"{where}: bbox outside {width}x{height}")
    return x1, y1, x2, y2


def validate_detection_replay(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    rows = _load_jsonl_rows(source)
    if not rows:
        raise ValueError(f"{source}: replay is empty")
    metadata = rows[0]
    if metadata.get("type") != "metadata" or metadata.get("schema") != REPLAY_SCHEMA:
        raise ValueError(f"{source}: first record must be {REPLAY_SCHEMA} metadata")

    source_commit = metadata.get("source_commit")
    if not _is_hex(source_commit, 40):
        raise ValueError("replay metadata source_commit must be a full 40-hex commit SHA")
    video = metadata.get("video")
    if not isinstance(video, str) or not video:
        raise ValueError("replay metadata video must be non-empty")
    video_sha = metadata.get("video_sha256")
    if video_sha is not None and not _is_hex(video_sha, 64):
        raise ValueError("replay metadata video_sha256 must be null or 64-hex")
    detector = metadata.get("detector")
    if not isinstance(detector, str) or not detector:
        raise ValueError("replay metadata detector must be non-empty")
    model_sha = metadata.get("model_sha256")
    if model_sha is not None and not _is_hex(model_sha, 64):
        raise ValueError("replay metadata model_sha256 must be null or 64-hex")
    provider = metadata.get("provider")
    if not isinstance(provider, str) or not provider:
        raise ValueError("replay metadata provider must be non-empty")
    if not isinstance(metadata.get("config"), dict):
        raise ValueError("replay metadata config must be an object")
    width = metadata.get("width")
    height = metadata.get("height")
    if not isinstance(width, int) or isinstance(width, bool) or width <= 0:
        raise ValueError("replay metadata width must be a positive integer")
    if not isinstance(height, int) or isinstance(height, bool) or height <= 0:
        raise ValueError("replay metadata height must be a positive integer")

    seen_frames: set[int] = set()
    previous_frame = -1
    detections = 0
    for row_index, row in enumerate(rows[1:], start=2):
        if row.get("type") != "frame":
            raise ValueError(f"{source}:{row_index}: expected frame record")
        if row.get("video") != video:
            raise ValueError(f"{source}:{row_index}: frame video differs from metadata")
        frame = row.get("frame")
        if not isinstance(frame, int) or isinstance(frame, bool) or frame < 0:
            raise ValueError(f"{source}:{row_index}: frame must be a non-negative integer")
        if frame in seen_frames:
            raise ValueError(f"{source}:{row_index}: duplicate frame {frame}")
        if frame <= previous_frame:
            raise ValueError(f"{source}:{row_index}: frames must be strictly increasing")
        seen_frames.add(frame)
        previous_frame = frame
        if row.get("width") != width or row.get("height") != height:
            raise ValueError(f"{source}:{row_index}: frame dimensions differ from metadata")
        timestamp = row.get("timestamp_s")
        if timestamp is not None and (
            not isinstance(timestamp, (int, float))
            or isinstance(timestamp, bool)
            or not math.isfinite(float(timestamp))
            or float(timestamp) < 0.0
        ):
            raise ValueError(f"{source}:{row_index}: timestamp_s must be null or finite >= 0")
        frame_detections = row.get("detections")
        if not isinstance(frame_detections, list):
            raise ValueError(f"{source}:{row_index}: detections must be a list")
        for det_index, detection in enumerate(frame_detections):
            where = f"{source}:{row_index}:detections[{det_index}]"
            if not isinstance(detection, dict):
                raise ValueError(f"{where}: detection must be an object")
            _validate_bbox(detection.get("bbox"), width, height, where)
            score = detection.get("score")
            if not isinstance(score, (int, float)) or isinstance(score, bool) or not math.isfinite(float(score)):
                raise ValueError(f"{where}: score must be finite")
            if not 0.0 <= float(score) <= 1.0:
                raise ValueError(f"{where}: score must be in [0,1]")
            class_id = detection.get("class_id")
            if not isinstance(class_id, int) or isinstance(class_id, bool):
                raise ValueError(f"{where}: class_id must be an integer")
            label = detection.get("label")
            if not isinstance(label, str) or not label:
                raise ValueError(f"{where}: label must be non-empty")
            appearance = detection.get("appearance")
            if appearance is not None:
                if isinstance(appearance, dict):
                    if not isinstance(appearance.get("schema"), str) or not appearance.get("schema"):
                        raise ValueError(f"{where}: appearance object must have a schema")
                elif isinstance(appearance, list):
                    appearance_schema = metadata["config"].get("appearance_schema")
                    if not isinstance(appearance_schema, str) or not appearance_schema:
                        raise ValueError(
                            f"{where}: appearance vector requires config.appearance_schema"
                        )
                    if not appearance or any(
                        isinstance(value, bool)
                        or not isinstance(value, (int, float))
                        or not math.isfinite(float(value))
                        for value in appearance
                    ):
                        raise ValueError(f"{where}: appearance vector must contain finite numbers")
                else:
                    raise ValueError(f"{where}: unsupported appearance metadata")
            detections += 1

    return {
        "schema": REPLAY_SCHEMA,
        "replay_sha256": sha256_file(source),
        "source_commit": source_commit,
        "video": video,
        "video_sha256": video_sha,
        "detector": detector,
        "model_sha256": model_sha,
        "provider": provider,
        "config_sha256": _canonical_sha256(metadata["config"]),
        "width": width,
        "height": height,
        "frame_records": len(seen_frames),
        "detections": detections,
    }


def _golden_video_map(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    golden = manifest.get("splits", {}).get("golden")
    if not isinstance(golden, dict):
        raise ValueError("Corpus manifest has no golden split")
    return {item["video"]: item for item in golden.get("videos", [])}


def stamp_qa_result(
    *,
    manifest_path: str | Path,
    result_path: str | Path,
    role: str,
    experiment: str,
) -> dict[str, Any]:
    manifest = load_frozen_manifest(manifest_path)
    result = _json_load(result_path)
    if result.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported QA result schema")
    golden = manifest["splits"]["golden"]
    if result.get("ground_truth_sha256") != golden.get("ground_truth_sha256"):
        raise ValueError("QA result was not scored against this manifest's golden ground truth")
    revision = result.get("revision")
    if not _is_hex(revision, 40):
        raise ValueError("QA result revision must be a full 40-hex subject commit SHA")
    if not isinstance(result.get("settings"), dict):
        raise ValueError("QA result settings are required")
    if not isinstance(result.get("evaluation"), dict):
        raise ValueError("QA result evaluation settings are required")

    expected_videos = _golden_video_map(manifest)
    observed_videos = {item.get("video"): item for item in result.get("input_videos", [])}
    if set(observed_videos) != set(expected_videos):
        raise ValueError("QA result input videos do not exactly match the frozen golden split")
    for video, expected in expected_videos.items():
        observed = observed_videos[video]
        for key in ("sha256", "width", "height"):
            if observed.get(key) != expected.get(key):
                raise ValueError(f"QA result {video!r} has different {key} from frozen corpus")

    model = result.get("model", {})
    if not isinstance(model.get("path"), str) or not model.get("path"):
        raise ValueError("QA result model identifier/path is required")
    if not _is_hex(model.get("sha256"), 64):
        raise ValueError("QA result model SHA-256 is required")
    providers = model.get("providers")
    if not isinstance(providers, list) or not providers or not all(isinstance(item, str) and item for item in providers):
        raise ValueError("QA result provider list is required")
    if not role.strip() or not experiment.strip():
        raise ValueError("role and experiment must be non-empty")

    return {
        "schema": STAMPED_RUN_SCHEMA,
        "kind": "qa_result",
        "role": role.strip(),
        "experiment": experiment.strip(),
        "corpus_revision": manifest["revision"],
        "corpus_sha256": manifest["corpus_sha256"],
        "qa_result_sha256": sha256_file(result_path),
        "result_payload_sha256": _canonical_sha256(result),
        "source_commit": revision,
        "settings_sha256": _canonical_sha256(result["settings"]),
        "evaluation_sha256": _canonical_sha256(result["evaluation"]),
        "result": result,
    }


def _validate_frozen_input_videos(
    manifest: dict[str, Any],
    input_videos: Any,
    *,
    where: str,
) -> list[dict[str, Any]]:
    if not isinstance(input_videos, list):
        raise ValueError(f"{where}: input_videos must be a list")
    expected_videos = _golden_video_map(manifest)
    observed_videos = {
        item.get("video"): item
        for item in input_videos
        if isinstance(item, dict)
    }
    if len(observed_videos) != len(input_videos):
        raise ValueError(f"{where}: every input video must be an object with a unique video id")
    if set(observed_videos) != set(expected_videos):
        raise ValueError(f"{where}: input videos do not exactly match the frozen golden split")
    for video, expected in expected_videos.items():
        observed = observed_videos[video]
        for key in ("sha256", "width", "height"):
            if observed.get(key) != expected.get(key):
                raise ValueError(f"{where}: {video!r} has different {key} from frozen corpus")
    return input_videos


def stamp_external_evidence(
    *,
    manifest_path: str | Path,
    evidence_path: str | Path,
    source_artifact_path: str | Path,
) -> dict[str, Any]:
    manifest = load_frozen_manifest(manifest_path)
    evidence = _json_load(evidence_path)
    if evidence.get("schema") != EVIDENCE_SCHEMA:
        raise ValueError(f"Expected {EVIDENCE_SCHEMA}")
    role = evidence.get("role")
    experiment = evidence.get("experiment")
    scope = evidence.get("scope")
    for field_name, field_value in (
        ("role", role),
        ("experiment", experiment),
        ("scope", scope),
    ):
        if not isinstance(field_value, str) or not field_value.strip():
            raise ValueError(f"evidence {field_name} must be non-empty")
    source_commit = evidence.get("source_commit")
    if not _is_hex(source_commit, 40):
        raise ValueError("evidence source_commit must be a full 40-hex SHA")
    if evidence.get("corpus_revision") != manifest["revision"]:
        raise ValueError("evidence corpus revision differs from frozen manifest")
    if evidence.get("corpus_sha256") != manifest["corpus_sha256"]:
        raise ValueError("evidence corpus SHA differs from frozen manifest")
    golden = manifest["splits"]["golden"]
    if evidence.get("ground_truth_sha256") != golden.get("ground_truth_sha256"):
        raise ValueError("evidence ground-truth SHA differs from frozen golden split")
    source_artifact_sha = evidence.get("source_artifact_sha256")
    if not _is_hex(source_artifact_sha, 64):
        raise ValueError("evidence source_artifact_sha256 must be 64-hex")
    source_artifact = Path(source_artifact_path)
    if not source_artifact.is_file():
        raise FileNotFoundError(source_artifact)
    actual_source_artifact_sha = sha256_file(source_artifact)
    if source_artifact_sha != actual_source_artifact_sha:
        raise ValueError(
            "evidence source_artifact_sha256 does not match the supplied source artifact"
        )
    for field_name in ("config", "evaluation", "quality", "compute", "provenance"):
        if not isinstance(evidence.get(field_name), dict):
            raise ValueError(f"evidence {field_name} must be an object")
    _validate_frozen_input_videos(
        manifest,
        evidence["provenance"].get("input_videos"),
        where="evidence provenance",
    )
    replay_sha = evidence["provenance"].get("replay_sha256")
    if replay_sha is not None and not _is_hex(replay_sha, 64):
        raise ValueError("evidence provenance replay_sha256 must be null or 64-hex")

    model = evidence.get("model")
    if model is not None:
        if not isinstance(model, dict):
            raise ValueError("evidence model must be null or an object")
        model_id = model.get("id")
        if not isinstance(model_id, str) or not model_id:
            raise ValueError("evidence model.id must be non-empty when model is present")
        model_sha = model.get("sha256")
        if model_sha is not None and not _is_hex(model_sha, 64):
            raise ValueError("evidence model.sha256 must be null or 64-hex")
        provider = model.get("provider")
        if not isinstance(provider, str) or not provider:
            raise ValueError("evidence model.provider must be non-empty when model is present")

    return {
        "schema": STAMPED_RUN_SCHEMA,
        "kind": "external_evidence",
        "role": role.strip(),
        "experiment": experiment.strip(),
        "corpus_revision": manifest["revision"],
        "corpus_sha256": manifest["corpus_sha256"],
        "source_commit": source_commit,
        "source_evidence_sha256": sha256_file(evidence_path),
        "source_artifact_sha256": actual_source_artifact_sha,
        "evidence_payload_sha256": _canonical_sha256(evidence),
        "settings_sha256": _canonical_sha256(evidence["config"]),
        "evaluation_sha256": _canonical_sha256(evidence["evaluation"]),
        "evidence": evidence,
    }


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _leaderboard_row(run: dict[str, Any]) -> dict[str, Any]:
    result = run["result"]
    metrics = result["metrics"]
    tracking = metrics.get("tracking", {})
    bbox = metrics.get("bbox_stability", {})
    detection_bbox = bbox.get("detection", {})
    tracking_bbox = bbox.get("tracking", {})
    performance = result.get("performance", {})
    by_size = metrics.get("by_size", {})
    model = result.get("model", {})
    return {
        "role": run["role"],
        "experiment": run["experiment"],
        "scope": "end-to-end-qa",
        "source_commit": run["source_commit"],
        "model": model.get("path"),
        "model_sha256": model.get("sha256"),
        "providers": model.get("providers"),
        "settings_sha256": run["settings_sha256"],
        "settings": result.get("settings"),
        "evaluation": result.get("evaluation"),
        "recall": metrics.get("recall"),
        "precision": metrics.get("precision"),
        "fn": metrics.get("false_negatives"),
        "fp": metrics.get("false_positives"),
        "recall_lt24": by_size.get("height_lt_24", {}).get("recall"),
        "recall_24_47": by_size.get("height_24_47", {}).get("recall"),
        "recall_48_95": by_size.get("height_48_95", {}).get("recall"),
        "recall_ge96": by_size.get("height_ge_96", {}).get("recall"),
        "track_recall": tracking.get("recall"),
        "id_switches": tracking.get("id_switches"),
        "fragmentations": tracking.get("fragmentations"),
        "mean_track_length_annotated_frames": tracking.get(
            "mean_uninterrupted_track_length_annotated_frames"
        ),
        "mean_recovery_latency_frames": tracking.get("mean_recovery_latency_frames"),
        "detection_center_jitter": detection_bbox.get("normalized_center_jitter_mean"),
        "detection_temporal_iou": detection_bbox.get("temporal_iou_mean"),
        "tracking_center_jitter": tracking_bbox.get("normalized_center_jitter_mean"),
        "tracking_temporal_iou": tracking_bbox.get("temporal_iou_mean"),
        "onnx_inference_calls": performance.get("onnx_inference_calls"),
        "onnx_calls_per_frame": performance.get("onnx_calls_per_frame"),
        "wall_seconds": performance.get("wall_seconds"),
        "processing_seconds_per_source_second": performance.get(
            "processing_seconds_per_source_second"
        ),
        "peak_vram_mb": performance.get("peak_vram_mb"),
    }


def _external_evidence_row(run: dict[str, Any]) -> dict[str, Any]:
    evidence = run["evidence"]
    quality = evidence["quality"]
    compute = evidence["compute"]
    model = evidence.get("model") or {}
    return {
        "role": run["role"],
        "experiment": run["experiment"],
        "scope": evidence["scope"],
        "source_commit": run["source_commit"],
        "model": model.get("id"),
        "model_sha256": model.get("sha256"),
        "providers": [model["provider"]] if model.get("provider") else None,
        "settings_sha256": run["settings_sha256"],
        "settings": evidence["config"],
        "evaluation": evidence["evaluation"],
        "recall": quality.get("recall"),
        "precision": quality.get("precision"),
        "fn": quality.get("fn"),
        "fp": quality.get("fp"),
        "recall_lt24": quality.get("recall_lt24"),
        "recall_24_47": quality.get("recall_24_47"),
        "recall_48_95": quality.get("recall_48_95"),
        "recall_ge96": quality.get("recall_ge96"),
        "track_recall": quality.get("track_recall"),
        "id_switches": quality.get("id_switches"),
        "fragmentations": quality.get("fragmentations"),
        "mean_track_length_annotated_frames": quality.get(
            "mean_track_length_annotated_frames"
        ),
        "mean_recovery_latency_frames": quality.get("mean_recovery_latency_frames"),
        "detection_center_jitter": quality.get("detection_center_jitter"),
        "detection_temporal_iou": quality.get("detection_temporal_iou"),
        "tracking_center_jitter": quality.get("tracking_center_jitter"),
        "tracking_temporal_iou": quality.get("tracking_temporal_iou"),
        "onnx_inference_calls": compute.get("onnx_inference_calls"),
        "onnx_calls_per_frame": compute.get("onnx_calls_per_frame"),
        "wall_seconds": compute.get("wall_seconds"),
        "processing_seconds_per_source_second": compute.get(
            "processing_seconds_per_source_second"
        ),
        "peak_vram_mb": compute.get("peak_vram_mb"),
    }


def build_leaderboard(
    *,
    manifest_path: str | Path,
    run_paths: Iterable[str | Path],
) -> dict[str, Any]:
    manifest = load_frozen_manifest(manifest_path)
    runs: list[dict[str, Any]] = []
    evaluation: dict[str, Any] | None = None
    identities: set[tuple[str, str]] = set()
    for path in run_paths:
        run = _json_load(path)
        if run.get("schema") != STAMPED_RUN_SCHEMA:
            raise ValueError(f"{path}: expected {STAMPED_RUN_SCHEMA}")
        if run.get("corpus_revision") != manifest["revision"] or run.get("corpus_sha256") != manifest["corpus_sha256"]:
            raise ValueError(f"{path}: corpus revision/hash differs from leaderboard manifest")
        kind = run.get("kind", "qa_result")
        if kind == "qa_result":
            result = run.get("result")
            if not isinstance(result, dict):
                raise ValueError(f"{path}: stamped run has no result object")
            if run.get("result_payload_sha256") != _canonical_sha256(result):
                raise ValueError(f"{path}: stamped result payload hash mismatch")
            settings = result.get("settings")
            current_evaluation = result.get("evaluation")
        elif kind == "external_evidence":
            evidence = run.get("evidence")
            if not isinstance(evidence, dict):
                raise ValueError(f"{path}: stamped run has no evidence object")
            if run.get("evidence_payload_sha256") != _canonical_sha256(evidence):
                raise ValueError(f"{path}: stamped evidence payload hash mismatch")
            settings = evidence.get("config")
            current_evaluation = evidence.get("evaluation")
        else:
            raise ValueError(f"{path}: unsupported stamped run kind {kind!r}")
        if run.get("settings_sha256") != _canonical_sha256(settings):
            raise ValueError(f"{path}: settings hash mismatch")
        if run.get("evaluation_sha256") != _canonical_sha256(current_evaluation):
            raise ValueError(f"{path}: evaluation hash mismatch")
        if evaluation is None:
            evaluation = current_evaluation
        elif current_evaluation != evaluation:
            raise ValueError(f"{path}: incompatible scoring settings")
        identity = (str(run.get("role")), str(run.get("experiment")))
        if identity in identities:
            raise ValueError(f"duplicate leaderboard row {identity}")
        identities.add(identity)
        runs.append(run)
    if not runs:
        raise ValueError("Leaderboard requires at least one stamped run")

    rows = [
        _external_evidence_row(run)
        if run.get("kind") == "external_evidence"
        else _leaderboard_row(run)
        for run in runs
    ]
    return {
        "schema": LEADERBOARD_SCHEMA,
        "corpus_revision": manifest["revision"],
        "corpus_sha256": manifest["corpus_sha256"],
        "evaluation": evaluation,
        "note": "Rows are evidence only; this table intentionally does not rank or declare a winner.",
        "rows": rows,
    }


def leaderboard_markdown(leaderboard: dict[str, Any]) -> str:
    columns = [
        ("Role", "role"),
        ("Experiment", "experiment"),
        ("Scope", "scope"),
        ("Recall", "recall"),
        ("Precision", "precision"),
        ("FN", "fn"),
        ("FP", "fp"),
        ("<24", "recall_lt24"),
        ("24-47", "recall_24_47"),
        ("48-95", "recall_48_95"),
        (">=96", "recall_ge96"),
        ("IDSW", "id_switches"),
        ("Frag", "fragmentations"),
        ("Track len", "mean_track_length_annotated_frames"),
        ("Recovery", "mean_recovery_latency_frames"),
        ("Det jitter", "detection_center_jitter"),
        ("Det tIoU", "detection_temporal_iou"),
        ("ONNX calls", "onnx_inference_calls"),
        ("Calls/frame", "onnx_calls_per_frame"),
        ("sec/src-sec", "processing_seconds_per_source_second"),
        ("Wall s", "wall_seconds"),
        ("Settings", "settings_sha256"),
    ]
    lines = [
        f"# SpectraTrack vNext leaderboard — {leaderboard['corpus_revision']}",
        "",
        leaderboard["note"],
        "",
        "| " + " | ".join(title for title, _ in columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in leaderboard["rows"]:
        values = []
        for _, key in columns:
            value = row.get(key)
            if key == "settings_sha256" and isinstance(value, str):
                value = value[:12]
            values.append(_fmt(value))
        lines.append("| " + " | ".join(values) + " |")
    lines.append("")
    return "\n".join(lines)


def _parse_frame_indices(value: str) -> list[int]:
    if not value.strip():
        return []
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SpectraTrack vNext CCTV QA experiment control")
    sub = parser.add_subparsers(dest="command", required=True)

    extract = sub.add_parser("extract-frames", help="Extract lossless review frames from a user CCTV video")
    extract.add_argument("--video", required=True)
    extract.add_argument("--video-id", required=True, help="Canonical corpus path, e.g. golden/night01.mp4")
    extract.add_argument("--output-dir", required=True)
    choice = extract.add_mutually_exclusive_group(required=True)
    choice.add_argument("--frames", help="Comma-separated zero-based frame indices")
    choice.add_argument("--every-seconds", type=float)
    extract.add_argument("--max-frames", type=int, default=200)

    annotate = sub.add_parser("annotate", help="Human-draw person boxes on an extracted frame batch")
    annotate.add_argument("--batch", required=True, help="frames.json from extract-frames")
    annotate.add_argument("--output", required=True, help="Canonical qa_benchmark JSONL ground truth")
    annotate.add_argument("--tags", default="", help="Comma-separated condition tags for this frame batch")

    validate = sub.add_parser("validate-corpus", help="Validate TRAIN/GOLDEN annotations and video provenance")
    validate.add_argument("--video-root", required=True)
    validate.add_argument("--golden-ground-truth", required=True)
    validate.add_argument("--train-ground-truth")
    validate.add_argument("--output")

    freeze = sub.add_parser("freeze-corpus", help="Freeze a human-confirmed corpus manifest/hash")
    freeze.add_argument("--video-root", required=True)
    freeze.add_argument("--golden-ground-truth", required=True)
    freeze.add_argument("--train-ground-truth")
    freeze.add_argument("--revision", required=True)
    freeze.add_argument("--reviewer", required=True)
    freeze.add_argument("--confirm-human-reviewed", action="store_true")
    freeze.add_argument("--output", required=True)

    replay = sub.add_parser("validate-replay", help=f"Validate canonical {REPLAY_SCHEMA} provenance")
    replay.add_argument("--replay", required=True)
    replay.add_argument("--output")

    stamp = sub.add_parser("stamp-result", help="Bind a qa_benchmark result to a frozen golden corpus")
    stamp.add_argument("--manifest", required=True)
    stamp.add_argument("--result", required=True)
    stamp.add_argument("--role", required=True)
    stamp.add_argument("--experiment", required=True)
    stamp.add_argument("--output", required=True)

    evidence = sub.add_parser(
        "stamp-evidence",
        help="Bind normalized A1-A4 research evidence to a frozen golden corpus",
    )
    evidence.add_argument("--manifest", required=True)
    evidence.add_argument("--evidence", required=True)
    evidence.add_argument("--source-artifact", required=True)
    evidence.add_argument("--output", required=True)

    leaderboard = sub.add_parser("leaderboard", help="Assemble comparable stamped runs without ranking them")
    leaderboard.add_argument("--manifest", required=True)
    leaderboard.add_argument("--run", action="append", required=True, dest="runs")
    leaderboard.add_argument("--output-json", required=True)
    leaderboard.add_argument("--output-md", required=True)
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    try:
        if args.command == "extract-frames":
            if args.max_frames <= 0:
                raise ValueError("--max-frames must be > 0")
            batch = extract_frames(
                args.video,
                video_id=args.video_id,
                output_dir=args.output_dir,
                frame_indices=_parse_frame_indices(args.frames) if args.frames is not None else None,
                every_seconds=args.every_seconds,
                max_frames=args.max_frames,
            )
            print(json.dumps(batch, indent=2))
            return 0

        if args.command == "annotate":
            tags = [item.strip() for item in args.tags.split(",") if item.strip()]
            annotate_frame_batch(args.batch, output_ground_truth=args.output, tags=tags)
            return 0

        if args.command in {"validate-corpus", "freeze-corpus"}:
            report = inspect_corpus(
                video_root=args.video_root,
                golden_ground_truth=args.golden_ground_truth,
                train_ground_truth=args.train_ground_truth,
            )
            if args.command == "validate-corpus":
                if args.output:
                    _json_dump(args.output, report)
                print(json.dumps(report, indent=2))
                return 0 if report["valid"] else 2
            manifest = build_frozen_manifest(
                report,
                revision=args.revision,
                reviewer=args.reviewer,
                human_confirmed=args.confirm_human_reviewed,
            )
            _json_dump(args.output, manifest)
            print(f"corpus_revision={manifest['revision']} corpus_sha256={manifest['corpus_sha256']}")
            return 0

        if args.command == "validate-replay":
            report = validate_detection_replay(args.replay)
            if args.output:
                _json_dump(args.output, report)
            print(json.dumps(report, indent=2))
            return 0

        if args.command == "stamp-result":
            stamped = stamp_qa_result(
                manifest_path=args.manifest,
                result_path=args.result,
                role=args.role,
                experiment=args.experiment,
            )
            _json_dump(args.output, stamped)
            print(f"stamped={args.output} settings_sha256={stamped['settings_sha256']}")
            return 0

        if args.command == "stamp-evidence":
            stamped = stamp_external_evidence(
                manifest_path=args.manifest,
                evidence_path=args.evidence,
                source_artifact_path=args.source_artifact,
            )
            _json_dump(args.output, stamped)
            print(f"stamped={args.output} settings_sha256={stamped['settings_sha256']}")
            return 0

        leaderboard = build_leaderboard(manifest_path=args.manifest, run_paths=args.runs)
        _json_dump(args.output_json, leaderboard)
        Path(args.output_md).write_text(leaderboard_markdown(leaderboard), encoding="utf-8")
        print(f"rows={len(leaderboard['rows'])} corpus_sha256={leaderboard['corpus_sha256']}")
        return 0
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
