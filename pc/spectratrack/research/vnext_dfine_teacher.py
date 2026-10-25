from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any, Iterable, Mapping, Sequence

import cv2
import numpy as np

from ..detector import _iou
from ..qa_benchmark import (
    GroundTruthFrame,
    GroundTruthObject,
    PredictedObject,
    _match_objects,
    evaluate_frames,
    ground_truth_sha256,
    load_ground_truth,
)
from ..types import Detection
from .vnext_detector_backend_lab import (
    CurrentYoloAdapter,
    _direct_image_path,
    _image_sequence_path,
    _load_benchmark_source_records,
)

TEACHER_SCHEMA = "spectratrack-dfine-offline-teacher-v1"
COMPARE_SCHEMA = "spectratrack-dfine-teacher-compare-v1"
RAW_SCHEMA = "spectratrack-dfine-external-v1"
DEFAULT_PERSON_LABELS = ("person",)
FORBIDDEN_TRAINING_TOKENS = (
    "nightowls",
    "smoke400",
    "holdout4200",
    "full5000",
    "slice5000",
)


@dataclass(frozen=True, slots=True)
class TeacherProvenance:
    model_family: str
    model_variant: str
    training_recipe: str
    weights_sha256: str
    model_source: str
    model_revision: str
    code_source: str
    code_revision: str
    runtime: str
    runtime_version: str

    def validate(self) -> None:
        values = asdict(self)
        for name, value in values.items():
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"teacher provenance {name} must be a non-empty string")
        if len(self.weights_sha256) != 64 or any(ch not in "0123456789abcdef" for ch in self.weights_sha256.lower()):
            raise ValueError("teacher provenance weights_sha256 must be a 64-character hex SHA-256")


@dataclass(frozen=True, slots=True)
class TeacherFrame:
    video: str
    frame: int
    width: int
    height: int
    detections: tuple[Detection, ...]


def _sha256_tree(path: str | Path) -> str:
    root = Path(path)
    if not root.is_dir():
        raise FileNotFoundError(f"model directory not found: {root}")
    digest = hashlib.sha256()
    files = sorted(item for item in root.rglob("*") if item.is_file())
    if not files:
        raise ValueError(f"model directory contains no files: {root}")
    for item in files:
        rel = item.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(rel).to_bytes(8, "big"))
        digest.update(rel)
        with item.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _is_eval_only_identifier(*values: str) -> bool:
    joined = " ".join(value.lower() for value in values if isinstance(value, str))
    return any(token in joined for token in FORBIDDEN_TRAINING_TOKENS)


def _parse_box(item: Mapping[str, Any], width: int, height: int) -> tuple[float, float, float, float]:
    raw = item.get("bbox")
    if not isinstance(raw, list) or len(raw) != 4:
        raise ValueError("teacher detection bbox must contain four numbers")
    try:
        box = tuple(float(value) for value in raw)
    except (TypeError, ValueError) as exc:
        raise ValueError("teacher detection bbox must contain numeric values") from exc
    if not all(math.isfinite(value) for value in box):
        raise ValueError("teacher detection bbox must contain finite values")

    space = item.get("coordinate_space", "xyxy_abs")
    if space == "xyxy_abs":
        x1, y1, x2, y2 = box
    elif space == "xyxy_norm":
        x1, y1, x2, y2 = (
            box[0] * width,
            box[1] * height,
            box[2] * width,
            box[3] * height,
        )
    elif space == "cxcywh_norm":
        cx, cy, bw, bh = box
        x1 = (cx - bw / 2.0) * width
        y1 = (cy - bh / 2.0) * height
        x2 = (cx + bw / 2.0) * width
        y2 = (cy + bh / 2.0) * height
    elif space == "xyxy_input":
        input_size = item.get("input_size")
        region = item.get("source_region")
        if (
            not isinstance(input_size, list)
            or len(input_size) != 2
            or not all(isinstance(value, (int, float)) for value in input_size)
        ):
            raise ValueError("xyxy_input detection requires input_size=[width,height]")
        if (
            not isinstance(region, list)
            or len(region) != 4
            or not all(isinstance(value, (int, float)) for value in region)
        ):
            raise ValueError("xyxy_input detection requires source_region=[x1,y1,x2,y2]")
        input_w, input_h = float(input_size[0]), float(input_size[1])
        rx1, ry1, rx2, ry2 = (float(value) for value in region)
        if input_w <= 0 or input_h <= 0 or rx2 <= rx1 or ry2 <= ry1:
            raise ValueError("invalid input_size/source_region for xyxy_input")
        scale_x = (rx2 - rx1) / input_w
        scale_y = (ry2 - ry1) / input_h
        x1, y1, x2, y2 = (
            rx1 + box[0] * scale_x,
            ry1 + box[1] * scale_y,
            rx1 + box[2] * scale_x,
            ry1 + box[3] * scale_y,
        )
    else:
        raise ValueError(f"unsupported teacher coordinate_space: {space!r}")

    clipped = (
        max(0.0, min(float(width), x1)),
        max(0.0, min(float(height), y1)),
        max(0.0, min(float(width), x2)),
        max(0.0, min(float(height), y2)),
    )
    if clipped[2] <= clipped[0] or clipped[3] <= clipped[1]:
        raise ValueError("teacher detection bbox becomes empty after coordinate remap/clipping")
    return clipped


def normalize_teacher_detections(
    raw_detections: Iterable[Mapping[str, Any]],
    *,
    width: int,
    height: int,
    score_floor: float,
    person_class_ids: set[int],
    person_labels: set[str],
    dedupe_iou: float,
) -> list[Detection]:
    if width <= 0 or height <= 0:
        raise ValueError("frame width/height must be positive")
    if not 0.0 <= score_floor <= 1.0:
        raise ValueError("score_floor must be in [0,1]")
    if not 0.0 <= dedupe_iou <= 1.0:
        raise ValueError("dedupe_iou must be in [0,1]")

    normalized: list[Detection] = []
    labels_lower = {label.lower() for label in person_labels}
    for index, item in enumerate(raw_detections):
        if not isinstance(item, Mapping):
            raise ValueError(f"teacher detection {index} must be an object")
        try:
            score = float(item.get("score"))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"teacher detection {index} score must be numeric") from exc
        if not math.isfinite(score) or not 0.0 <= score <= 1.0:
            raise ValueError(f"teacher detection {index} score must be finite and in [0,1]")
        if score < score_floor:
            continue

        class_id_raw = item.get("class_id")
        label_raw = item.get("label")
        class_id = int(class_id_raw) if isinstance(class_id_raw, int) and not isinstance(class_id_raw, bool) else None
        label = str(label_raw) if isinstance(label_raw, str) else ""
        is_person = (class_id is not None and class_id in person_class_ids) or label.lower() in labels_lower
        if not is_person:
            continue
        normalized.append(Detection(_parse_box(item, width, height), score, 0, "person"))

    ordered = sorted(
        normalized,
        key=lambda detection: (
            -detection.score,
            detection.bbox[0],
            detection.bbox[1],
            detection.bbox[2],
            detection.bbox[3],
        ),
    )
    kept: list[Detection] = []
    for candidate in ordered:
        if any(_iou(candidate.bbox, current.bbox) >= dedupe_iou for current in kept):
            continue
        kept.append(candidate)
    return kept


def _provenance_from_metadata(metadata: Mapping[str, Any]) -> TeacherProvenance:
    provenance = metadata.get("provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError("teacher metadata requires provenance object")
    result = TeacherProvenance(
        model_family=str(provenance.get("model_family", "")),
        model_variant=str(provenance.get("model_variant", "")),
        training_recipe=str(provenance.get("training_recipe", "")),
        weights_sha256=str(provenance.get("weights_sha256", "")).lower(),
        model_source=str(provenance.get("model_source", "")),
        model_revision=str(provenance.get("model_revision", "")),
        code_source=str(provenance.get("code_source", "")),
        code_revision=str(provenance.get("code_revision", "")),
        runtime=str(provenance.get("runtime", "")),
        runtime_version=str(provenance.get("runtime_version", "")),
    )
    result.validate()
    return result


def normalize_external_teacher_file(
    input_path: str | Path,
    output_path: str | Path,
    *,
    score_floor: float,
    person_class_ids: set[int],
    person_labels: set[str],
    dedupe_iou: float,
) -> dict[str, Any]:
    source = Path(input_path)
    metadata: dict[str, Any] | None = None
    frames: list[dict[str, Any]] = []
    seen: set[tuple[str, int]] = set()
    with source.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            if not raw.strip():
                continue
            item = json.loads(raw)
            if item.get("type") == "metadata":
                if metadata is not None:
                    raise ValueError("teacher input has multiple metadata records")
                if item.get("schema") != RAW_SCHEMA:
                    raise ValueError(f"unsupported teacher input schema: {item.get('schema')!r}")
                metadata = dict(item)
                continue
            if item.get("type") != "frame":
                raise ValueError(f"{source}:{line_number}: expected metadata/frame record")
            video = item.get("video")
            frame = item.get("frame")
            width = item.get("width")
            height = item.get("height")
            if not isinstance(video, str) or not video:
                raise ValueError(f"{source}:{line_number}: video must be non-empty")
            if not isinstance(frame, int) or frame < 0:
                raise ValueError(f"{source}:{line_number}: frame must be non-negative integer")
            if not isinstance(width, int) or not isinstance(height, int):
                raise ValueError(f"{source}:{line_number}: width/height must be integers")
            key = (video, frame)
            if key in seen:
                raise ValueError(f"duplicate teacher frame {video}#{frame}")
            seen.add(key)
            detections = normalize_teacher_detections(
                item.get("detections", []),
                width=width,
                height=height,
                score_floor=score_floor,
                person_class_ids=person_class_ids,
                person_labels=person_labels,
                dedupe_iou=dedupe_iou,
            )
            frames.append(
                {
                    "type": "frame",
                    "video": video,
                    "frame": frame,
                    "width": width,
                    "height": height,
                    "detections": [
                        {
                            "bbox": list(detection.bbox),
                            "score": detection.score,
                            "class_id": 0,
                            "label": "person",
                        }
                        for detection in detections
                    ],
                }
            )
    if metadata is None:
        raise ValueError("teacher input missing metadata")
    provenance = _provenance_from_metadata(metadata)
    corpus_revision = metadata.get("corpus_revision")
    source_revision = metadata.get("source_revision")
    if not isinstance(corpus_revision, str) or not corpus_revision:
        raise ValueError("teacher metadata corpus_revision must be non-empty")
    if not isinstance(source_revision, str) or not source_revision:
        raise ValueError("teacher metadata source_revision must be non-empty")
    source_sha256 = metadata.get("source_sha256")
    if (
        not isinstance(source_sha256, str)
        or len(source_sha256) != 64
        or any(ch not in "0123456789abcdef" for ch in source_sha256.lower())
    ):
        raise ValueError("teacher metadata source_sha256 must be a 64-character hex SHA-256")

    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    out_metadata = {
        "type": "metadata",
        "schema": TEACHER_SCHEMA,
        "corpus_revision": corpus_revision,
        "source_revision": source_revision,
        "source_sha256": source_sha256.lower(),
        "provenance": asdict(provenance),
        "normalization": {
            "score_floor": score_floor,
            "person_class_ids": sorted(person_class_ids),
            "person_labels": sorted(person_labels),
            "dedupe_iou": dedupe_iou,
        },
        "training_selection_eligible": not _is_eval_only_identifier(corpus_revision, source_revision),
    }
    with target.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(out_metadata, sort_keys=True) + "\n")
        for frame in sorted(frames, key=lambda row: (row["video"], row["frame"])):
            handle.write(json.dumps(frame, sort_keys=True) + "\n")
    return {
        "schema": TEACHER_SCHEMA,
        "frames": len(frames),
        "provenance": asdict(provenance),
        "training_selection_eligible": out_metadata["training_selection_eligible"],
    }


def load_teacher_file(path: str | Path) -> tuple[dict[str, Any], dict[tuple[str, int], list[PredictedObject]]]:
    metadata: dict[str, Any] | None = None
    predictions: dict[tuple[str, int], list[PredictedObject]] = {}
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            if not raw.strip():
                continue
            item = json.loads(raw)
            if item.get("type") == "metadata":
                if item.get("schema") != TEACHER_SCHEMA:
                    raise ValueError(f"unsupported teacher schema: {item.get('schema')!r}")
                _provenance_from_metadata(item)
                metadata = dict(item)
                continue
            if item.get("type") != "frame":
                raise ValueError(f"{path}:{line_number}: unknown teacher record type")
            key = (str(item["video"]), int(item["frame"]))
            if key in predictions:
                raise ValueError(f"duplicate teacher frame {key}")
            detections = []
            for detection in item.get("detections", []):
                bbox = tuple(float(value) for value in detection["bbox"])
                if len(bbox) != 4:
                    raise ValueError("canonical teacher bbox must have four coordinates")
                detections.append(PredictedObject(bbox, "person", float(detection["score"])))
            predictions[key] = detections
    if metadata is None:
        raise ValueError("teacher file missing metadata")
    return metadata, predictions


def _max_iou(box: tuple[float, float, float, float], truth: Sequence[GroundTruthObject]) -> float:
    values = [_iou(box, item.bbox) for item in truth if item.label == "person" and not item.ignore]
    return max(values, default=0.0)


def _diagnose_frame(
    frame: GroundTruthFrame,
    teacher: list[PredictedObject],
    baseline: list[PredictedObject],
    match_iou: float,
) -> dict[str, Any]:
    truth = list(enumerate(frame.objects))
    teacher_matches, teacher_used = _match_objects(truth, teacher, match_iou)
    baseline_matches, baseline_used = _match_objects(truth, baseline, match_iou)
    scored_truth = [
        (index, obj)
        for index, obj in truth
        if obj.label == "person" and not obj.ignore
    ]
    teacher_hit = set(teacher_matches)
    baseline_hit = set(baseline_matches)
    teacher_fp = [
        (index, pred)
        for index, pred in enumerate(teacher)
        if index not in teacher_used
    ]
    baseline_fp = [
        (index, pred)
        for index, pred in enumerate(baseline)
        if index not in baseline_used
    ]

    def gt_record(index: int, obj: GroundTruthObject) -> dict[str, Any]:
        return {
            "object_id": obj.object_id,
            "bbox": list(obj.bbox),
            "attributes": list(obj.attributes),
            "teacher_hit": index in teacher_hit,
            "baseline_hit": index in baseline_hit,
        }

    return {
        "video": frame.video,
        "frame": frame.frame,
        "tags": list(frame.tags),
        "teacher_recovers_baseline_fn": [
            gt_record(index, obj)
            for index, obj in scored_truth
            if index in teacher_hit and index not in baseline_hit
        ],
        "baseline_recovers_teacher_fn": [
            gt_record(index, obj)
            for index, obj in scored_truth
            if index in baseline_hit and index not in teacher_hit
        ],
        "missed_by_both": [
            gt_record(index, obj)
            for index, obj in scored_truth
            if index not in teacher_hit and index not in baseline_hit
        ],
        "teacher_false_positives": [
            {
                "bbox": list(pred.bbox),
                "score": pred.score,
                "max_person_iou": _max_iou(pred.bbox, frame.objects),
                "reason": "teacher_false_positive",
            }
            for _, pred in teacher_fp
        ],
        "baseline_false_positives": [
            {
                "bbox": list(pred.bbox),
                "score": pred.score,
                "max_person_iou": _max_iou(pred.bbox, frame.objects),
                "reason": "baseline_false_positive",
            }
            for _, pred in baseline_fp
        ],
    }


def _load_frame_image(video_root: str | Path, source_record: Mapping[str, Any]) -> np.ndarray:
    direct = _direct_image_path(video_root, dict(source_record))
    if direct is not None:
        frame = cv2.imread(str(direct), cv2.IMREAD_COLOR)
        if frame is None:
            raise RuntimeError(f"cannot read image: {direct}")
        return frame
    sequence = _image_sequence_path(video_root, dict(source_record))
    if sequence is not None:
        frame = cv2.imread(str(sequence), cv2.IMREAD_COLOR)
        if frame is None:
            raise RuntimeError(f"cannot read image: {sequence}")
        return frame
    raise FileNotFoundError("teacher/current comparison currently requires direct-image or image-sequence ground truth")


def compare_teacher_to_current(args: argparse.Namespace) -> dict[str, Any]:
    ground_truth = load_ground_truth(args.ground_truth)
    source_records = _load_benchmark_source_records(args.ground_truth)
    teacher_metadata, teacher_predictions = load_teacher_file(args.teacher)
    teacher_corpus = str(teacher_metadata.get("corpus_revision", ""))
    if teacher_corpus != args.corpus_revision:
        raise ValueError(
            f"teacher corpus revision mismatch: file={teacher_corpus!r} expected={args.corpus_revision!r}"
        )

    baseline_adapter = CurrentYoloAdapter(
        args.current_model,
        input_size=args.input_size,
        threshold=args.current_threshold,
        nms_iou=args.current_nms_iou,
        prefer_dml=not args.cpu,
    )
    baseline_predictions: dict[tuple[str, int], list[PredictedObject]] = {}
    diagnostics: list[dict[str, Any]] = []
    inference_calls = 0
    timing = {"preprocess_ms": 0.0, "inference_ms": 0.0, "postprocess_ms": 0.0, "wall_ms": 0.0}
    started = time.perf_counter()

    for frame in ground_truth:
        key = (frame.video, frame.frame)
        if key not in teacher_predictions:
            raise ValueError(f"teacher file missing frozen frame {frame.video}#{frame.frame}")
        source_record = source_records.get(key)
        if source_record is None:
            raise ValueError(f"ground truth missing source metadata for {frame.video}#{frame.frame}")
        image = _load_frame_image(args.video_root, source_record)
        result = baseline_adapter.detect(image)
        current = [
            PredictedObject(detection.bbox, detection.label, detection.score)
            for detection in result.detections
        ]
        baseline_predictions[key] = current
        inference_calls += result.inference_calls
        timing["preprocess_ms"] += result.preprocess_ms
        timing["inference_ms"] += result.inference_ms
        timing["postprocess_ms"] += result.postprocess_ms
        timing["wall_ms"] += result.wall_ms
        diagnostics.append(
            _diagnose_frame(
                frame,
                teacher_predictions[key],
                current,
                args.match_iou,
            )
        )

    teacher_metrics = evaluate_frames(
        ground_truth,
        teacher_predictions,
        label="person",
        iou_threshold=args.match_iou,
    )
    baseline_metrics = evaluate_frames(
        ground_truth,
        baseline_predictions,
        label="person",
        iou_threshold=args.match_iou,
    )
    training_eligible = bool(teacher_metadata.get("training_selection_eligible", False))
    summary = {
        "teacher_recovers_baseline_fn": sum(len(row["teacher_recovers_baseline_fn"]) for row in diagnostics),
        "baseline_recovers_teacher_fn": sum(len(row["baseline_recovers_teacher_fn"]) for row in diagnostics),
        "missed_by_both": sum(len(row["missed_by_both"]) for row in diagnostics),
        "teacher_false_positives": sum(len(row["teacher_false_positives"]) for row in diagnostics),
        "baseline_false_positives": sum(len(row["baseline_false_positives"]) for row in diagnostics),
    }
    hard_negative_candidates = [
        {
            "video": row["video"],
            "frame": row["frame"],
            **fp,
            "training_selection_eligible": training_eligible,
        }
        for row in diagnostics
        for fp in row["teacher_false_positives"]
    ]
    hard_negative_candidates.sort(
        key=lambda item: (
            -float(item["score"]),
            item["video"],
            int(item["frame"]),
            tuple(item["bbox"]),
        )
    )

    return {
        "schema": COMPARE_SCHEMA,
        "source_commit": args.source_commit,
        "corpus_revision": args.corpus_revision,
        "ground_truth_sha256": ground_truth_sha256(args.ground_truth),
        "teacher_metadata": teacher_metadata,
        "current_detector": {
            "model": str(args.current_model),
            "model_sha256": hashlib.sha256(Path(args.current_model).read_bytes()).hexdigest(),
            "providers": baseline_adapter.providers,
            "input_size": args.input_size,
            "threshold": args.current_threshold,
            "nms_iou": args.current_nms_iou,
        },
        "match_iou": args.match_iou,
        "teacher_metrics": teacher_metrics,
        "current_metrics": baseline_metrics,
        "diagnostic_summary": summary,
        "diagnostics": diagnostics,
        "hard_negative_candidates": hard_negative_candidates,
        "training_selection_eligible": training_eligible,
        "selection_note": (
            "Eligible only as a candidate-selection source; human review is still required."
            if training_eligible
            else "EVAL-ONLY corpus: diagnostics MUST NOT seed training/fine-tune/hard-negative data."
        ),
        "performance": {
            **timing,
            "benchmark_wall_s": time.perf_counter() - started,
            "current_inference_calls": inference_calls,
            "teacher_inference_calls": 0,
            "teacher_evidence": "precomputed_offline",
        },
    }


class TransformersDfFineOfflineTeacher:
    """Optional local-only D-FINE runtime; never imported by production paths."""

    def __init__(
        self,
        model_dir: str | Path,
        *,
        model_variant: str,
        training_recipe: str,
        model_source: str,
        model_revision: str,
        code_revision: str,
        device: str,
    ) -> None:
        root = Path(model_dir)
        if not root.is_dir():
            raise FileNotFoundError(f"local D-FINE model directory not found: {root}")
        try:
            import torch
            import transformers
            from transformers import AutoImageProcessor, DFineForObjectDetection
        except ImportError as exc:
            raise RuntimeError(
                "optional D-FINE teacher runtime requires local packages torch, Pillow and "
                "transformers with DFineForObjectDetection support; CI uses fixture contracts instead"
            ) from exc

        self._torch = torch
        self._transformers = transformers
        self.processor = AutoImageProcessor.from_pretrained(root, local_files_only=True)
        self.model = DFineForObjectDetection.from_pretrained(root, local_files_only=True).to(device)
        self.model.eval()
        self.device = device
        self.provenance = TeacherProvenance(
            model_family="D-FINE",
            model_variant=model_variant,
            training_recipe=training_recipe,
            weights_sha256=_sha256_tree(root),
            model_source=model_source,
            model_revision=model_revision,
            code_source="https://github.com/Peterande/D-FINE",
            code_revision=code_revision,
            runtime="transformers",
            runtime_version=str(transformers.__version__),
        )
        self.provenance.validate()

    def infer(self, frame_bgr: np.ndarray, threshold: float) -> list[dict[str, Any]]:
        from PIL import Image

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        with self._torch.no_grad():
            outputs = self.model(**inputs)
        target_sizes = [(image.height, image.width)]
        result = self.processor.post_process_object_detection(
            outputs,
            target_sizes=target_sizes,
            threshold=threshold,
        )[0]
        id2label = getattr(self.model.config, "id2label", {})
        detections = []
        for score, label, box in zip(result["scores"], result["labels"], result["boxes"]):
            class_id = int(label.item())
            detections.append(
                {
                    "bbox": [float(value) for value in box.tolist()],
                    "score": float(score.item()),
                    "class_id": class_id,
                    "label": str(id2label.get(class_id, class_id)),
                    "coordinate_space": "xyxy_abs",
                }
            )
        return detections


def _write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SpectraTrack vNext A1 D-FINE offline teacher research")
    sub = parser.add_subparsers(dest="command", required=True)

    normalize = sub.add_parser("normalize", help="Normalize strict external D-FINE JSONL to teacher contract")
    normalize.add_argument("--input", required=True)
    normalize.add_argument("--output", required=True)
    normalize.add_argument("--score-floor", type=float, default=0.05)
    normalize.add_argument("--person-class-id", type=int, action="append", default=[])
    normalize.add_argument("--person-label", action="append", default=list(DEFAULT_PERSON_LABELS))
    normalize.add_argument("--dedupe-iou", type=float, default=0.95)

    compare = sub.add_parser("compare-current", help="Compare precomputed teacher to current YOLO on same frames")
    compare.add_argument("--teacher", required=True)
    compare.add_argument("--ground-truth", required=True)
    compare.add_argument("--video-root", required=True)
    compare.add_argument("--current-model", required=True)
    compare.add_argument("--output", required=True)
    compare.add_argument("--source-commit", required=True)
    compare.add_argument("--corpus-revision", required=True)
    compare.add_argument("--input-size", type=int, default=960)
    compare.add_argument("--current-threshold", type=float, default=0.35)
    compare.add_argument("--current-nms-iou", type=float, default=0.45)
    compare.add_argument("--match-iou", type=float, default=0.5)
    compare.add_argument("--cpu", action="store_true")
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.command == "normalize":
        summary = normalize_external_teacher_file(
            args.input,
            args.output,
            score_floor=args.score_floor,
            person_class_ids=set(args.person_class_id),
            person_labels=set(args.person_label),
            dedupe_iou=args.dedupe_iou,
        )
        print(json.dumps(summary, sort_keys=True))
        return 0

    result = compare_teacher_to_current(args)
    _write_json(args.output, result)
    print(json.dumps(result["diagnostic_summary"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
