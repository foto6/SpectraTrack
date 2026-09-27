from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import statistics
import time
from typing import Any, Iterable


from .appearance import attach_appearance
from .detector import YoloOnnxDetector
from .integrity import sha256_file
from .motion import GlobalMotionEstimator
from .qa_benchmark import PredictedObject, evaluate_frames, load_ground_truth
from .tracker import MultiObjectTracker


CANDIDATES = ("control", "tile-person-only-postprocess", "contiguous-input")
STAGE_FIELDS = (
    "decode_ms",
    "motion_ms",
    "detector_policy_ms",
    "detector_preprocess_ms",
    "detector_inference_ms",
    "detector_postprocess_ms",
    "detector_full_pass_ms",
    "detector_raw_roi_ms",
    "detector_enhanced_roi_ms",
    "enhancement_router_ops_ms",
    "fusion_ms",
    "appearance_ms",
    "tracker_ms",
    "serialization_ms",
    "frame_total_ms",
    "process_cpu_ms",
    "inference_host_wait_residual_ms",
    "system_cpu_idle_pct",
)


@dataclass(frozen=True, slots=True)
class SmokeFrame:
    video: str
    frame: int
    source: str
    source_sequence: str

    @property
    def key(self) -> str:
        return f"{self.video}#{self.frame}"


def _percentile(sorted_values: list[float], fraction: float) -> float:
    if not sorted_values:
        raise ValueError("cannot compute percentile of empty sequence")
    index = max(0, min(len(sorted_values) - 1, int(math.ceil(len(sorted_values) * fraction)) - 1))
    return float(sorted_values[index])


def distribution(values: Iterable[float | int | None]) -> dict[str, float | int | None]:
    samples = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    if not samples:
        return {
            "count": 0,
            "total": 0.0,
            "mean": None,
            "p50": None,
            "p95": None,
            "p99": None,
            "max": None,
        }
    ordered = sorted(samples)
    return {
        "count": len(samples),
        "total": float(sum(samples)),
        "mean": float(statistics.fmean(samples)),
        "p50": _percentile(ordered, 0.50),
        "p95": _percentile(ordered, 0.95),
        "p99": _percentile(ordered, 0.99),
        "max": float(ordered[-1]),
    }


def _read_json_or_jsonl(path: str | Path) -> Any:
    source = Path(path)
    text = source.read_text(encoding="utf-8")
    stripped = text.lstrip()
    if source.suffix.lower() == ".jsonl" or (stripped and not stripped.startswith(("[", "{"))):
        records: list[Any] = []
        for line_number, raw in enumerate(text.splitlines(), start=1):
            raw = raw.strip()
            if not raw or raw.startswith("#"):
                continue
            try:
                records.append(json.loads(raw))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{source}:{line_number}: invalid JSON") from exc
        return records
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        records = []
        for line_number, raw in enumerate(text.splitlines(), start=1):
            raw = raw.strip()
            if not raw or raw.startswith("#"):
                continue
            try:
                records.append(json.loads(raw))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{source}:{line_number}: invalid JSON") from exc
        return records


def _selection_items(payload: Any) -> list[Any]:
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in (
            "entries",
            "selected_frames",
            "frame_ids",
            "frames",
            "items",
            "selected",
            "frame_keys",
            "selection",
        ):
            if key in payload:
                return _selection_items(payload[key])
        if "video" in payload and "frame" in payload:
            return [payload]
    raise ValueError("selection manifest does not expose a recognized frame list")


def _selection_aliases(item: Any) -> list[str]:
    aliases: list[str] = []
    if isinstance(item, (str, int)):
        aliases.append(str(item))
        return aliases
    if not isinstance(item, dict):
        return aliases

    video = item.get("video")
    frame = item.get("frame")
    if isinstance(video, str) and isinstance(frame, int) and not isinstance(frame, bool):
        aliases.extend((f"{video}#{frame}", f"frame:{video}#{frame}"))

    for key in ("key", "frame_key", "frame_id", "id"):
        value = item.get(key)
        if isinstance(value, (str, int)):
            aliases.append(str(value))

    source = item.get("source")
    if isinstance(source, str):
        normalized = source.replace("\\", "/")
        aliases.extend((normalized, f"source:{normalized}"))

    for key in ("source_frame", "image_id"):
        value = item.get(key)
        if isinstance(value, int) and not isinstance(value, bool):
            aliases.extend((str(value), f"source_frame:{value}", f"image_id:{value}"))

    metadata = item.get("source_metadata")
    if isinstance(metadata, dict):
        image_id = metadata.get("image_id")
        if isinstance(image_id, int) and not isinstance(image_id, bool):
            aliases.extend((str(image_id), f"source_frame:{image_id}", f"image_id:{image_id}"))
    return aliases


def _row_aliases(row: dict[str, Any]) -> set[str]:
    video = row.get("video")
    frame = row.get("frame")
    aliases: set[str] = set()
    if isinstance(video, str) and isinstance(frame, int) and not isinstance(frame, bool):
        aliases.update((f"{video}#{frame}", f"frame:{video}#{frame}"))

    source = row.get("source")
    if isinstance(source, str):
        normalized = source.replace("\\", "/")
        aliases.update((normalized, f"source:{normalized}"))

    source_frame = row.get("source_frame")
    if isinstance(source_frame, int) and not isinstance(source_frame, bool):
        aliases.update((str(source_frame), f"source_frame:{source_frame}", f"image_id:{source_frame}"))

    metadata = row.get("source_metadata")
    if isinstance(metadata, dict):
        image_id = metadata.get("image_id")
        if isinstance(image_id, int) and not isinstance(image_id, bool):
            aliases.update((str(image_id), f"source_frame:{image_id}", f"image_id:{image_id}"))
    return aliases


def load_selected_image_frames(
    ground_truth_path: str | Path,
    selection_manifest_path: str | Path,
    *,
    expected_frames: int | None = 400,
) -> tuple[list[SmokeFrame], dict[str, Any]]:
    rows_payload = _read_json_or_jsonl(ground_truth_path)
    if not isinstance(rows_payload, list):
        raise ValueError("ground truth must be JSONL/list records")

    by_alias: dict[str, dict[str, Any]] = {}
    ambiguous: set[str] = set()
    for row in rows_payload:
        if not isinstance(row, dict):
            continue
        for alias in _row_aliases(row):
            previous = by_alias.get(alias)
            if previous is not None and previous is not row:
                ambiguous.add(alias)
            else:
                by_alias[alias] = row
    for alias in ambiguous:
        by_alias.pop(alias, None)

    manifest_payload = _read_json_or_jsonl(selection_manifest_path)
    items = _selection_items(manifest_payload)
    selected: list[SmokeFrame] = []
    seen: set[str] = set()
    unresolved: list[Any] = []

    for item in items:
        row = None
        for alias in _selection_aliases(item):
            row = by_alias.get(alias)
            if row is not None:
                break
        if row is None:
            unresolved.append(item)
            continue

        video = row.get("video")
        frame = row.get("frame")
        source = row.get("source")
        sequence = row.get("source_sequence")
        if not isinstance(video, str) or not isinstance(frame, int) or not isinstance(source, str):
            raise ValueError("selected ground-truth row lacks video/frame/source metadata")
        key = f"{video}#{frame}"
        if key in seen:
            raise ValueError(f"selection contains duplicate frame {key}")
        seen.add(key)
        selected.append(
            SmokeFrame(
                video=video,
                frame=frame,
                source=source,
                source_sequence=sequence if isinstance(sequence, str) and sequence else video,
            )
        )

    if unresolved:
        preview = ", ".join(repr(item)[:120] for item in unresolved[:3])
        raise ValueError(f"selection contains {len(unresolved)} unresolved frame ids: {preview}")
    if expected_frames is not None and len(selected) != expected_frames:
        raise ValueError(f"selection has {len(selected)} frames, expected exactly {expected_frames}")
    return selected, {
        "ground_truth_sha256": sha256_file(ground_truth_path),
        "selection_manifest_sha256": sha256_file(selection_manifest_path),
        "selected_frames": len(selected),
        "selection_keys_sha256": hashlib.sha256(
            ("\n".join(item.key for item in selected) + "\n").encode("utf-8")
        ).hexdigest(),
    }


class _CpuIdleSampler:
    def __init__(self) -> None:
        try:
            import psutil  # type: ignore
        except ImportError:
            self._psutil = None
        else:
            self._psutil = psutil

    @property
    def available(self) -> bool:
        return self._psutil is not None

    def snapshot(self):
        if self._psutil is None:
            return None
        return self._psutil.cpu_times()

    @staticmethod
    def _total(value: Any) -> float:
        return float(sum(float(item) for item in value))

    def idle_percent(self, before: Any, after: Any) -> float | None:
        if before is None or after is None:
            return None
        total = self._total(after) - self._total(before)
        idle = float(getattr(after, "idle", 0.0)) - float(getattr(before, "idle", 0.0))
        if total <= 0.0:
            return None
        return max(0.0, min(100.0, idle * 100.0 / total))


def _serialize_result(key: str, detections: list[Any], tracks: list[Any]) -> bytes:
    payload = {
        "key": key,
        "detections": [
            {
                "bbox": [float(value) for value in item.bbox],
                "class_id": int(item.class_id),
                "label": item.label,
                "score": float(item.score),
            }
            for item in detections
        ],
        "tracks": [
            {
                "bbox": [float(value) for value in item.bbox],
                "class_id": int(item.class_id),
                "label": item.label,
                "score": float(item.score),
                "track_id": int(item.track_id),
            }
            for item in tracks
        ],
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _frame_record(
    *,
    key: str,
    decode_ms: float,
    motion_ms: float,
    detector_ms: float,
    appearance_ms: float,
    tracker_ms: float,
    serialization_ms: float,
    frame_total_ms: float,
    process_cpu_ms: float,
    system_cpu_idle_pct: float | None,
    detector: YoloOnnxDetector,
    detections: int,
    tracks: int,
) -> dict[str, Any]:
    inference_ms = float(detector.last_stage_ms.get("inference", 0.0))
    inference_cpu_ms = float(detector.last_stage_cpu_ms.get("inference", 0.0))
    return {
        "key": key,
        "decode_ms": decode_ms,
        "motion_ms": motion_ms,
        "detector_policy_ms": detector_ms,
        "detector_preprocess_ms": float(detector.last_stage_ms.get("preprocess", 0.0)),
        "detector_inference_ms": inference_ms,
        "detector_postprocess_ms": float(detector.last_stage_ms.get("postprocess", 0.0)),
        "detector_full_pass_ms": float(detector.last_policy_ms.get("full_pass", 0.0)),
        "detector_raw_roi_ms": float(detector.last_policy_ms.get("raw_roi_passes", 0.0)),
        "detector_enhanced_roi_ms": float(detector.last_policy_ms.get("enhanced_roi_passes", 0.0)),
        "enhancement_router_ops_ms": float(detector.last_policy_ms.get("enhancement_router_ops", 0.0)),
        "fusion_ms": float(detector.last_policy_ms.get("fusion", 0.0)),
        "appearance_ms": appearance_ms,
        "tracker_ms": tracker_ms,
        "serialization_ms": serialization_ms,
        "frame_total_ms": frame_total_ms,
        "process_cpu_ms": process_cpu_ms,
        "inference_host_wait_residual_ms": max(0.0, inference_ms - inference_cpu_ms),
        "system_cpu_idle_pct": system_cpu_idle_pct,
        "onnx_calls": int(detector.last_inference_calls),
        "full_frame_calls": int(detector.last_policy_counts.get("full_frame_calls", 0)),
        "raw_roi_calls": int(detector.last_policy_counts.get("raw_roi_calls", 0)),
        "enhanced_roi_calls": int(detector.last_policy_counts.get("enhanced_roi_calls", 0)),
        "tile_count": int(detector.last_policy_counts.get("tile_count", 0)),
        "fusion_inputs": int(detector.last_policy_counts.get("fusion_inputs", detections)),
        "fusion_outputs": int(detector.last_policy_counts.get("fusion_outputs", detections)),
        "detections": detections,
        "tracks": tracks,
    }


def _stage_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    summary = {name: distribution(record.get(name) for record in records) for name in STAGE_FIELDS}
    detector_internal = {}
    for name in ("detector_preprocess_ms", "detector_inference_ms", "detector_postprocess_ms"):
        detector_internal[name] = float(summary[name]["total"] or 0.0)
    detector_total = float(summary["detector_policy_ms"]["total"] or 0.0)
    detector_internal["detector_policy_overhead_ms"] = max(
        0.0,
        detector_total - sum(detector_internal.values()),
    )
    summary["detector_internal_totals_ms"] = detector_internal

    exclusive = {
        "decode_ms": float(summary["decode_ms"]["total"] or 0.0),
        "motion_ms": float(summary["motion_ms"]["total"] or 0.0),
        "detector_policy_ms": detector_total,
        "appearance_ms": float(summary["appearance_ms"]["total"] or 0.0),
        "tracker_ms": float(summary["tracker_ms"]["total"] or 0.0),
        "serialization_ms": float(summary["serialization_ms"]["total"] or 0.0),
    }
    summary["pipeline_exclusive_totals_ms"] = exclusive
    summary["top_pipeline_costs"] = [
        {"stage": key, "total_ms": value}
        for key, value in sorted(exclusive.items(), key=lambda item: item[1], reverse=True)[:3]
    ]
    summary["top_detector_internal_costs"] = [
        {"stage": key, "total_ms": value}
        for key, value in sorted(detector_internal.items(), key=lambda item: item[1], reverse=True)[:3]
    ]
    return summary


def _projection(records: list[dict[str, Any]], full_frames: int = 5000) -> dict[str, Any]:
    if not records:
        raise ValueError("cannot project empty profile")
    scale = full_frames / len(records)
    frame_total_s = sum(float(item["frame_total_ms"]) for item in records) / 1000.0
    projected_wall_s = frame_total_s * scale
    return {
        "method": "linear smoke mean projection",
        "smoke_frames": len(records),
        "full_frames": full_frames,
        "scale": scale,
        "projected_wall_seconds": projected_wall_s,
        "projected_wall_hours": projected_wall_s / 3600.0,
        "note": "Projection assumes the frozen smoke slice is representative of FULL5000 stage mix.",
    }


def amdahl_upper_bound(*, affected_ms: float, total_ms: float) -> float:
    """Maximum speedup if the affected work were made free."""
    if total_ms <= 0.0:
        raise ValueError("total_ms must be > 0")
    if affected_ms < 0.0 or affected_ms > total_ms:
        raise ValueError("affected_ms must be in [0, total_ms]")
    remaining = total_ms - affected_ms
    return float("inf") if remaining <= 0.0 else total_ms / remaining


def research_cost_model(
    *,
    smoke_wall_seconds: float,
    projected_full_seconds: float,
    candidate_count: int,
    promoted_count: int,
) -> dict[str, Any]:
    if smoke_wall_seconds <= 0.0 or projected_full_seconds <= 0.0:
        raise ValueError("wall times must be > 0")
    if candidate_count <= 0:
        raise ValueError("candidate_count must be > 0")
    if promoted_count < 0 or promoted_count > candidate_count:
        raise ValueError("promoted_count must be between 0 and candidate_count")
    naive = candidate_count * projected_full_seconds
    smoke_first = candidate_count * smoke_wall_seconds + promoted_count * projected_full_seconds
    savings = naive - smoke_first
    return {
        "candidate_count": candidate_count,
        "promoted_count": promoted_count,
        "naive_full5000_seconds": naive,
        "smoke_first_seconds": smoke_first,
        "saved_seconds": savings,
        "saved_fraction": savings / naive,
    }


def _candidate_flags(candidate: str) -> tuple[bool, bool]:
    if candidate not in CANDIDATES:
        raise ValueError(f"unknown candidate: {candidate}")
    return candidate == "tile-person-only-postprocess", candidate == "contiguous-input"


def profile_selected_images(
    *,
    ground_truth_path: str | Path,
    selection_manifest_path: str | Path,
    source_root: str | Path,
    model_path: str | Path,
    candidate: str,
    source_commit: str,
    expected_frames: int = 400,
    input_size: int = 960,
    person_conf: float = 0.12,
    tile_size: int = 640,
    overlap: float = 0.20,
    merge_iou: float = 0.55,
    prefer_gpu: bool = True,
    candidate_count: int = 2,
    promoted_count: int = 1,
) -> dict[str, Any]:
    if len(source_commit) != 40 or any(char not in "0123456789abcdef" for char in source_commit.lower()):
        raise ValueError("source_commit must be a full 40-hex commit SHA")
    frames, provenance = load_selected_image_frames(
        ground_truth_path,
        selection_manifest_path,
        expected_frames=expected_frames,
    )
    root = Path(source_root)
    model = Path(model_path)
    if not model.is_file():
        raise FileNotFoundError(model)

    tile_person_only, contiguous_input = _candidate_flags(candidate)
    detector = YoloOnnxDetector(
        model,
        input_size=input_size,
        conf_threshold=0.35,
        iou_threshold=0.45,
        prefer_gpu=prefer_gpu,
    )
    detector.force_contiguous_input = contiguous_input

    first = root / frames[0].source
    warmup_frame = __import__("cv2").imread(str(first))
    if warmup_frame is None:
        raise FileNotFoundError(first)
    detector.detect(warmup_frame)

    motion = GlobalMotionEstimator()
    tracker = MultiObjectTracker()
    cpu_idle = _CpuIdleSampler()
    records: list[dict[str, Any]] = []
    output_digest = hashlib.sha256()
    predictions: dict[tuple[str, int], list[PredictedObject]] = {}
    previous_sequence: str | None = None
    wall_started = time.perf_counter()

    import cv2

    for item in frames:
        if item.source_sequence != previous_sequence:
            motion = GlobalMotionEstimator()
            tracker = MultiObjectTracker()
            previous_sequence = item.source_sequence

        frame_wall_started = time.perf_counter()
        frame_cpu_started = time.process_time()
        cpu_before = cpu_idle.snapshot()

        decode_started = time.perf_counter()
        image_path = root / item.source
        frame = cv2.imread(str(image_path))
        decode_ms = (time.perf_counter() - decode_started) * 1000.0
        if frame is None:
            raise FileNotFoundError(image_path)

        motion_started = time.perf_counter()
        camera = motion.update(frame)
        motion_ms = (time.perf_counter() - motion_started) * 1000.0

        detector_started = time.perf_counter()
        detections = detector.detect_people_recall(
            frame,
            person_threshold=person_conf,
            tile_size=tile_size,
            tile_overlap=overlap,
            merge_iou_threshold=merge_iou,
            enhancement_mode="off",
            tile_person_only_postprocess=tile_person_only,
        )
        detector_ms = (time.perf_counter() - detector_started) * 1000.0

        predictions[(item.video, item.frame)] = [
            PredictedObject(tuple(det.bbox), det.label, float(det.score)) for det in detections
        ]

        appearance_started = time.perf_counter()
        attach_appearance(frame, detections)
        appearance_ms = (time.perf_counter() - appearance_started) * 1000.0

        tracker_started = time.perf_counter()
        camera_motion = (camera.dx, camera.dy) if camera.valid else (0.0, 0.0)
        camera_transform = camera.affine if camera.valid else None
        tracks = tracker.update(
            detections,
            camera_motion=camera_motion,
            camera_transform=camera_transform,
        )
        tracker_ms = (time.perf_counter() - tracker_started) * 1000.0

        serialization_started = time.perf_counter()
        encoded = _serialize_result(item.key, detections, tracks)
        output_digest.update(encoded)
        output_digest.update(b"\n")
        serialization_ms = (time.perf_counter() - serialization_started) * 1000.0

        cpu_after = cpu_idle.snapshot()
        frame_total_ms = (time.perf_counter() - frame_wall_started) * 1000.0
        process_cpu_ms = (time.process_time() - frame_cpu_started) * 1000.0
        records.append(
            _frame_record(
                key=item.key,
                decode_ms=decode_ms,
                motion_ms=motion_ms,
                detector_ms=detector_ms,
                appearance_ms=appearance_ms,
                tracker_ms=tracker_ms,
                serialization_ms=serialization_ms,
                frame_total_ms=frame_total_ms,
                process_cpu_ms=process_cpu_ms,
                system_cpu_idle_pct=cpu_idle.idle_percent(cpu_before, cpu_after),
                detector=detector,
                detections=len(detections),
                tracks=len(tracks),
            )
        )

    wall_seconds = time.perf_counter() - wall_started
    selected_keys = {item.key for item in frames}
    gt = [
        item
        for item in load_ground_truth(ground_truth_path)
        if f"{item.video}#{item.frame}" in selected_keys
    ]
    metrics = evaluate_frames(gt, predictions, None, label="person", iou_threshold=0.5)
    quality = {
        "tracking_supported": False,
        "tp": metrics["tp"],
        "fp": metrics["fp"],
        "fn": metrics["fn"],
        "gt": metrics["gt"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "matched_ground_truth": metrics["matched_ground_truth"],
        "missed_ground_truth": metrics["missed_ground_truth"],
    }

    projection = _projection(records, 5000)
    counts = {
        "onnx_calls": sum(int(item["onnx_calls"]) for item in records),
        "full_frame_calls": sum(int(item["full_frame_calls"]) for item in records),
        "raw_roi_calls": sum(int(item["raw_roi_calls"]) for item in records),
        "enhanced_roi_calls": sum(int(item["enhanced_roi_calls"]) for item in records),
    }
    return {
        "schema": "spectratrack-performance-smoke-v1",
        "source_commit": source_commit,
        "candidate": candidate,
        "candidate_locked": {
            "tile_person_only_postprocess": tile_person_only,
            "force_contiguous_input": contiguous_input,
            "enhancement_mode": "off",
        },
        "provenance": {
            **provenance,
            "ground_truth": str(ground_truth_path),
            "selection_manifest": str(selection_manifest_path),
            "source_root": str(source_root),
            "model": str(model),
            "model_sha256": sha256_file(model),
            "providers": detector.providers,
        },
        "settings": {
            "input_size": input_size,
            "person_conf": person_conf,
            "tile_size": tile_size,
            "tile_overlap": overlap,
            "merge_iou": merge_iou,
            "prefer_gpu": prefer_gpu,
        },
        "frames": len(records),
        "wall_seconds": wall_seconds,
        "processing_fps": len(records) / wall_seconds if wall_seconds > 0 else None,
        "output_sha256": output_digest.hexdigest(),
        "quality": quality,
        "counts": counts,
        "counts_per_frame": {
            key: distribution(item[key] for item in records)
            for key in ("onnx_calls", "full_frame_calls", "raw_roi_calls", "enhanced_roi_calls")
        },
        "stages": _stage_summary(records),
        "frames_detail": records,
        "cpu_idle_sampling": {
            "available": cpu_idle.available,
            "source": "psutil.cpu_times" if cpu_idle.available else None,
            "reason": None if cpu_idle.available else "psutil is not installed in this runtime",
        },
        "gpu_utilization": {
            "value_pct": None,
            "reason": (
                "ONNX Runtime DirectML exposes provider selection but not portable per-process GPU utilization. "
                "No utilization value is guessed."
            ),
        },
        "transfer_sync": {
            "metric": "inference_host_wait_residual_ms",
            "note": (
                "session.run is synchronous; wall minus process CPU is reported as a host-wait residual that "
                "includes accelerator execution, transfers and synchronization and must not be read as transfer-only cost."
            ),
        },
        "full5000_projection": projection,
        "research_cost_model": research_cost_model(
            smoke_wall_seconds=wall_seconds,
            projected_full_seconds=float(projection["projected_wall_seconds"]),
            candidate_count=candidate_count,
            promoted_count=promoted_count,
        ),
    }


def profile_video(
    *,
    video_path: str | Path,
    model_path: str | Path,
    candidate: str,
    source_commit: str,
    max_frames: int,
    input_size: int = 960,
    person_conf: float = 0.12,
    tile_size: int = 640,
    overlap: float = 0.20,
    merge_iou: float = 0.55,
    prefer_gpu: bool = True,
) -> dict[str, Any]:
    import cv2

    video = Path(video_path)
    model = Path(model_path)
    if not video.is_file():
        raise FileNotFoundError(video)
    if not model.is_file():
        raise FileNotFoundError(model)
    if max_frames <= 0:
        raise ValueError("max_frames must be > 0")

    tile_person_only, contiguous_input = _candidate_flags(candidate)
    detector = YoloOnnxDetector(model, input_size=input_size, prefer_gpu=prefer_gpu)
    detector.force_contiguous_input = contiguous_input
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {video}")
    ok, warmup = cap.read()
    if not ok or warmup is None:
        cap.release()
        raise RuntimeError("video produced no frames")
    detector.detect(warmup)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    motion = GlobalMotionEstimator()
    tracker = MultiObjectTracker()
    cpu_idle = _CpuIdleSampler()
    records: list[dict[str, Any]] = []
    output_digest = hashlib.sha256()
    wall_started = time.perf_counter()
    try:
        for frame_index in range(max_frames):
            frame_wall_started = time.perf_counter()
            frame_cpu_started = time.process_time()
            cpu_before = cpu_idle.snapshot()

            decode_started = time.perf_counter()
            ok, frame = cap.read()
            decode_ms = (time.perf_counter() - decode_started) * 1000.0
            if not ok or frame is None:
                break

            motion_started = time.perf_counter()
            camera = motion.update(frame)
            motion_ms = (time.perf_counter() - motion_started) * 1000.0

            detector_started = time.perf_counter()
            detections = detector.detect_people_recall(
                frame,
                person_threshold=person_conf,
                tile_size=tile_size,
                tile_overlap=overlap,
                merge_iou_threshold=merge_iou,
                enhancement_mode="off",
                tile_person_only_postprocess=tile_person_only,
            )
            detector_ms = (time.perf_counter() - detector_started) * 1000.0

            appearance_started = time.perf_counter()
            attach_appearance(frame, detections)
            appearance_ms = (time.perf_counter() - appearance_started) * 1000.0

            tracker_started = time.perf_counter()
            camera_motion = (camera.dx, camera.dy) if camera.valid else (0.0, 0.0)
            camera_transform = camera.affine if camera.valid else None
            tracks = tracker.update(detections, camera_motion=camera_motion, camera_transform=camera_transform)
            tracker_ms = (time.perf_counter() - tracker_started) * 1000.0

            serialization_started = time.perf_counter()
            encoded = _serialize_result(f"{video.name}#{frame_index}", detections, tracks)
            output_digest.update(encoded)
            output_digest.update(b"\n")
            serialization_ms = (time.perf_counter() - serialization_started) * 1000.0

            cpu_after = cpu_idle.snapshot()
            frame_total_ms = (time.perf_counter() - frame_wall_started) * 1000.0
            process_cpu_ms = (time.process_time() - frame_cpu_started) * 1000.0
            records.append(
                _frame_record(
                    key=f"{video.name}#{frame_index}",
                    decode_ms=decode_ms,
                    motion_ms=motion_ms,
                    detector_ms=detector_ms,
                    appearance_ms=appearance_ms,
                    tracker_ms=tracker_ms,
                    serialization_ms=serialization_ms,
                    frame_total_ms=frame_total_ms,
                    process_cpu_ms=process_cpu_ms,
                    system_cpu_idle_pct=cpu_idle.idle_percent(cpu_before, cpu_after),
                    detector=detector,
                    detections=len(detections),
                    tracks=len(tracks),
                )
            )
    finally:
        cap.release()

    wall_seconds = time.perf_counter() - wall_started
    return {
        "schema": "spectratrack-performance-video-probe-v1",
        "source_commit": source_commit,
        "candidate": candidate,
        "candidate_locked": {
            "tile_person_only_postprocess": tile_person_only,
            "force_contiguous_input": contiguous_input,
            "enhancement_mode": "off",
        },
        "provenance": {
            "video": str(video),
            "video_sha256": sha256_file(video),
            "model": str(model),
            "model_sha256": sha256_file(model),
            "providers": detector.providers,
        },
        "settings": {
            "input_size": input_size,
            "person_conf": person_conf,
            "tile_size": tile_size,
            "tile_overlap": overlap,
            "merge_iou": merge_iou,
            "prefer_gpu": prefer_gpu,
        },
        "frames": len(records),
        "wall_seconds": wall_seconds,
        "processing_fps": len(records) / wall_seconds if wall_seconds > 0 else None,
        "output_sha256": output_digest.hexdigest(),
        "quality": None,
        "counts": {
            "onnx_calls": sum(int(item["onnx_calls"]) for item in records),
            "full_frame_calls": sum(int(item["full_frame_calls"]) for item in records),
            "raw_roi_calls": sum(int(item["raw_roi_calls"]) for item in records),
            "enhanced_roi_calls": sum(int(item["enhanced_roi_calls"]) for item in records),
        },
        "counts_per_frame": {
            key: distribution(item[key] for item in records)
            for key in ("onnx_calls", "full_frame_calls", "raw_roi_calls", "enhanced_roi_calls")
        },
        "stages": _stage_summary(records),
        "frames_detail": records,
        "cpu_idle_sampling": {
            "available": cpu_idle.available,
            "source": "psutil.cpu_times" if cpu_idle.available else None,
            "reason": None if cpu_idle.available else "psutil is not installed in this runtime",
        },
        "gpu_utilization": {
            "value_pct": None,
            "reason": "DirectML provider exposes no portable per-process utilization metric; no value is guessed.",
        },
        "transfer_sync": {
            "metric": "inference_host_wait_residual_ms",
            "note": "Residual includes accelerator execution, transfer and synchronization.",
        },
    }


def compare_profiles(
    control: dict[str, Any],
    candidate: dict[str, Any],
    *,
    minimum_speedup: float = 1.10,
) -> dict[str, Any]:
    if minimum_speedup <= 1.0:
        raise ValueError("minimum_speedup must be > 1")
    for path in (
        ("provenance", "model_sha256"),
        ("settings", "input_size"),
        ("settings", "person_conf"),
        ("settings", "tile_size"),
        ("settings", "tile_overlap"),
        ("settings", "merge_iou"),
    ):
        left: Any = control
        right: Any = candidate
        for key in path:
            left = left[key]
            right = right[key]
        if left != right:
            raise ValueError(f"profile mismatch at {'.'.join(path)}")
    for optional in ("selection_manifest_sha256", "selection_keys_sha256", "video_sha256"):
        left = control.get("provenance", {}).get(optional)
        right = candidate.get("provenance", {}).get(optional)
        if left is not None or right is not None:
            if left != right:
                raise ValueError(f"profile mismatch at provenance.{optional}")

    control_wall = float(control["wall_seconds"])
    candidate_wall = float(candidate["wall_seconds"])
    speedup = control_wall / candidate_wall if candidate_wall > 0 else float("inf")
    output_equal = control.get("output_sha256") == candidate.get("output_sha256")

    quality_delta = None
    if control.get("quality") is not None and candidate.get("quality") is not None:
        cq = control["quality"]
        kq = candidate["quality"]
        quality_delta = {
            "recall": float(kq["recall"]) - float(cq["recall"]),
            "precision": float(kq["precision"]) - float(cq["precision"]),
            "fn": int(kq["fn"]) - int(cq["fn"]),
            "fp": int(kq["fp"]) - int(cq["fp"]),
        }

    meaningful = speedup >= minimum_speedup
    semantics_ok = output_equal
    promotion = "promote_to_full" if meaningful and semantics_ok else "stop_on_smoke"
    reasons = []
    if not meaningful:
        reasons.append(f"speedup {speedup:.3f}x is below {minimum_speedup:.3f}x gate")
    if not semantics_ok:
        reasons.append("output hash differs from control")
    return {
        "schema": "spectratrack-performance-smoke-compare-v1",
        "control_candidate": control.get("candidate"),
        "candidate": candidate.get("candidate"),
        "control_wall_seconds": control_wall,
        "candidate_wall_seconds": candidate_wall,
        "speedup_x": speedup,
        "minimum_speedup_x": minimum_speedup,
        "output_hash_equal": output_equal,
        "quality_delta": quality_delta,
        "promotion": promotion,
        "reasons": reasons,
    }


def _dump(payload: Any, output: str | None) -> None:
    encoded = json.dumps(payload, indent=2, sort_keys=True)
    if output:
        target = Path(output)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)


def main() -> int:
    parser = argparse.ArgumentParser(description="SpectraTrack A4 deterministic smoke profiling")
    sub = parser.add_subparsers(dest="command", required=True)

    selected = sub.add_parser("profile-selected-images")
    selected.add_argument("--ground-truth", required=True)
    selected.add_argument("--selection-manifest", required=True)
    selected.add_argument("--source-root", required=True)
    selected.add_argument("--model", required=True)
    selected.add_argument("--candidate", choices=CANDIDATES, default="control")
    selected.add_argument("--source-commit", required=True)
    selected.add_argument("--expected-frames", type=int, default=400)
    selected.add_argument("--input-size", type=int, default=960)
    selected.add_argument("--person-conf", type=float, default=0.12)
    selected.add_argument("--tile-size", type=int, default=640)
    selected.add_argument("--overlap", type=float, default=0.20)
    selected.add_argument("--merge-iou", type=float, default=0.55)
    selected.add_argument("--candidate-count", type=int, default=2)
    selected.add_argument("--promoted-count", type=int, default=1)
    selected.add_argument("--cpu", action="store_true")
    selected.add_argument("--output")

    video = sub.add_parser("profile-video")
    video.add_argument("--video", required=True)
    video.add_argument("--model", required=True)
    video.add_argument("--candidate", choices=CANDIDATES, default="control")
    video.add_argument("--source-commit", required=True)
    video.add_argument("--max-frames", type=int, default=125)
    video.add_argument("--input-size", type=int, default=960)
    video.add_argument("--person-conf", type=float, default=0.12)
    video.add_argument("--tile-size", type=int, default=640)
    video.add_argument("--overlap", type=float, default=0.20)
    video.add_argument("--merge-iou", type=float, default=0.55)
    video.add_argument("--cpu", action="store_true")
    video.add_argument("--output")

    compare = sub.add_parser("compare")
    compare.add_argument("--control", required=True)
    compare.add_argument("--candidate", required=True)
    compare.add_argument("--minimum-speedup", type=float, default=1.10)
    compare.add_argument("--output")

    args = parser.parse_args()
    if args.command == "profile-selected-images":
        result = profile_selected_images(
            ground_truth_path=args.ground_truth,
            selection_manifest_path=args.selection_manifest,
            source_root=args.source_root,
            model_path=args.model,
            candidate=args.candidate,
            source_commit=args.source_commit,
            expected_frames=args.expected_frames,
            input_size=args.input_size,
            person_conf=args.person_conf,
            tile_size=args.tile_size,
            overlap=args.overlap,
            merge_iou=args.merge_iou,
            prefer_gpu=not args.cpu,
            candidate_count=args.candidate_count,
            promoted_count=args.promoted_count,
        )
        _dump(result, args.output)
        return 0
    if args.command == "profile-video":
        result = profile_video(
            video_path=args.video,
            model_path=args.model,
            candidate=args.candidate,
            source_commit=args.source_commit,
            max_frames=args.max_frames,
            input_size=args.input_size,
            person_conf=args.person_conf,
            tile_size=args.tile_size,
            overlap=args.overlap,
            merge_iou=args.merge_iou,
            prefer_gpu=not args.cpu,
        )
        _dump(result, args.output)
        return 0

    control = json.loads(Path(args.control).read_text(encoding="utf-8"))
    candidate = json.loads(Path(args.candidate).read_text(encoding="utf-8"))
    result = compare_profiles(control, candidate, minimum_speedup=args.minimum_speedup)
    _dump(result, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
