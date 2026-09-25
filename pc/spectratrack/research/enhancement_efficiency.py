from __future__ import annotations

import argparse
import json
import math
import subprocess
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Iterable

import cv2
import numpy as np

from spectratrack.enhance import adaptive_analysis_frame, assess_frame_quality
from spectratrack.enhancement_recall import corroborate_enhanced_detections
from spectratrack.integrity import sha256_file
from spectratrack.qa_benchmark import GroundTruthFrame, load_ground_truth
from spectratrack.types import Detection

SCHEMA = "spectratrack-vnext-enhancement-profile-v1"
ROI_SCHEMA = "spectratrack-vnext-enhancement-roi-v1"
OPERATIONS = (
    "gamma",
    "clahe",
    "gamma_clahe",
    "bilateral",
    "sharpen",
    "current_adaptive",
    "current_adaptive_cached",
)
SELECTIVE_GATES = (
    "quality",
    "dark",
    "blur",
    "compression",
    "weak-person",
    "known-track",
    "uncertainty",
    "quality-or-evidence",
)
QUALITY_THRESHOLDS = {"darkness": 0.22, "blur": 0.18, "compression": 0.20, "noise": 0.24}


@dataclass(frozen=True)
class RoiRecord:
    video: str
    frame: int
    roi_id: str
    bbox: tuple[int, int, int, int]
    signals: dict[str, bool]


@dataclass(frozen=True)
class QualitySnapshot:
    values: dict[str, float]
    stage_ms: dict[str, float]


def _elapsed_ms(started: float) -> float:
    return (time.perf_counter() - started) * 1000.0


def _git_head() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=2,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None


def _validate_frame(frame: np.ndarray) -> None:
    if frame is None or frame.size == 0 or frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError("frame must be a non-empty BGR image")


def quality_snapshot(
    frame: np.ndarray,
    *,
    max_side: int = 640,
    compression_full_resolution: bool = True,
) -> QualitySnapshot:
    """Instrument the current quality formulas without changing production behavior."""
    _validate_frame(frame)
    if max_side <= 0:
        raise ValueError("max_side must be positive")

    stage_ms: dict[str, float] = {}
    started = time.perf_counter()
    gray_full = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    stage_ms["grayscale"] = _elapsed_ms(started)
    height, width = gray_full.shape

    started = time.perf_counter()
    scale = min(1.0, float(max_side) / max(height, width))
    if scale < 1.0:
        gray = cv2.resize(
            gray_full,
            (max(1, int(round(width * scale))), max(1, int(round(height * scale)))),
            interpolation=cv2.INTER_AREA,
        )
    else:
        gray = gray_full
    stage_ms["reduce"] = _elapsed_ms(started)

    started = time.perf_counter()
    gray_f = gray.astype(np.float32)
    mean_luma = float(np.mean(gray_f)) / 255.0
    darkness = float(np.clip((0.42 - mean_luma) / 0.42, 0.0, 1.0))
    stage_ms["darkness"] = _elapsed_ms(started)

    started = time.perf_counter()
    lap_var = float(cv2.Laplacian(gray, cv2.CV_32F).var())
    blur = float(np.clip((120.0 - lap_var) / 120.0, 0.0, 1.0))
    stage_ms["blur"] = _elapsed_ms(started)

    started = time.perf_counter()
    smooth = cv2.GaussianBlur(gray, (3, 3), 0.0)
    residual = cv2.absdiff(gray, smooth)
    noise_level = float(np.median(residual))
    noise = float(np.clip((noise_level - 1.5) / 16.0, 0.0, 1.0))
    stage_ms["noise"] = _elapsed_ms(started)

    started = time.perf_counter()
    compression_gray = gray_full if compression_full_resolution else gray
    comp_h, comp_w = compression_gray.shape
    block_gray = compression_gray.astype(np.float32)
    block_samples: list[float] = []
    inner_samples: list[float] = []

    if comp_w > 16:
        boundaries = np.arange(8, comp_w, 8)
        if len(boundaries):
            block_samples.append(
                float(np.mean(np.abs(block_gray[:, boundaries] - block_gray[:, boundaries - 1])))
            )
        inner = np.arange(4, comp_w, 8)
        if len(inner):
            inner_samples.append(
                float(np.mean(np.abs(block_gray[:, inner] - block_gray[:, inner - 1])))
            )

    if comp_h > 16:
        boundaries = np.arange(8, comp_h, 8)
        if len(boundaries):
            block_samples.append(
                float(np.mean(np.abs(block_gray[boundaries, :] - block_gray[boundaries - 1, :])))
            )
        inner = np.arange(4, comp_h, 8)
        if len(inner):
            inner_samples.append(
                float(np.mean(np.abs(block_gray[inner, :] - block_gray[inner - 1, :])))
            )

    block_edge = float(np.mean(block_samples)) if block_samples else 0.0
    inner_edge = float(np.mean(inner_samples)) if inner_samples else block_edge
    compression = float(
        np.clip((block_edge - inner_edge) / max(inner_edge + 4.0, 1.0), 0.0, 1.0)
    )
    stage_ms["compression"] = _elapsed_ms(started)

    started = time.perf_counter()
    short_side = float(min(height, width))
    low_resolution = float(np.clip((720.0 - short_side) / 720.0, 0.0, 1.0))
    stage_ms["low_resolution"] = _elapsed_ms(started)

    return QualitySnapshot(
        {
            "blur": blur,
            "darkness": darkness,
            "compression": compression,
            "low_resolution": low_resolution,
            "noise": noise,
        },
        stage_ms,
    )


def _has_structure(frame: np.ndarray) -> bool:
    return float(np.std(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))) >= 8.0


def operation_gate(operation: str, frame: np.ndarray, quality: dict[str, float]) -> bool:
    if operation not in OPERATIONS:
        raise ValueError(f"unknown operation: {operation}")
    dark = quality["darkness"] >= QUALITY_THRESHOLDS["darkness"]
    bilateral = (
        quality["noise"] >= QUALITY_THRESHOLDS["noise"]
        or quality["compression"] >= QUALITY_THRESHOLDS["compression"]
    )
    sharpen = (
        _has_structure(frame)
        and quality["blur"] >= QUALITY_THRESHOLDS["blur"]
        and quality["noise"] < 0.62
        and quality["compression"] < 0.70
    )
    if operation in {"gamma", "clahe", "gamma_clahe"}:
        return dark
    if operation == "bilateral":
        return bilateral
    if operation == "sharpen":
        return sharpen
    return dark or bilateral or sharpen


def _gamma(frame: np.ndarray, quality: dict[str, float]) -> np.ndarray:
    gamma = 1.0 - min(0.32, quality["darkness"] * 0.28)
    lut = np.clip(
        np.power(np.arange(256, dtype=np.float32) / 255.0, gamma) * 255.0,
        0,
        255,
    ).astype(np.uint8)
    return cv2.LUT(frame, lut)


def _clahe(frame: np.ndarray, quality: dict[str, float]) -> np.ndarray:
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    lightness, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(
        clipLimit=1.8 + quality["darkness"] * 0.8,
        tileGridSize=(8, 8),
    )
    return cv2.cvtColor(cv2.merge([clahe.apply(lightness), a, b]), cv2.COLOR_LAB2BGR)


def _current_adaptive_from_quality(
    frame: np.ndarray,
    quality: dict[str, float],
) -> np.ndarray:
    """Reproduce the current adaptive operation sequence using one cached quality map."""
    out = frame
    if quality["darkness"] >= QUALITY_THRESHOLDS["darkness"]:
        out = _clahe(_gamma(out, quality), quality)

    if (
        quality["noise"] >= QUALITY_THRESHOLDS["noise"]
        or quality["compression"] >= QUALITY_THRESHOLDS["compression"]
    ):
        severity = max(quality["noise"], quality["compression"])
        diameter = 5 if severity < 0.65 else 7
        out = cv2.bilateralFilter(out, diameter, 24, 24)

    if (
        _has_structure(frame)
        and quality["blur"] >= QUALITY_THRESHOLDS["blur"]
        and quality["noise"] < 0.62
        and quality["compression"] < 0.70
    ):
        amount = min(0.38, 0.12 + quality["blur"] * 0.28)
        blurred = cv2.GaussianBlur(out, (0, 0), 0.9)
        out = cv2.addWeighted(out, 1.0 + amount, blurred, -amount, 0)
    return out


def apply_operation(
    operation: str,
    frame: np.ndarray,
    quality: dict[str, float],
) -> tuple[np.ndarray, float]:
    """Apply one candidate to raw pixels; candidates are never chained."""
    _validate_frame(frame)
    started = time.perf_counter()
    if operation == "gamma":
        out = _gamma(frame, quality)
    elif operation == "clahe":
        out = _clahe(frame, quality)
    elif operation == "gamma_clahe":
        out = _clahe(_gamma(frame, quality), quality)
    elif operation == "bilateral":
        severity = max(quality["noise"], quality["compression"])
        diameter = 5 if severity < 0.65 else 7
        out = cv2.bilateralFilter(frame, diameter, 24, 24)
    elif operation == "sharpen":
        amount = min(0.38, 0.12 + quality["blur"] * 0.28)
        blurred = cv2.GaussianBlur(frame, (0, 0), 0.9)
        out = cv2.addWeighted(frame, 1.0 + amount, blurred, -amount, 0)
    elif operation == "current_adaptive":
        out, _quality, _operations = adaptive_analysis_frame(frame)
    elif operation == "current_adaptive_cached":
        out = _current_adaptive_from_quality(frame, quality)
    else:
        raise ValueError(f"unknown operation: {operation}")
    return out, _elapsed_ms(started)


def selective_gate(
    gate: str,
    quality: dict[str, float],
    signals: dict[str, bool],
) -> bool:
    if gate not in SELECTIVE_GATES:
        raise ValueError(f"unknown selective gate: {gate}")
    dark = quality["darkness"] >= QUALITY_THRESHOLDS["darkness"]
    blur = quality["blur"] >= QUALITY_THRESHOLDS["blur"]
    compression = quality["compression"] >= QUALITY_THRESHOLDS["compression"]
    quality_issue = dark or blur or compression or quality["noise"] >= QUALITY_THRESHOLDS["noise"]

    if gate == "quality":
        return quality_issue
    if gate == "dark":
        return dark
    if gate == "blur":
        return blur
    if gate == "compression":
        return compression
    if gate == "weak-person":
        return bool(signals.get("weak_person", False))
    if gate == "known-track":
        return bool(signals.get("known_track", False))
    if gate == "uncertainty":
        return bool(signals.get("uncertainty", False))
    return quality_issue or any(
        bool(signals.get(name, False))
        for name in ("weak_person", "known_track", "uncertainty")
    )


def load_roi_manifest(path: str | Path) -> tuple[dict[str, Any], list[RoiRecord]]:
    metadata: dict[str, Any] = {}
    records: list[RoiRecord] = []
    seen_ids: set[str] = set()
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                item = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON on line {line_number}") from exc
            if item.get("type") == "metadata":
                if metadata:
                    raise ValueError("ROI manifest contains more than one metadata record")
                if item.get("schema") != ROI_SCHEMA:
                    raise ValueError(f"unsupported ROI manifest schema: {item.get('schema')!r}")
                metadata = dict(item)
                continue
            if item.get("type") != "roi":
                raise ValueError(f"line {line_number}: type must be metadata or roi")

            video = item.get("video")
            frame = item.get("frame")
            roi_id = item.get("roi_id")
            bbox = item.get("bbox")
            signals = item.get("signals", {})
            if not isinstance(video, str) or not video:
                raise ValueError(f"line {line_number}: video must be a non-empty string")
            if isinstance(frame, bool) or not isinstance(frame, int) or frame < 0:
                raise ValueError(f"line {line_number}: frame must be a non-negative integer")
            if not isinstance(roi_id, str) or not roi_id:
                raise ValueError(f"line {line_number}: roi_id must be a non-empty string")
            if roi_id in seen_ids:
                raise ValueError(f"duplicate roi_id: {roi_id}")
            if (
                not isinstance(bbox, list)
                or len(bbox) != 4
                or any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in bbox)
            ):
                raise ValueError(f"line {line_number}: bbox must be [x1,y1,x2,y2]")
            x1, y1, x2, y2 = (int(round(float(value))) for value in bbox)
            if x1 < 0 or y1 < 0 or x2 <= x1 or y2 <= y1:
                raise ValueError(f"line {line_number}: bbox must have positive non-negative area")
            if not isinstance(signals, dict) or any(not isinstance(value, bool) for value in signals.values()):
                raise ValueError(f"line {line_number}: signals must map names to booleans")
            seen_ids.add(roi_id)
            records.append(RoiRecord(video, frame, roi_id, (x1, y1, x2, y2), dict(signals)))

    if not metadata:
        raise ValueError("ROI manifest is missing metadata")
    if not records:
        raise ValueError("ROI manifest contains no roi records")
    return metadata, records


def _read_frames(
    video_root: Path,
    records: Iterable[RoiRecord],
) -> tuple[dict[tuple[str, int], np.ndarray], dict[str, float], dict[str, str]]:
    requested: dict[str, set[int]] = defaultdict(set)
    for record in records:
        requested[record.video].add(record.frame)

    frames: dict[tuple[str, int], np.ndarray] = {}
    fps_by_video: dict[str, float] = {}
    hashes: dict[str, str] = {}
    for video, frame_ids in sorted(requested.items()):
        path = video_root / video
        if not path.is_file():
            raise FileNotFoundError(f"video is missing: {path}")
        hashes[video] = sha256_file(path)
        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            raise RuntimeError(f"cannot open video: {path}")
        fps_by_video[video] = float(capture.get(cv2.CAP_PROP_FPS))
        try:
            for frame_index in sorted(frame_ids):
                capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                ok, frame = capture.read()
                if not ok or frame is None:
                    raise RuntimeError(f"cannot read {video} frame {frame_index}")
                frames[(video, frame_index)] = frame
        finally:
            capture.release()
    return frames, fps_by_video, hashes


def _sampled_source_seconds(records: Iterable[RoiRecord], fps_by_video: dict[str, float]) -> float | None:
    frames_by_video: dict[str, set[int]] = defaultdict(set)
    for record in records:
        frames_by_video[record.video].add(record.frame)
    total = 0.0
    for video, frames in frames_by_video.items():
        fps = fps_by_video.get(video, 0.0)
        if fps <= 0:
            return None
        total += len(frames) / fps
    return total


def _crop(frame: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    x1, y1, x2, y2 = bbox
    height, width = frame.shape[:2]
    if x2 > width or y2 > height:
        raise ValueError(f"ROI {bbox} exceeds frame size {width}x{height}")
    return frame[y1:y2, x1:x2]


def _local_truth(
    truth: GroundTruthFrame | None,
    record: RoiRecord,
    label: str,
) -> list[tuple[str, tuple[float, float, float, float]]]:
    if truth is None:
        return []
    x1, y1, x2, y2 = record.bbox
    output: list[tuple[str, tuple[float, float, float, float]]] = []
    for index, obj in enumerate(truth.objects):
        if obj.ignore or obj.label.lower() != label.lower():
            continue
        bx1, by1, bx2, by2 = obj.bbox
        center_x = (bx1 + bx2) * 0.5
        center_y = (by1 + by2) * 0.5
        if not (x1 <= center_x < x2 and y1 <= center_y < y2):
            continue
        object_id = obj.object_id if obj.object_id is not None else f"anon-{index}"
        key = f"{record.video}#{record.frame}#{object_id}"
        output.append((key, (bx1 - x1, by1 - y1, bx2 - x1, by2 - y1)))
    return output


def _bbox_iou(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
) -> float:
    xx1 = max(a[0], b[0])
    yy1 = max(a[1], b[1])
    xx2 = min(a[2], b[2])
    yy2 = min(a[3], b[3])
    inter = max(0.0, xx2 - xx1) * max(0.0, yy2 - yy1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return inter / max(area_a + area_b - inter, 1e-6)


def _match_truth(
    truth: list[tuple[str, tuple[float, float, float, float]]],
    detections: Iterable[Detection],
    match_iou: float,
) -> tuple[dict[str, tuple[float, Detection, tuple[float, float, float, float]]], int]:
    detections = list(detections)
    candidates: list[tuple[float, int, int]] = []
    for truth_index, (_key, truth_box) in enumerate(truth):
        for detection_index, detection in enumerate(detections):
            iou = _bbox_iou(truth_box, detection.bbox)
            if iou >= match_iou:
                candidates.append((iou, truth_index, detection_index))
    candidates.sort(reverse=True)

    matches: dict[str, tuple[float, Detection, tuple[float, float, float, float]]] = {}
    used_truth: set[int] = set()
    used_detections: set[int] = set()
    for iou, truth_index, detection_index in candidates:
        if truth_index in used_truth or detection_index in used_detections:
            continue
        used_truth.add(truth_index)
        used_detections.add(detection_index)
        key, truth_box = truth[truth_index]
        matches[key] = (iou, detections[detection_index], truth_box)
    return matches, len(detections) - len(used_detections)


def _merge_best(
    target: dict[str, tuple[float, Detection, tuple[float, float, float, float]]],
    source: dict[str, tuple[float, Detection, tuple[float, float, float, float]]],
) -> None:
    for key, value in source.items():
        if key not in target or value[0] > target[key][0]:
            target[key] = value


def _jitter(
    matches: dict[str, tuple[float, Detection, tuple[float, float, float, float]]],
) -> dict[str, float | int | None]:
    grouped: dict[tuple[str, str], list[tuple[int, tuple[float, float, float, float]]]] = defaultdict(list)
    for key, (_iou, detection, truth_box) in matches.items():
        parts = key.rsplit("#", 2)
        if len(parts) != 3 or parts[2].startswith("anon-"):
            continue
        video, frame_text, object_id = parts
        frame_index = int(frame_text)
        gx1, gy1, gx2, gy2 = truth_box
        px1, py1, px2, py2 = detection.bbox
        gt_width = max(gx2 - gx1, 1.0)
        gt_height = max(gy2 - gy1, 1.0)
        residual = (
            (((px1 + px2) - (gx1 + gx2)) * 0.5) / gt_width,
            (((py1 + py2) - (gy1 + gy2)) * 0.5) / gt_height,
            math.log(max(px2 - px1, 1.0) / gt_width),
            math.log(max(py2 - py1, 1.0) / gt_height),
        )
        grouped[(video, object_id)].append((frame_index, residual))

    center_steps: list[float] = []
    size_steps: list[float] = []
    for observations in grouped.values():
        observations.sort()
        for (_frame_a, a), (_frame_b, b) in zip(observations, observations[1:]):
            center_steps.append(math.hypot(b[0] - a[0], b[1] - a[1]))
            size_steps.append(math.hypot(b[2] - a[2], b[3] - a[3]))
    return {
        "center_residual_step_mean": mean(center_steps) if center_steps else None,
        "size_residual_step_mean": mean(size_steps) if size_steps else None,
        "steps": len(center_steps),
    }


def _optional_delta(candidate: float | None, baseline: float | None) -> float | None:
    if candidate is None or baseline is None:
        return None
    return candidate - baseline


class DetectorProbe:
    def __init__(self, detector: Any) -> None:
        self.detector = detector

    def detect(self, frame: np.ndarray, threshold: float) -> tuple[list[Detection], dict[str, float]]:
        before_calls = int(getattr(self.detector, "last_inference_calls", 0))
        before_stages = dict(getattr(self.detector, "last_stage_ms", {}))
        started = time.perf_counter()
        thresholds = dict(getattr(self.detector, "class_thresholds", {}))
        thresholds["person"] = float(threshold)
        detections = self.detector._detect_once(frame, thresholds)
        after_calls = int(getattr(self.detector, "last_inference_calls", 0))
        after_stages = dict(getattr(self.detector, "last_stage_ms", {}))
        return [d for d in detections if d.label.lower() == "person"], {
            "wall_ms": _elapsed_ms(started),
            "inference_ms": after_stages.get("inference", 0.0) - before_stages.get("inference", 0.0),
            "onnx_calls": float(after_calls - before_calls),
        }


def _new_accumulator() -> dict[str, Any]:
    return {
        "eligible": 0,
        "selective": 0,
        "affected": 0,
        "preprocess_ms": 0.0,
        "enhanced_detector_ms": 0.0,
        "enhanced_inference_ms": 0.0,
        "enhanced_calls": 0,
        "raw_fp": 0,
        "candidate_fp": 0,
        "raw_matches": {},
        "candidate_matches": {},
    }


def _truth_index(
    ground_truth: str | None,
) -> tuple[dict[tuple[str, int], GroundTruthFrame], str | None]:
    if not ground_truth:
        return {}, None
    items = load_ground_truth(ground_truth)
    return {(item.video, item.frame): item for item in items}, sha256_file(ground_truth)


def _summarize_operation(
    data: dict[str, Any],
    *,
    roi_count: int,
    raw_calls: int,
    raw_ms: float,
    raw_inference_ms: float,
    quality_ms: float,
    sampled_source_seconds: float | None,
    has_gt: bool,
) -> dict[str, Any]:
    raw_matches = data["raw_matches"]
    candidate_matches = data["candidate_matches"]
    raw_keys = set(raw_matches)
    candidate_keys = set(candidate_matches)
    common = raw_keys & candidate_keys
    iou_delta = [candidate_matches[key][0] - raw_matches[key][0] for key in common]
    raw_jitter = _jitter(raw_matches)
    candidate_jitter = _jitter(candidate_matches)

    attributed_ms = quality_ms + raw_ms + data["preprocess_ms"] + data["enhanced_detector_ms"]
    per_source_second = None
    if sampled_source_seconds and sampled_source_seconds > 0:
        per_source_second = attributed_ms / 1000.0 / sampled_source_seconds

    return {
        "activation": {
            "operation_gate_rois": data["eligible"],
            "operation_gate_frequency": data["eligible"] / roi_count,
            "selective_gate_rois": data["selective"],
            "selective_gate_frequency": data["selective"] / roi_count,
            "affected_rois": data["affected"],
            "affected_frequency": data["affected"] / roi_count,
        },
        "cost": {
            "quality_assessment_ms_attributed": quality_ms,
            "raw_probe_detector_ms_attributed": raw_ms,
            "raw_probe_inference_ms_attributed": raw_inference_ms,
            "operation_preprocessing_ms": data["preprocess_ms"],
            "enhanced_detector_ms": data["enhanced_detector_ms"],
            "enhanced_inference_ms": data["enhanced_inference_ms"],
            "raw_probe_calls": raw_calls,
            "enhanced_inference_calls": data["enhanced_calls"],
            "total_inference_calls_if_run_independently": raw_calls + data["enhanced_calls"],
            "additional_detector_calls_vs_raw_probe": data["enhanced_calls"],
            "attributed_total_ms": attributed_ms,
            "processing_seconds_per_sampled_source_second": per_source_second,
        },
        "quality": {
            "ground_truth_available": has_gt,
            "recovered_gt_persons": len(candidate_keys - raw_keys) if has_gt else None,
            "lost_gt_persons": len(raw_keys - candidate_keys) if has_gt else None,
            "raw_matched_gt_persons": len(raw_keys) if has_gt else None,
            "candidate_matched_gt_persons": len(candidate_keys) if has_gt else None,
            "raw_fp_observations_pre_fusion": data["raw_fp"] if has_gt else None,
            "candidate_fp_observations_pre_fusion": data["candidate_fp"] if has_gt else None,
            "new_fp_observation_delta_pre_fusion": (
                data["candidate_fp"] - data["raw_fp"] if has_gt else None
            ),
            "mean_best_match_iou_delta": mean(iou_delta) if has_gt and iou_delta else None,
            "bbox_jitter": {
                "raw": raw_jitter if has_gt else None,
                "candidate": candidate_jitter if has_gt else None,
                "center_residual_step_delta": (
                    _optional_delta(
                        candidate_jitter["center_residual_step_mean"],
                        raw_jitter["center_residual_step_mean"],
                    )
                    if has_gt
                    else None
                ),
                "size_residual_step_delta": (
                    _optional_delta(
                        candidate_jitter["size_residual_step_mean"],
                        raw_jitter["size_residual_step_mean"],
                    )
                    if has_gt
                    else None
                ),
            },
        },
    }


def run_profile(args: argparse.Namespace) -> dict[str, Any]:
    from spectratrack.detector import YoloOnnxDetector

    manifest_meta, records = load_roi_manifest(args.roi_manifest)
    frames, fps_by_video, video_hashes = _read_frames(Path(args.video_root), records)
    sampled_source_seconds = _sampled_source_seconds(records, fps_by_video)
    truth_by_frame, truth_hash = _truth_index(args.ground_truth)
    source_commit = args.source_commit or _git_head()
    if not source_commit:
        raise RuntimeError("--source-commit is required when git HEAD cannot be resolved")

    detector = YoloOnnxDetector(
        args.model,
        input_size=args.input_size,
        conf_threshold=args.conf,
        iou_threshold=args.nms_iou,
        prefer_gpu=not args.cpu,
    )
    detector._reset_policy_metrics()
    probe = DetectorProbe(detector)
    accumulators = {operation: _new_accumulator() for operation in args.operations}

    quality_ms = 0.0
    quality_stage_ms: dict[str, float] = defaultdict(float)
    raw_ms = 0.0
    raw_inference_ms = 0.0
    raw_calls = 0
    started = time.perf_counter()

    for record in records:
        roi = _crop(frames[(record.video, record.frame)], record.bbox)

        quality_started = time.perf_counter()
        quality = assess_frame_quality(roi)
        quality_ms += _elapsed_ms(quality_started)
        snapshot = quality_snapshot(roi)
        for name, value in snapshot.stage_ms.items():
            quality_stage_ms[name] += value

        raw_probes, raw_cost = probe.detect(roi, args.probe_conf)
        raw_ms += raw_cost["wall_ms"]
        raw_inference_ms += raw_cost["inference_ms"]
        raw_calls += int(raw_cost["onnx_calls"])
        raw_strong = [d for d in raw_probes if d.score >= args.person_conf]

        truth = _local_truth(
            truth_by_frame.get((record.video, record.frame)),
            record,
            args.label,
        )
        raw_matches, raw_fp = _match_truth(truth, raw_strong, args.match_iou)

        for operation in args.operations:
            data = accumulators[operation]
            _merge_best(data["raw_matches"], raw_matches)
            data["raw_fp"] += raw_fp

            eligible = operation_gate(operation, roi, quality)
            selected = selective_gate(args.selective_gate, quality, record.signals)
            data["eligible"] += int(eligible)
            data["selective"] += int(selected)

            enhanced_accepted: list[Detection] = []
            if eligible and selected:
                data["affected"] += 1
                enhanced_roi, operation_ms = apply_operation(operation, roi, quality)
                data["preprocess_ms"] += operation_ms
                enhanced, enhanced_cost = probe.detect(enhanced_roi, args.person_conf)
                data["enhanced_detector_ms"] += enhanced_cost["wall_ms"]
                data["enhanced_inference_ms"] += enhanced_cost["inference_ms"]
                data["enhanced_calls"] += int(enhanced_cost["onnx_calls"])
                enhanced_accepted = corroborate_enhanced_detections(
                    raw_probes,
                    enhanced,
                    min_iou=args.corroboration_iou,
                )

            candidate = raw_strong + enhanced_accepted
            candidate_matches, candidate_fp = _match_truth(truth, candidate, args.match_iou)
            _merge_best(data["candidate_matches"], candidate_matches)
            data["candidate_fp"] += candidate_fp

    wall_seconds = time.perf_counter() - started
    operations = {
        operation: _summarize_operation(
            data,
            roi_count=len(records),
            raw_calls=raw_calls,
            raw_ms=raw_ms,
            raw_inference_ms=raw_inference_ms,
            quality_ms=quality_ms,
            sampled_source_seconds=sampled_source_seconds,
            has_gt=bool(args.ground_truth),
        )
        for operation, data in accumulators.items()
    }
    return {
        "schema": SCHEMA,
        "experiment_id": args.experiment_id,
        "role": "A3-enhancement",
        "source_commit": source_commit,
        "immutable_baseline": args.immutable_baseline,
        "roi_manifest": {
            "path": str(args.roi_manifest),
            "sha256": sha256_file(args.roi_manifest),
            "metadata": manifest_meta,
            "roi_count": len(records),
        },
        "ground_truth": {
            "path": str(args.ground_truth) if args.ground_truth else None,
            "sha256": truth_hash,
            "corpus_revision": args.corpus_revision,
        },
        "model": {
            "path": str(args.model),
            "sha256": sha256_file(args.model),
            "providers": list(detector.providers),
            "requested_input_size": args.input_size,
            "actual_input_width": detector.input_w,
            "actual_input_height": detector.input_h,
        },
        "videos": {
            video: {"sha256": video_hashes[video], "fps": fps_by_video[video]}
            for video in sorted(video_hashes)
        },
        "settings": {
            "person_conf": args.person_conf,
            "probe_conf": args.probe_conf,
            "conf": args.conf,
            "nms_iou": args.nms_iou,
            "match_iou": args.match_iou,
            "corroboration_iou": args.corroboration_iou,
            "selective_gate": args.selective_gate,
            "operations": list(args.operations),
            "provider_preference": "CPU" if args.cpu else "DirectML-then-CPU",
            "tile_geometry_owned_by_a3": False,
            "final_fusion_owned_by_a3": False,
            "raw_corroboration_required": True,
        },
        "quality_router": {
            "assessment_calls": len(records),
            "assessment_total_ms": quality_ms,
            "component_ms": dict(quality_stage_ms),
            "quality_map_reused_across_operations": True,
            "naive_assessment_calls_without_reuse": len(records) * len(args.operations),
        },
        "performance": {
            "wall_seconds_for_multi_candidate_experiment": wall_seconds,
            "sampled_source_seconds": sampled_source_seconds,
            "actual_low_level_onnx_calls_across_shared_experiment": int(detector.last_inference_calls),
            "note": (
                "Candidate tables attribute the shared raw-probe pass to each candidate. "
                "Full-frame detector calls and final fusion remain A1/A4-owned."
            ),
        },
        "operations": operations,
    }


def run_quality_audit(args: argparse.Namespace) -> dict[str, Any]:
    metadata, records = load_roi_manifest(args.roi_manifest)
    frames, fps_by_video, video_hashes = _read_frames(Path(args.video_root), records)
    source_commit = args.source_commit or _git_head()
    if not source_commit:
        raise RuntimeError("--source-commit is required when git HEAD cannot be resolved")

    current_ms = 0.0
    activation: dict[str, int] = defaultdict(int)
    candidates: dict[int, dict[str, Any]] = {
        size: {
            "ms": 0.0,
            "error": defaultdict(float),
            "flips": defaultdict(int),
            "stage_ms": defaultdict(float),
        }
        for size in args.max_sides
    }

    for record in records:
        roi = _crop(frames[(record.video, record.frame)], record.bbox)
        started = time.perf_counter()
        current = assess_frame_quality(roi)
        current_ms += _elapsed_ms(started)

        for signal, threshold in QUALITY_THRESHOLDS.items():
            activation[signal] += int(current[signal] >= threshold)

        for size in args.max_sides:
            started = time.perf_counter()
            snapshot = quality_snapshot(
                roi,
                max_side=size,
                compression_full_resolution=False,
            )
            data = candidates[size]
            data["ms"] += _elapsed_ms(started)
            for stage, value in snapshot.stage_ms.items():
                data["stage_ms"][stage] += value
            for signal, value in snapshot.values.items():
                data["error"][signal] += abs(value - current[signal])
            for signal, threshold in QUALITY_THRESHOLDS.items():
                current_gate = current[signal] >= threshold
                candidate_gate = snapshot.values[signal] >= threshold
                data["flips"][signal] += int(current_gate != candidate_gate)

    count = len(records)
    return {
        "schema": SCHEMA,
        "experiment_id": args.experiment_id,
        "role": "A3-enhancement",
        "source_commit": source_commit,
        "immutable_baseline": args.immutable_baseline,
        "roi_manifest": {
            "path": str(args.roi_manifest),
            "sha256": sha256_file(args.roi_manifest),
            "metadata": metadata,
            "roi_count": count,
        },
        "videos": {
            video: {"sha256": video_hashes[video], "fps": fps_by_video[video]}
            for video in sorted(video_hashes)
        },
        "current_router": {
            "assessment_total_ms": current_ms,
            "activation_frequency": {
                signal: activation[signal] / count for signal in QUALITY_THRESHOLDS
            },
        },
        "reduced_candidates": {
            str(size): {
                "assessment_total_ms": data["ms"],
                "mean_abs_error": {
                    signal: value / count for signal, value in data["error"].items()
                },
                "gate_flip_frequency": {
                    signal: data["flips"][signal] / count for signal in QUALITY_THRESHOLDS
                },
                "component_ms": dict(data["stage_ms"]),
            }
            for size, data in candidates.items()
        },
        "notes": [
            "Reduced candidates are profiling-only; production thresholds are unchanged.",
            "Reduced compression is measured on the reduced representation.",
            "Temporal quality-map reuse across frames is not evaluated here.",
        ],
    }


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _print_profile(payload: dict[str, Any]) -> None:
    print("operation\taffected\tprep_ms\tenh_calls\ttotal_calls\trecovered\tlost\tnew_fp_pre_fusion")
    for name, result in payload["operations"].items():
        activation = result["activation"]
        cost = result["cost"]
        quality = result["quality"]
        print(
            f"{name}\t{activation['affected_rois']}\t{cost['operation_preprocessing_ms']:.3f}\t"
            f"{cost['enhanced_inference_calls']}\t{cost['total_inference_calls_if_run_independently']}\t"
            f"{quality['recovered_gt_persons']}\t{quality['lost_gt_persons']}\t"
            f"{quality['new_fp_observation_delta_pre_fusion']}"
        )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="SpectraTrack vNext A3 enhancement efficiency research profiler"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    audit = sub.add_parser("quality-audit")
    audit.add_argument("--roi-manifest", required=True)
    audit.add_argument("--video-root", required=True)
    audit.add_argument("--output", required=True)
    audit.add_argument("--experiment-id", required=True)
    audit.add_argument("--source-commit")
    audit.add_argument(
        "--immutable-baseline",
        default="3ebc4d50213593cac62b97399447fccf6bbc1755",
    )
    audit.add_argument("--max-sides", nargs="+", type=int, default=[320, 160])

    profile = sub.add_parser("profile")
    profile.add_argument("--roi-manifest", required=True)
    profile.add_argument("--video-root", required=True)
    profile.add_argument("--ground-truth")
    profile.add_argument("--corpus-revision")
    profile.add_argument("--model", required=True)
    profile.add_argument("--output", required=True)
    profile.add_argument("--experiment-id", required=True)
    profile.add_argument("--source-commit")
    profile.add_argument(
        "--immutable-baseline",
        default="3ebc4d50213593cac62b97399447fccf6bbc1755",
    )
    profile.add_argument("--input-size", type=int, default=640)
    profile.add_argument("--conf", type=float, default=0.35)
    profile.add_argument("--nms-iou", type=float, default=0.45)
    profile.add_argument("--person-conf", type=float, default=0.12)
    profile.add_argument("--probe-conf", type=float, default=0.08)
    profile.add_argument("--corroboration-iou", type=float, default=0.10)
    profile.add_argument("--match-iou", type=float, default=0.5)
    profile.add_argument("--label", default="person")
    profile.add_argument("--cpu", action="store_true")
    profile.add_argument("--operations", nargs="+", choices=OPERATIONS, default=list(OPERATIONS))
    profile.add_argument("--selective-gate", choices=SELECTIVE_GATES, default="quality")
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    if args.command == "quality-audit":
        if any(size <= 0 for size in args.max_sides):
            parser.error("--max-sides values must be positive")
        result = run_quality_audit(args)
        _write_json(args.output, result)
        print(json.dumps(result["current_router"], indent=2, sort_keys=True))
        return 0

    if not 0.0 < args.probe_conf <= args.person_conf <= 1.0:
        parser.error("require 0 < --probe-conf <= --person-conf <= 1")
    if not 0.0 <= args.corroboration_iou <= 1.0:
        parser.error("--corroboration-iou must be in [0, 1]")
    if not 0.0 < args.match_iou <= 1.0:
        parser.error("--match-iou must be in (0, 1]")
    if args.ground_truth and not args.corpus_revision:
        parser.error("--corpus-revision is required when --ground-truth is supplied")

    result = run_profile(args)
    _write_json(args.output, result)
    _print_profile(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
