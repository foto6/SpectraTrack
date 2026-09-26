from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import cv2

from spectratrack.enhancement_recall import (
    corroborate_enhanced_detections,
    detections_corroborate,
    translate_detection,
)
from spectratrack.integrity import sha256_file
from spectratrack.research.enhancement_efficiency import (
    DetectorProbe,
    apply_operation,
    assess_frame_quality,
    load_roi_manifest,
    operation_gate,
)

SCHEMA = "spectratrack-vnext-enhancement-alternates-v1"
STRICT_OPERATIONS = ("bilateral", "current_adaptive_cached")
PERSON_CONF = 0.12
WEAK_MIN = 0.12
WEAK_MAX = 0.35
CORROBORATION_IOU = 0.10


def _source_image(source_root: Path, source: str):
    path = Path(source)
    if not path.is_absolute():
        path = source_root / path
    if not path.is_file():
        raise FileNotFoundError(path)
    frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if frame is None:
        raise RuntimeError(f"cannot read source image: {path}")
    return path, frame


def _detection_json(detection) -> dict[str, Any]:
    return {
        "bbox": [float(value) for value in detection.bbox],
        "score": float(detection.score),
        "class_id": int(detection.class_id),
        "label": str(detection.label),
    }


def _validate_strict_records(records) -> None:
    seen_frames: set[tuple[str, int]] = set()
    for record in records:
        key = (record.video, record.frame)
        if key in seen_frames:
            raise ValueError(f"strict evidence manifest selects more than one ROI for frame {key}")
        seen_frames.add(key)
        if record.signals.get("weak_person") is not True:
            raise ValueError(f"{record.roi_id}: strict manifest requires weak_person=true")
        if record.source_kind != "tile" or not record.source_id:
            raise ValueError(
                f"{record.roi_id}: strict post-fusion handoff requires upstream tile source identity"
            )
        if not record.source:
            raise ValueError(
                f"{record.roi_id}: strict post-fusion handoff requires immutable source image path"
            )
        if not record.raw_support:
            raise ValueError(
                f"{record.roi_id}: strict post-fusion handoff requires frozen raw_support"
            )
        for raw in record.raw_support:
            if raw.label.lower() != "person":
                raise ValueError(f"{record.roi_id}: raw_support must contain person detections only")
            if not WEAK_MIN <= raw.score < WEAK_MAX:
                raise ValueError(
                    f"{record.roi_id}: raw_support score {raw.score} is outside [{WEAK_MIN}, {WEAK_MAX})"
                )


def run_strict_evidence(args: argparse.Namespace) -> dict[str, Any]:
    from spectratrack.detector import YoloOnnxDetector

    metadata, records = load_roi_manifest(args.roi_manifest)
    _validate_strict_records(records)
    source_root = Path(args.source_root)
    detector = YoloOnnxDetector(
        args.model,
        input_size=args.input_size,
        conf_threshold=args.conf,
        iou_threshold=args.nms_iou,
        prefer_gpu=not args.cpu,
    )
    detector._reset_policy_metrics()
    probe = DetectorProbe(detector)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    alternates: list[dict[str, Any]] = []
    quality_gate_on = 0
    enhanced_calls = 0
    operation_ms = 0.0
    enhanced_detector_ms = 0.0
    enhanced_inference_ms = 0.0
    accepted_measurements = 0
    source_hashes: dict[str, str] = {}
    started = time.perf_counter()

    for record in records:
        image_path, frame = _source_image(source_root, record.source)
        source_key = str(image_path)
        if source_key not in source_hashes:
            source_hashes[source_key] = sha256_file(image_path)

        x1, y1, x2, y2 = record.bbox
        height, width = frame.shape[:2]
        if x1 < 0 or y1 < 0 or x2 > width or y2 > height or x2 <= x1 or y2 <= y1:
            raise ValueError(
                f"{record.roi_id}: ROI {record.bbox} is outside source image {width}x{height}"
            )
        roi = frame[y1:y2, x1:x2]
        quality = assess_frame_quality(roi)
        eligible = operation_gate(args.operation, roi, quality)
        if not eligible:
            continue
        quality_gate_on += 1

        enhanced_roi, preprocess_ms = apply_operation(args.operation, roi, quality)
        operation_ms += preprocess_ms
        enhanced_local, cost = probe.detect(enhanced_roi, PERSON_CONF)
        enhanced_calls += int(cost["onnx_calls"])
        enhanced_detector_ms += float(cost["wall_ms"])
        enhanced_inference_ms += float(cost["inference_ms"])

        enhanced_full = [
            translate_detection(detection, x1, y1)
            for detection in enhanced_local
            if detection.label.lower() == "person"
        ]
        raw_support = list(record.raw_support)
        accepted = corroborate_enhanced_detections(
            raw_support,
            enhanced_full,
            min_iou=CORROBORATION_IOU,
        )

        for candidate in accepted:
            supporters = [
                raw
                for raw in raw_support
                if detections_corroborate(raw, candidate, min_iou=CORROBORATION_IOU)
            ]
            if not supporters:
                raise AssertionError("corroborated candidate lost its raw supporter")
            accepted_measurements += 1
            alternates.append(
                {
                    "type": "alternate",
                    "video": record.video,
                    "frame": record.frame,
                    "roi_id": record.roi_id,
                    "source_kind": record.source_kind,
                    "source_id": record.source_id,
                    "source_region": list(record.bbox),
                    "operation": args.operation,
                    "candidate": _detection_json(candidate),
                    "raw_support": [_detection_json(raw) for raw in supporters],
                    "semantics": {
                        "same_raw_source": True,
                        "independent_evidence_increment": 0,
                        "raw_corroborated": True,
                    },
                }
            )

    wall_seconds = time.perf_counter() - started
    result_metadata = {
        "type": "metadata",
        "schema": SCHEMA,
        "operation": args.operation,
        "source_commit": args.source_commit,
        "roi_manifest": {
            "path": str(args.roi_manifest),
            "sha256": sha256_file(args.roi_manifest),
            "metadata": metadata,
        },
        "model": {
            "path": str(args.model),
            "sha256": sha256_file(args.model),
            "providers": list(detector.providers),
            "requested_input_size": args.input_size,
            "actual_input_width": detector.input_w,
            "actual_input_height": detector.input_h,
        },
        "strict_configuration": {
            "selective_gate": "weak-person",
            "max_enhanced_rois_per_source_frame": 1,
            "raw_corroboration_required": True,
            "raw_support_source": "frozen A1 prefusion evidence; no additional raw inference",
            "person_conf": PERSON_CONF,
            "weak_score_min_inclusive": WEAK_MIN,
            "weak_score_max_exclusive": WEAK_MAX,
            "corroboration_iou": CORROBORATION_IOU,
        },
    }
    summary = {
        "type": "summary",
        "selected_weak_rois": len(records),
        "quality_gate_on_rois": quality_gate_on,
        "additional_raw_onnx_calls": 0,
        "extra_enhancement_onnx_calls": enhanced_calls,
        "accepted_alternate_measurements": accepted_measurements,
        "operation_preprocessing_ms": operation_ms,
        "enhanced_detector_ms": enhanced_detector_ms,
        "enhanced_inference_ms": enhanced_inference_ms,
        "wall_seconds": wall_seconds,
        "source_image_count": len(source_hashes),
        "source_images_sha256": {
            path: digest for path, digest in sorted(source_hashes.items())
        },
        "note": (
            "Alternate measurements preserve the original A1 source_id and must not be "
            "counted as a new independent evidence source by final fusion."
        ),
    }
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(result_metadata, sort_keys=True) + "\n")
        for item in alternates:
            handle.write(json.dumps(item, sort_keys=True) + "\n")
        handle.write(json.dumps(summary, sort_keys=True) + "\n")

    return {
        "output": str(output_path),
        "output_sha256": sha256_file(output_path),
        **{key: value for key, value in summary.items() if key != "type"},
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run locked A3 strict enhancement using frozen A1 raw support"
    )
    parser.add_argument("--roi-manifest", required=True)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--operation", required=True, choices=STRICT_OPERATIONS)
    parser.add_argument("--output", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--input-size", type=int, default=960)
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--nms-iou", type=float, default=0.45)
    parser.add_argument("--cpu", action="store_true")
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    result = run_strict_evidence(args)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
