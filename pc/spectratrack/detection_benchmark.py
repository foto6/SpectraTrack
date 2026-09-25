from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Iterable

import cv2

from .detector import YoloOnnxDetector
from .integrity import sha256_file
from .types import BBox, Detection


def bbox_iou(a: BBox, b: BBox) -> float:
    xx1 = max(float(a[0]), float(b[0]))
    yy1 = max(float(a[1]), float(b[1]))
    xx2 = min(float(a[2]), float(b[2]))
    yy2 = min(float(a[3]), float(b[3]))
    inter = max(0.0, xx2 - xx1) * max(0.0, yy2 - yy1)
    area_a = max(0.0, float(a[2] - a[0])) * max(0.0, float(a[3] - a[1]))
    area_b = max(0.0, float(b[2] - b[0])) * max(0.0, float(b[3] - b[1]))
    return inter / max(area_a + area_b - inter, 1e-6)


def match_person_detections(
    detections: Iterable[Detection],
    truth_boxes: list[BBox],
    iou_threshold: float = 0.5,
) -> tuple[int, int, int, set[int]]:
    predictions = sorted(
        (d for d in detections if d.label.lower() == "person"),
        key=lambda d: d.score,
        reverse=True,
    )
    unmatched = set(range(len(truth_boxes)))
    matched_truth: set[int] = set()
    true_positives = 0

    for detection in predictions:
        best_index = None
        best_iou = 0.0
        for index in unmatched:
            overlap = bbox_iou(detection.bbox, truth_boxes[index])
            if overlap > best_iou:
                best_iou = overlap
                best_index = index
        if best_index is not None and best_iou >= iou_threshold:
            unmatched.remove(best_index)
            matched_truth.add(best_index)
            true_positives += 1

    false_positives = len(predictions) - true_positives
    false_negatives = len(truth_boxes) - true_positives
    return true_positives, false_positives, false_negatives, matched_truth


def person_height_bucket(box: BBox) -> str:
    height = max(0.0, float(box[3]) - float(box[1]))
    if height < 32.0:
        return "<32px"
    if height < 64.0:
        return "32-63px"
    return "64px+"


def load_manifest(path: str | Path) -> dict:
    manifest_path = Path(path)
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not isinstance(data.get("samples"), list):
        raise ValueError("manifest must be a JSON object with a samples array")
    if not data["samples"]:
        raise ValueError("manifest samples must not be empty")

    for sample_index, sample in enumerate(data["samples"]):
        if not isinstance(sample, dict) or not isinstance(sample.get("image"), str):
            raise ValueError(f"sample {sample_index} must contain an image path")
        persons = sample.get("persons", [])
        if not isinstance(persons, list):
            raise ValueError(f"sample {sample_index} persons must be an array")
        for box_index, box in enumerate(persons):
            if (
                not isinstance(box, list)
                or len(box) != 4
                or not all(isinstance(value, (int, float)) and not isinstance(value, bool) for value in box)
            ):
                raise ValueError(f"sample {sample_index} person box {box_index} must be [x1,y1,x2,y2]")
            if float(box[2]) <= float(box[0]) or float(box[3]) <= float(box[1]):
                raise ValueError(f"sample {sample_index} person box {box_index} has invalid extents")
    return data


def run_benchmark(
    model_path: str | Path,
    manifest_path: str | Path,
    *,
    mode: str = "standard",
    input_size: int = 640,
    conf: float = 0.35,
    iou: float = 0.45,
    person_conf: float = 0.12,
    person_tile_size: int = 640,
    person_tile_overlap: float = 0.20,
    person_merge_iou: float = 0.55,
    match_iou: float = 0.50,
    prefer_gpu: bool = True,
) -> dict:
    if mode not in {"standard", "people-recall"}:
        raise ValueError("mode must be standard or people-recall")
    if not 0.0 < match_iou <= 1.0:
        raise ValueError("match_iou must be in (0, 1]")

    manifest_file = Path(manifest_path)
    manifest = load_manifest(manifest_file)
    detector = YoloOnnxDetector(
        model_path,
        input_size=input_size,
        conf_threshold=conf,
        iou_threshold=iou,
        prefer_gpu=prefer_gpu,
    )

    totals = {"tp": 0, "fp": 0, "fn": 0}
    height_totals = {bucket: {"truth": 0, "matched": 0} for bucket in ("<32px", "32-63px", "64px+")}
    tag_totals: dict[str, dict[str, int]] = {}
    latencies_ms: list[float] = []

    for sample in manifest["samples"]:
        image_path = (manifest_file.parent / sample["image"]).resolve()
        frame = cv2.imread(str(image_path))
        if frame is None:
            raise RuntimeError(f"Cannot read benchmark image: {image_path}")

        truth_boxes = [tuple(float(value) for value in box) for box in sample.get("persons", [])]

        started = time.perf_counter()
        if mode == "people-recall":
            detections = detector.detect_people_recall(
                frame,
                person_threshold=person_conf,
                tile_size=person_tile_size,
                tile_overlap=person_tile_overlap,
                merge_iou_threshold=person_merge_iou,
            )
        else:
            detections = detector.detect(frame)
        latencies_ms.append((time.perf_counter() - started) * 1000.0)

        tp, fp, fn, matched_truth = match_person_detections(detections, truth_boxes, match_iou)
        totals["tp"] += tp
        totals["fp"] += fp
        totals["fn"] += fn

        for index, box in enumerate(truth_boxes):
            bucket = person_height_bucket(box)
            height_totals[bucket]["truth"] += 1
            if index in matched_truth:
                height_totals[bucket]["matched"] += 1

        for tag in sample.get("tags", []):
            key = str(tag)
            stats = tag_totals.setdefault(key, {"truth": 0, "matched": 0})
            stats["truth"] += len(truth_boxes)
            stats["matched"] += len(matched_truth)

    truth_count = totals["tp"] + totals["fn"]
    predicted_count = totals["tp"] + totals["fp"]
    recall = totals["tp"] / truth_count if truth_count else 1.0
    precision = totals["tp"] / predicted_count if predicted_count else (1.0 if truth_count == 0 else 0.0)
    total_seconds = sum(latencies_ms) / 1000.0
    sample_count = len(manifest["samples"])

    by_height = {}
    for bucket, stats in height_totals.items():
        by_height[bucket] = {
            **stats,
            "recall": stats["matched"] / stats["truth"] if stats["truth"] else None,
        }
    by_tag = {}
    for tag, stats in sorted(tag_totals.items()):
        by_tag[tag] = {
            **stats,
            "recall": stats["matched"] / stats["truth"] if stats["truth"] else None,
        }

    return {
        "model": str(Path(model_path)),
        "model_sha256": sha256_file(model_path),
        "providers": detector.providers,
        "manifest": str(manifest_file),
        "mode": mode,
        "settings": {
            "requested_input_size": input_size,
            "model_input_width": detector.input_w,
            "model_input_height": detector.input_h,
            "conf": conf,
            "iou": iou,
            "person_conf": person_conf if mode == "people-recall" else None,
            "person_tile_size": person_tile_size if mode == "people-recall" else None,
            "person_tile_overlap": person_tile_overlap if mode == "people-recall" else None,
            "person_merge_iou": person_merge_iou if mode == "people-recall" else None,
            "match_iou": match_iou,
        },
        "samples": sample_count,
        "truth_persons": truth_count,
        "true_positives": totals["tp"],
        "false_positives": totals["fp"],
        "false_negatives": totals["fn"],
        "recall": recall,
        "precision": precision,
        "false_positives_per_frame": totals["fp"] / sample_count,
        "mean_latency_ms": statistics.fmean(latencies_ms),
        "median_latency_ms": statistics.median(latencies_ms),
        "frames_per_second": sample_count / total_seconds if total_seconds > 0.0 else None,
        "vram_mb": None,
        "vram_note": "DirectML VRAM is not measured portably by this benchmark; do not infer or fabricate it.",
        "recall_by_person_height": by_height,
        "recall_by_tag": by_tag,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark person detection against annotated still frames.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", default="")
    parser.add_argument("--mode", choices=("standard", "people-recall"), default="standard")
    parser.add_argument("--input-size", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--person-conf", type=float, default=0.12)
    parser.add_argument("--person-tile-size", type=int, default=640)
    parser.add_argument("--person-tile-overlap", type=float, default=0.20)
    parser.add_argument("--person-merge-iou", type=float, default=0.55)
    parser.add_argument("--match-iou", type=float, default=0.50)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    result = run_benchmark(
        args.model,
        args.manifest,
        mode=args.mode,
        input_size=args.input_size,
        conf=args.conf,
        iou=args.iou,
        person_conf=args.person_conf,
        person_tile_size=args.person_tile_size,
        person_tile_overlap=args.person_tile_overlap,
        person_merge_iou=args.person_merge_iou,
        match_iou=args.match_iou,
        prefer_gpu=not args.cpu,
    )
    payload = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(payload + "\n", encoding="utf-8")
    print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
