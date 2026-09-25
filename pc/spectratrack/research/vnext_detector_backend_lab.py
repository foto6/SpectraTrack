from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import time
from typing import Any

import cv2
import numpy as np
import onnxruntime as ort

from ..detector import YoloOnnxDetector
from ..integrity import sha256_file
from ..qa_benchmark import PredictedObject, evaluate_frames, ground_truth_sha256, load_ground_truth
from ..types import Detection


@dataclass(frozen=True, slots=True)
class BackendSpec:
    backend_id: str
    family: str
    code_license: str
    weights_license: str
    onnx_contract: str
    preprocessing: str
    postprocessing: str
    directml_status: str


BACKEND_SPECS = {
    "current-yolo": BackendSpec(
        "current-yolo",
        "SpectraTrack current YOLO-compatible ONNX baseline",
        "SpectraTrack adapter code: MIT; upstream model/export license must be recorded separately",
        "model-specific; do not infer from ONNX filename",
        "existing SpectraTrack fixed-size YOLO raw or end-to-end output contract",
        "letterbox 114; BGR->RGB; float32 / 255; NCHW",
        "existing decoder-local class-aware NMS",
        "baseline path prefers DirectML on Windows and falls back to CPU",
    ),
    "rf-detr": BackendSpec(
        "rf-detr",
        "RF-DETR core Nano-Large research candidate",
        "Apache-2.0 for core training/inference code per upstream documentation snapshot",
        "core Nano-Large documented Apache-2.0; exact checkpoint provenance still required",
        "single NCHW image input; raw normalized cxcywh boxes plus class logits",
        "RGB; exact half-pixel bilinear resize without antialias; [0,1]; ImageNet mean/std; NCHW",
        "per-class sigmoid; optional background slot exclusion; global query/class top-k; no added NMS",
        "ONNX Runtime documented upstream; DirectML on target export is UNVERIFIED until local probe",
    ),
    "rt-detrv2": BackendSpec(
        "rt-detrv2",
        "official RT-DETRv2 PyTorch research candidate",
        "Apache-2.0 repository",
        "exact checkpoint provenance/license must be recorded before comparison",
        "inputs images + orig_target_sizes; outputs labels + boxes + scores; official export opset 16",
        "RGB resize to export input; float32 / 255; NCHW",
        "postprocessor is exported into ONNX; consume labels/xyxy boxes/scores; no added NMS",
        "ONNX Runtime documented upstream; DirectML on target export is UNVERIFIED until local probe",
    ),
}


@dataclass(frozen=True, slots=True)
class AdapterResult:
    detections: tuple[Detection, ...]
    preprocess_ms: float
    inference_ms: float
    postprocess_ms: float
    wall_ms: float
    inference_calls: int


def _provider_order(prefer_dml: bool) -> list[str]:
    available = ort.get_available_providers()
    providers: list[str] = []
    if prefer_dml and "DmlExecutionProvider" in available:
        providers.append("DmlExecutionProvider")
    if "CPUExecutionProvider" in available:
        providers.append("CPUExecutionProvider")
    return providers or available


def _session(model_path: str | Path, prefer_dml: bool) -> ort.InferenceSession:
    providers = _provider_order(prefer_dml)
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    if "DmlExecutionProvider" in providers:
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.enable_mem_pattern = False
    return ort.InferenceSession(str(model_path), sess_options=options, providers=providers)


def _half_pixel_resize_rgb(image: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """Bilinear resize with half-pixel source coordinates and no antialias."""
    in_h, in_w = image.shape[:2]
    if (in_h, in_w) == (out_h, out_w):
        return image.astype(np.float32, copy=True)

    y = (np.arange(out_h, dtype=np.float32) + 0.5) * (in_h / out_h) - 0.5
    x = (np.arange(out_w, dtype=np.float32) + 0.5) * (in_w / out_w) - 0.5
    y = np.clip(y, 0.0, in_h - 1.0)
    x = np.clip(x, 0.0, in_w - 1.0)
    y0 = np.floor(y).astype(np.int32)
    x0 = np.floor(x).astype(np.int32)
    y1 = np.minimum(y0 + 1, in_h - 1)
    x1 = np.minimum(x0 + 1, in_w - 1)
    wy = (y - y0).reshape(-1, 1, 1)
    wx = (x - x0).reshape(1, -1, 1)

    top = image[y0][:, x0] * (1.0 - wx) + image[y0][:, x1] * wx
    bottom = image[y1][:, x0] * (1.0 - wx) + image[y1][:, x1] * wx
    return (top * (1.0 - wy) + bottom * wy).astype(np.float32)


def _rf_preprocess(frame_bgr: np.ndarray, input_h: int, input_w: int) -> np.ndarray:
    rgb = frame_bgr[:, :, ::-1].astype(np.float32)
    resized = _half_pixel_resize_rgb(rgb, input_h, input_w) / 255.0
    mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
    normalized = (resized - mean) / std
    return np.transpose(normalized, (2, 0, 1))[None].astype(np.float32)


def _rtdetr_preprocess(frame_bgr: np.ndarray, input_h: int, input_w: int) -> np.ndarray:
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (input_w, input_h), interpolation=cv2.INTER_LINEAR)
    return np.transpose(resized.astype(np.float32) / 255.0, (2, 0, 1))[None]


def _clip_box(box: np.ndarray, width: int, height: int) -> tuple[float, float, float, float] | None:
    x1 = float(np.clip(box[0], 0.0, width))
    y1 = float(np.clip(box[1], 0.0, height))
    x2 = float(np.clip(box[2], 0.0, width))
    y2 = float(np.clip(box[3], 0.0, height))
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2, y2)


def decode_rfdetr_person(
    boxes_cxcywh: np.ndarray,
    logits: np.ndarray,
    *,
    width: int,
    height: int,
    threshold: float,
    person_class_id: int,
    background_class_id: int | None = -1,
    num_select: int | None = None,
) -> list[Detection]:
    boxes = np.asarray(boxes_cxcywh, dtype=np.float32)
    scores_raw = np.asarray(logits, dtype=np.float32)
    if boxes.ndim != 2 or boxes.shape[1] != 4:
        raise ValueError(f"RF-DETR boxes must be [Q,4], got {boxes.shape}")
    if scores_raw.ndim != 2 or scores_raw.shape[0] != boxes.shape[0]:
        raise ValueError(f"RF-DETR logits must be [Q,C] with same Q, got {scores_raw.shape}")
    if not 0 <= person_class_id < scores_raw.shape[1]:
        raise ValueError("person_class_id outside RF-DETR logits")
    if background_class_id is not None:
        resolved_background = background_class_id % scores_raw.shape[1]
        if resolved_background == person_class_id:
            raise ValueError("person_class_id cannot equal background_class_id")

    scores = 1.0 / (1.0 + np.exp(-np.clip(scores_raw, -88.0, 88.0)))
    flat: list[tuple[float, int, int]] = []
    for query_index in range(scores.shape[0]):
        for class_id in range(scores.shape[1]):
            if background_class_id is not None and class_id == background_class_id % scores.shape[1]:
                continue
            flat.append((float(scores[query_index, class_id]), query_index, class_id))
    flat.sort(reverse=True)
    cap = boxes.shape[0] if num_select is None else int(num_select)
    selected = [item for item in flat[:cap] if item[0] > threshold and item[2] == person_class_id]

    detections: list[Detection] = []
    scale = np.asarray([width, height, width, height], dtype=np.float32)
    for score, query_index, _ in selected:
        cx, cy, bw, bh = boxes[query_index]
        xyxy = np.asarray([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], dtype=np.float32) * scale
        clipped = _clip_box(xyxy, width, height)
        if clipped is not None:
            detections.append(Detection(clipped, score, 0, "person"))
    return detections


class CurrentYoloAdapter:
    def __init__(
        self,
        model_path: str | Path,
        *,
        input_size: int,
        threshold: float,
        nms_iou: float,
        prefer_dml: bool,
    ) -> None:
        self.model_path = str(model_path)
        self.detector = YoloOnnxDetector(
            model_path,
            input_size=input_size,
            conf_threshold=threshold,
            iou_threshold=nms_iou,
            prefer_gpu=prefer_dml,
        )

    @property
    def providers(self) -> list[str]:
        return list(self.detector.providers)

    def contract(self) -> dict[str, Any]:
        return {
            "inputs": [{"name": self.detector.input_name, "shape": list(self.detector.input.shape)}],
            "outputs": [
                {"name": item.name, "shape": list(item.shape)}
                for item in self.detector.session.get_outputs()
            ],
            "providers": self.providers,
            "actual_input": [self.detector.input_h, self.detector.input_w],
        }

    def detect(self, frame: np.ndarray) -> AdapterResult:
        started = time.perf_counter()
        detections = tuple(item for item in self.detector.detect(frame) if item.label.lower() == "person")
        wall = (time.perf_counter() - started) * 1000.0
        stages = self.detector.last_stage_ms
        return AdapterResult(
            detections,
            float(stages.get("preprocess", 0.0)),
            float(stages.get("inference", 0.0)),
            float(stages.get("postprocess", 0.0)),
            wall,
            int(self.detector.last_inference_calls),
        )


class RfDetrOnnxAdapter:
    def __init__(
        self,
        model_path: str | Path,
        *,
        threshold: float,
        person_class_id: int,
        background_class_id: int | None,
        num_select: int | None,
        prefer_dml: bool,
    ) -> None:
        self.model_path = str(model_path)
        self.threshold = float(threshold)
        self.person_class_id = int(person_class_id)
        self.background_class_id = background_class_id
        self.num_select = num_select
        self.session = _session(model_path, prefer_dml)
        inputs = self.session.get_inputs()
        if len(inputs) != 1 or len(inputs[0].shape) != 4:
            raise ValueError("RF-DETR research adapter expects one NCHW image input")
        _, channels, height, width = inputs[0].shape
        if channels != 3 or not isinstance(height, int) or not isinstance(width, int):
            raise ValueError(f"RF-DETR research adapter requires fixed 3-channel spatial input, got {inputs[0].shape}")
        self.input_name = inputs[0].name
        self.input_h = int(height)
        self.input_w = int(width)

    @property
    def providers(self) -> list[str]:
        return list(self.session.get_providers())

    def contract(self) -> dict[str, Any]:
        return {
            "inputs": [{"name": item.name, "shape": list(item.shape)} for item in self.session.get_inputs()],
            "outputs": [{"name": item.name, "shape": list(item.shape)} for item in self.session.get_outputs()],
            "providers": self.providers,
            "actual_input": [self.input_h, self.input_w],
        }

    def _output_indices(self, arrays: list[np.ndarray]) -> tuple[int, int]:
        names = [item.name for item in self.session.get_outputs()]
        boxes_index = next((i for i, name in enumerate(names) if "dets" in name.lower()), None)
        logits_index = next((i for i, name in enumerate(names) if "labels" in name.lower()), None)
        if boxes_index is not None and logits_index is not None:
            return boxes_index, logits_index
        box_candidates = [i for i, value in enumerate(arrays) if value.ndim == 3 and value.shape[-1] == 4]
        logit_candidates = [i for i, value in enumerate(arrays) if value.ndim == 3 and value.shape[-1] != 4]
        if len(box_candidates) == 1 and len(logit_candidates) == 1:
            return box_candidates[0], logit_candidates[0]
        raise ValueError(f"cannot identify RF-DETR outputs names={names} shapes={[list(v.shape) for v in arrays]}")

    def detect(self, frame: np.ndarray) -> AdapterResult:
        wall_started = time.perf_counter()
        started = time.perf_counter()
        blob = _rf_preprocess(frame, self.input_h, self.input_w)
        preprocess_ms = (time.perf_counter() - started) * 1000.0

        started = time.perf_counter()
        raw = [np.asarray(value) for value in self.session.run(None, {self.input_name: blob})]
        inference_ms = (time.perf_counter() - started) * 1000.0

        started = time.perf_counter()
        boxes_index, logits_index = self._output_indices(raw)
        detections = decode_rfdetr_person(
            raw[boxes_index][0],
            raw[logits_index][0],
            width=frame.shape[1],
            height=frame.shape[0],
            threshold=self.threshold,
            person_class_id=self.person_class_id,
            background_class_id=self.background_class_id,
            num_select=self.num_select,
        )
        postprocess_ms = (time.perf_counter() - started) * 1000.0
        return AdapterResult(
            tuple(detections),
            preprocess_ms,
            inference_ms,
            postprocess_ms,
            (time.perf_counter() - wall_started) * 1000.0,
            1,
        )


class RtDetrv2OnnxAdapter:
    def __init__(
        self,
        model_path: str | Path,
        *,
        threshold: float,
        person_class_id: int,
        prefer_dml: bool,
    ) -> None:
        self.model_path = str(model_path)
        self.threshold = float(threshold)
        self.person_class_id = int(person_class_id)
        self.session = _session(model_path, prefer_dml)
        inputs = {item.name: item for item in self.session.get_inputs()}
        if "images" not in inputs or "orig_target_sizes" not in inputs:
            raise ValueError(f"RT-DETRv2 export must expose images + orig_target_sizes, got {sorted(inputs)}")
        image_shape = inputs["images"].shape
        if len(image_shape) != 4 or image_shape[1] != 3:
            raise ValueError(f"RT-DETRv2 images input must be NCHW RGB, got {image_shape}")
        if not isinstance(image_shape[2], int) or not isinstance(image_shape[3], int):
            raise ValueError(f"RT-DETRv2 research adapter requires fixed spatial input, got {image_shape}")
        self.input_h = int(image_shape[2])
        self.input_w = int(image_shape[3])

    @property
    def providers(self) -> list[str]:
        return list(self.session.get_providers())

    def contract(self) -> dict[str, Any]:
        return {
            "inputs": [{"name": item.name, "shape": list(item.shape)} for item in self.session.get_inputs()],
            "outputs": [{"name": item.name, "shape": list(item.shape)} for item in self.session.get_outputs()],
            "providers": self.providers,
            "actual_input": [self.input_h, self.input_w],
        }

    def detect(self, frame: np.ndarray) -> AdapterResult:
        wall_started = time.perf_counter()
        started = time.perf_counter()
        blob = _rtdetr_preprocess(frame, self.input_h, self.input_w)
        target_sizes = np.asarray([[frame.shape[0], frame.shape[1]]], dtype=np.int64)
        preprocess_ms = (time.perf_counter() - started) * 1000.0

        started = time.perf_counter()
        outputs = self.session.run(None, {"images": blob, "orig_target_sizes": target_sizes})
        inference_ms = (time.perf_counter() - started) * 1000.0

        started = time.perf_counter()
        names = [item.name for item in self.session.get_outputs()]
        values = {name: np.asarray(value) for name, value in zip(names, outputs)}
        if not {"labels", "boxes", "scores"} <= set(values):
            raise ValueError(f"RT-DETRv2 export outputs must include labels/boxes/scores, got {names}")
        labels = values["labels"][0]
        boxes = values["boxes"][0]
        scores = values["scores"][0]
        detections: list[Detection] = []
        for label, box, score in zip(labels, boxes, scores):
            if int(label) != self.person_class_id or float(score) <= self.threshold:
                continue
            clipped = _clip_box(np.asarray(box, dtype=np.float32), frame.shape[1], frame.shape[0])
            if clipped is not None:
                detections.append(Detection(clipped, float(score), 0, "person"))
        postprocess_ms = (time.perf_counter() - started) * 1000.0
        return AdapterResult(
            tuple(detections),
            preprocess_ms,
            inference_ms,
            postprocess_ms,
            (time.perf_counter() - wall_started) * 1000.0,
            1,
        )


def build_adapter(args: argparse.Namespace):
    if args.backend == "current-yolo":
        return CurrentYoloAdapter(
            args.model,
            input_size=args.input_size,
            threshold=args.threshold,
            nms_iou=args.nms_iou,
            prefer_dml=not args.cpu,
        )
    if args.backend == "rf-detr":
        background = None if args.rf_background_class_id == "none" else int(args.rf_background_class_id)
        return RfDetrOnnxAdapter(
            args.model,
            threshold=args.threshold,
            person_class_id=args.person_class_id,
            background_class_id=background,
            num_select=args.rf_num_select,
            prefer_dml=not args.cpu,
        )
    if args.backend == "rt-detrv2":
        return RtDetrv2OnnxAdapter(
            args.model,
            threshold=args.threshold,
            person_class_id=args.person_class_id,
            prefer_dml=not args.cpu,
        )
    raise ValueError(f"unsupported backend: {args.backend}")


def _matched_localization_iou(frame, predictions: list[PredictedObject], match_iou: float) -> list[float]:
    candidates: list[tuple[float, int, int]] = []
    truth = [obj for obj in frame.objects if obj.label == "person" and not obj.ignore]
    for gt_index, obj in enumerate(truth):
        for pred_index, prediction in enumerate(predictions):
            x1 = max(obj.bbox[0], prediction.bbox[0])
            y1 = max(obj.bbox[1], prediction.bbox[1])
            x2 = min(obj.bbox[2], prediction.bbox[2])
            y2 = min(obj.bbox[3], prediction.bbox[3])
            inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
            gt_area = (obj.bbox[2] - obj.bbox[0]) * (obj.bbox[3] - obj.bbox[1])
            pr_area = (prediction.bbox[2] - prediction.bbox[0]) * (prediction.bbox[3] - prediction.bbox[1])
            overlap = inter / max(gt_area + pr_area - inter, 1e-9)
            if overlap >= match_iou:
                candidates.append((overlap, gt_index, pred_index))
    candidates.sort(reverse=True)
    matched_gt: set[int] = set()
    matched_pred: set[int] = set()
    result = []
    for overlap, gt_index, pred_index in candidates:
        if gt_index in matched_gt or pred_index in matched_pred:
            continue
        matched_gt.add(gt_index)
        matched_pred.add(pred_index)
        result.append(overlap)
    return result


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    ground_truth = load_ground_truth(args.ground_truth)
    by_video: dict[str, list[Any]] = {}
    for frame in ground_truth:
        by_video.setdefault(frame.video, []).append(frame)

    adapter = build_adapter(args)
    predictions: dict[tuple[str, int], list[PredictedObject]] = {}
    timing = {"preprocess_ms": 0.0, "inference_ms": 0.0, "postprocess_ms": 0.0, "detector_wall_ms": 0.0}
    inference_calls = 0
    localization: list[float] = []
    benchmark_started = time.perf_counter()

    for video, annotations in sorted(by_video.items()):
        path = Path(args.video_root) / video
        if not path.is_file():
            raise FileNotFoundError(f"benchmark video missing: {path}")
        needed = {item.frame: item for item in annotations}
        last_needed = max(needed)
        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            raise RuntimeError(f"cannot open benchmark video: {path}")
        frame_index = 0
        try:
            while frame_index <= last_needed:
                ok, frame_bgr = capture.read()
                if not ok or frame_bgr is None:
                    raise RuntimeError(f"{path} ended before annotated frame {last_needed}")
                annotation = needed.get(frame_index)
                if annotation is not None:
                    result = adapter.detect(frame_bgr)
                    predicted = [
                        PredictedObject(item.bbox, item.label, item.score)
                        for item in result.detections
                    ]
                    predictions[(video, frame_index)] = predicted
                    localization.extend(_matched_localization_iou(annotation, predicted, args.match_iou))
                    timing["preprocess_ms"] += result.preprocess_ms
                    timing["inference_ms"] += result.inference_ms
                    timing["postprocess_ms"] += result.postprocess_ms
                    timing["detector_wall_ms"] += result.wall_ms
                    inference_calls += result.inference_calls
                frame_index += 1
        finally:
            capture.release()

    metrics = evaluate_frames(ground_truth, predictions, label="person", iou_threshold=args.match_iou)
    policy_runs = len(ground_truth)
    benchmark_wall_s = time.perf_counter() - benchmark_started
    return {
        "schema": "spectratrack-vnext-detector-backend-result-v1",
        "source_commit": args.source_commit,
        "corpus_revision": args.corpus_revision,
        "ground_truth_sha256": ground_truth_sha256(args.ground_truth),
        "backend": args.backend,
        "backend_spec": BACKEND_SPECS[args.backend].__dict__,
        "model": str(args.model),
        "model_sha256": sha256_file(args.model),
        "providers": adapter.providers,
        "provider_priority": adapter.providers[0] if adapter.providers else None,
        "provider_note": "Provider list/priority is recorded; per-node DirectML->CPU fallback is not inferred.",
        "contract": adapter.contract(),
        "settings": {
            "threshold": args.threshold,
            "match_iou": args.match_iou,
            "input_size": args.input_size if args.backend == "current-yolo" else None,
            "nms_iou": args.nms_iou if args.backend == "current-yolo" else None,
            "person_class_id": args.person_class_id if args.backend != "current-yolo" else 0,
            "rf_background_class_id": args.rf_background_class_id if args.backend == "rf-detr" else None,
            "rf_num_select": args.rf_num_select if args.backend == "rf-detr" else None,
        },
        "performance": {
            **timing,
            "benchmark_wall_s": benchmark_wall_s,
            "policy_runs": policy_runs,
            "inference_calls": inference_calls,
            "peak_vram_mb": args.peak_vram_mb,
            "vram_source": args.vram_source if args.peak_vram_mb is not None else None,
        },
        "metrics": {
            **metrics,
            "bbox_localization_iou": sum(localization) / len(localization) if localization else None,
        },
    }


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--backend", choices=tuple(BACKEND_SPECS), required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--threshold", type=float, default=0.35)
    parser.add_argument("--input-size", type=int, default=640)
    parser.add_argument("--nms-iou", type=float, default=0.45)
    parser.add_argument("--person-class-id", type=int, default=0)
    parser.add_argument("--rf-background-class-id", default="-1", help="integer class slot or 'none'")
    parser.add_argument("--rf-num-select", type=int)
    parser.add_argument("--cpu", action="store_true")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SpectraTrack vNext A1 detector-backend research lab")
    sub = parser.add_subparsers(dest="command", required=True)

    specs = sub.add_parser("specs", help="Print frozen research adapter assumptions")
    specs.add_argument("--output")

    probe = sub.add_parser("probe", help="Open ONNX and record its runtime contract without quality claims")
    _add_common(probe)
    probe.add_argument("--output", required=True)

    benchmark = sub.add_parser("benchmark", help="Evaluate one backend on the frozen A5 annotated frames")
    _add_common(benchmark)
    benchmark.add_argument("--ground-truth", required=True)
    benchmark.add_argument("--video-root", required=True)
    benchmark.add_argument("--source-commit", required=True)
    benchmark.add_argument("--corpus-revision", required=True)
    benchmark.add_argument("--match-iou", type=float, default=0.5)
    benchmark.add_argument("--peak-vram-mb", type=float)
    benchmark.add_argument("--vram-source")
    benchmark.add_argument("--output", required=True)
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    if args.command == "specs":
        payload = {name: spec.__dict__ for name, spec in BACKEND_SPECS.items()}
        text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
        if args.output:
            Path(args.output).write_text(text, encoding="utf-8")
        else:
            print(text, end="")
        return 0

    model_path = Path(args.model)
    if not model_path.is_file():
        raise SystemExit(f"model not found: {model_path}")

    if args.command == "probe":
        adapter = build_adapter(args)
        payload = {
            "schema": "spectratrack-vnext-detector-probe-v1",
            "backend": args.backend,
            "model": str(model_path),
            "model_sha256": sha256_file(model_path),
            "spec": BACKEND_SPECS[args.backend].__dict__,
            "contract": adapter.contract(),
            "providers": adapter.providers,
            "quality_evidence": False,
            "note": "Session creation/contract probe only. DirectML inference remains unverified until an actual run.",
        }
    else:
        if args.peak_vram_mb is not None and not args.vram_source:
            raise SystemExit("--vram-source is required with --peak-vram-mb")
        payload = run_benchmark(args)

    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload.get("performance", payload.get("contract", {})), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
