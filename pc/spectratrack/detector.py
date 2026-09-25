from __future__ import annotations

from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import onnxruntime as ort

from .types import Detection


COCO80 = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard",
    "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
    "teddy bear", "hair drier", "toothbrush",
]


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    xx1 = max(float(a[0]), float(b[0]))
    yy1 = max(float(a[1]), float(b[1]))
    xx2 = min(float(a[2]), float(b[2]))
    yy2 = min(float(a[3]), float(b[3]))
    inter = max(0.0, xx2 - xx1) * max(0.0, yy2 - yy1)
    area_a = max(0.0, float(a[2] - a[0])) * max(0.0, float(a[3] - a[1]))
    area_b = max(0.0, float(b[2] - b[0])) * max(0.0, float(b[3] - b[1]))
    return inter / max(area_a + area_b - inter, 1e-6)


def _classwise_nms(boxes: np.ndarray, scores: np.ndarray, classes: np.ndarray, threshold: float) -> list[int]:
    keep: list[int] = []
    for cls in np.unique(classes):
        idx = np.where(classes == cls)[0]
        idx = idx[np.argsort(scores[idx])[::-1]]
        while len(idx):
            best = int(idx[0])
            keep.append(best)
            if len(idx) == 1:
                break
            remaining = []
            for candidate in idx[1:]:
                if _iou(boxes[best], boxes[int(candidate)]) < threshold:
                    remaining.append(int(candidate))
            idx = np.asarray(remaining, dtype=np.int64)
    return keep


def _looks_like_end2end(pred: np.ndarray, label_count: int) -> bool:
    if pred.ndim != 2:
        return False
    if pred.shape[1] == 7:
        # Common [batch,x1,y1,x2,y2,score,class] layout.
        batch = pred[:, 0]
        if len(batch) and np.all(np.isfinite(batch)) and np.max(np.abs(batch - np.rint(batch))) < 1e-3:
            pred = pred[:, 1:]
        else:
            return False
    if pred.shape[1] != 6 or label_count <= 2 or len(pred) == 0:
        return False
    scores = pred[:, 4]
    classes = pred[:, 5]
    return (
        np.all(np.isfinite(scores))
        and np.all(np.isfinite(classes))
        and float(np.min(scores)) >= -1e-4
        and float(np.max(scores)) <= 1.0001
        and float(np.max(np.abs(classes - np.rint(classes)))) < 1e-3
    )


def decode_end2end_predictions(
    pred: np.ndarray,
    frame_w: int,
    frame_h: int,
    input_w: int,
    input_h: int,
    scale: float,
    pad_x: float,
    pad_y: float,
    labels: list[str],
    conf_threshold: float,
    iou_threshold: float,
) -> list[Detection]:
    pred = np.asarray(pred, dtype=np.float32)
    if pred.shape[1] == 7:
        pred = pred[:, 1:]
    mask = pred[:, 4] >= conf_threshold
    pred = pred[mask]
    if len(pred) == 0:
        return []
    boxes = pred[:, :4].copy()
    scores = pred[:, 4].copy()
    class_ids = np.rint(pred[:, 5]).astype(np.int32)

    if float(np.nanmax(np.abs(boxes))) <= 2.5:
        boxes[:, [0, 2]] *= input_w
        boxes[:, [1, 3]] *= input_h

    keep = _classwise_nms(boxes, scores, class_ids, iou_threshold)
    out: list[Detection] = []
    for i in keep:
        x1 = (float(boxes[i, 0]) - pad_x) / scale
        y1 = (float(boxes[i, 1]) - pad_y) / scale
        x2 = (float(boxes[i, 2]) - pad_x) / scale
        y2 = (float(boxes[i, 3]) - pad_y) / scale
        x1 = max(0.0, min(x1, frame_w - 1.0))
        y1 = max(0.0, min(y1, frame_h - 1.0))
        x2 = max(0.0, min(x2, frame_w - 1.0))
        y2 = max(0.0, min(y2, frame_h - 1.0))
        if x2 <= x1 or y2 <= y1:
            continue
        cid = int(class_ids[i])
        label = labels[cid] if 0 <= cid < len(labels) else f"class_{cid}"
        out.append(Detection((x1, y1, x2, y2), float(scores[i]), cid, label))
    return out


class YoloOnnxDetector:
    """YOLOv8/YOLO11-style ONNX detector.

    Expected output is either [1, 84, N] or [1, N, 84] for COCO-like models,
    where the first four values are xywh and the remaining values are class scores.
    """

    def __init__(
        self,
        model_path: str | Path,
        input_size: int = 640,
        conf_threshold: float = 0.35,
        iou_threshold: float = 0.45,
        labels: Iterable[str] | None = None,
        prefer_gpu: bool = True,
    ) -> None:
        self.model_path = str(model_path)
        self.input_size = int(input_size)
        self.conf_threshold = float(conf_threshold)
        self.iou_threshold = float(iou_threshold)
        self.labels = list(labels or COCO80)

        available = ort.get_available_providers()
        providers: list[str] = []
        if prefer_gpu and "DmlExecutionProvider" in available:
            providers.append("DmlExecutionProvider")
        if "CPUExecutionProvider" in available:
            providers.append("CPUExecutionProvider")
        if not providers:
            providers = available

        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        # DirectML requires sequential execution and disabled memory patterns.
        if "DmlExecutionProvider" in providers:
            options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            options.enable_mem_pattern = False

        self.session = ort.InferenceSession(self.model_path, sess_options=options, providers=providers)
        self.providers = self.session.get_providers()
        self.input = self.session.get_inputs()[0]
        self.input_name = self.input.name

        shape = self.input.shape
        if len(shape) == 4 and isinstance(shape[2], int) and isinstance(shape[3], int):
            self.input_h = int(shape[2])
            self.input_w = int(shape[3])
        else:
            self.input_h = self.input_size
            self.input_w = self.input_size

    def _letterbox(self, frame: np.ndarray) -> tuple[np.ndarray, float, float, float]:
        h, w = frame.shape[:2]
        scale = min(self.input_w / w, self.input_h / h)
        nw, nh = int(round(w * scale)), int(round(h * scale))
        resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
        canvas = np.full((self.input_h, self.input_w, 3), 114, dtype=np.uint8)
        pad_x = (self.input_w - nw) / 2.0
        pad_y = (self.input_h - nh) / 2.0
        left, top = int(round(pad_x - 0.1)), int(round(pad_y - 0.1))
        canvas[top:top + nh, left:left + nw] = resized
        return canvas, scale, float(left), float(top)

    def detect(self, frame_bgr: np.ndarray) -> list[Detection]:
        image, scale, pad_x, pad_y = self._letterbox(frame_bgr)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        blob = rgb.astype(np.float32) / 255.0
        blob = np.transpose(blob, (2, 0, 1))[None, ...]

        outputs = self.session.run(None, {self.input_name: blob})
        pred = np.asarray(outputs[0])
        pred = np.squeeze(pred)
        if pred.ndim != 2:
            raise RuntimeError(f"Unsupported detector output shape: {outputs[0].shape}")

        # YOLO exports commonly return [features, boxes]; normalize to [boxes, features].
        if pred.shape[0] < pred.shape[1] and pred.shape[0] <= 512:
            pred = pred.T
        if pred.shape[1] < 6:
            raise RuntimeError(f"Detector output has too few features: {pred.shape}")

        h, w = frame_bgr.shape[:2]
        if _looks_like_end2end(pred, len(self.labels)):
            return decode_end2end_predictions(
                pred, w, h, self.input_w, self.input_h,
                scale, pad_x, pad_y, self.labels,
                self.conf_threshold, self.iou_threshold,
            )

        boxes_xywh = pred[:, :4]
        class_scores = pred[:, 4:]
        class_ids = np.argmax(class_scores, axis=1)
        scores = class_scores[np.arange(class_scores.shape[0]), class_ids]
        mask = scores >= self.conf_threshold
        if not np.any(mask):
            return []

        boxes_xywh = boxes_xywh[mask]
        class_ids = class_ids[mask].astype(np.int32)
        scores = scores[mask].astype(np.float32)

        # Some exports can emit normalized xywh, most emit pixels. Detect normalized case.
        if np.nanmax(boxes_xywh) <= 2.5:
            boxes_xywh[:, [0, 2]] *= self.input_w
            boxes_xywh[:, [1, 3]] *= self.input_h

        boxes = np.empty_like(boxes_xywh, dtype=np.float32)
        boxes[:, 0] = boxes_xywh[:, 0] - boxes_xywh[:, 2] / 2.0
        boxes[:, 1] = boxes_xywh[:, 1] - boxes_xywh[:, 3] / 2.0
        boxes[:, 2] = boxes_xywh[:, 0] + boxes_xywh[:, 2] / 2.0
        boxes[:, 3] = boxes_xywh[:, 1] + boxes_xywh[:, 3] / 2.0

        keep = _classwise_nms(boxes, scores, class_ids, self.iou_threshold)
        detections: list[Detection] = []
        for i in keep:
            x1 = (float(boxes[i, 0]) - pad_x) / scale
            y1 = (float(boxes[i, 1]) - pad_y) / scale
            x2 = (float(boxes[i, 2]) - pad_x) / scale
            y2 = (float(boxes[i, 3]) - pad_y) / scale
            x1 = max(0.0, min(x1, w - 1.0))
            y1 = max(0.0, min(y1, h - 1.0))
            x2 = max(0.0, min(x2, w - 1.0))
            y2 = max(0.0, min(y2, h - 1.0))
            cid = int(class_ids[i])
            label = self.labels[cid] if 0 <= cid < len(self.labels) else f"class_{cid}"
            detections.append(Detection((x1, y1, x2, y2), float(scores[i]), cid, label))
        return detections
