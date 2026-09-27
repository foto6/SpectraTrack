from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator

import numpy as np

from .enhance import adaptive_analysis_frame
from .types import Detection

TileRegion = tuple[int, int, int, int]
AdaptiveTile = tuple[
    TileRegion,
    np.ndarray,
    np.ndarray | None,
    dict[str, float],
    tuple[str, ...],
]
DetectionFn = Callable[[np.ndarray, float], Iterable[Detection]]


def _validated_region(region: TileRegion, frame_w: int, frame_h: int) -> TileRegion:
    x1, y1, x2, y2 = region
    if x1 < 0 or y1 < 0 or x2 > frame_w or y2 > frame_h:
        raise ValueError("tile region must stay inside the frame")
    if x2 <= x1 or y2 <= y1:
        raise ValueError("tile region must have positive area")
    return x1, y1, x2, y2


def iter_adaptive_regions(
    frame: np.ndarray,
    regions: Iterable[TileRegion],
) -> Iterator[AdaptiveTile]:
    """Yield detector-owned raw regions plus optional enhanced analysis copies.

    Region generation intentionally remains detector-owned so enhancement does
    not duplicate tile geometry/overlap policy. The raw view is always kept;
    the enhanced copy is None when the quality router selects no operation.
    """
    if frame is None or frame.size == 0 or frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError("frame must be a non-empty BGR image")

    height, width = frame.shape[:2]
    for region in regions:
        x1, y1, x2, y2 = _validated_region(region, width, height)
        raw_tile = frame[y1:y2, x1:x2]
        processed, quality, operations = adaptive_analysis_frame(raw_tile)
        enhanced_tile = processed if operations else None
        yield region, raw_tile, enhanced_tile, quality, operations


def translate_detection(detection: Detection, offset_x: int, offset_y: int) -> Detection:
    """Translate a tile-local detection into full-frame coordinates."""
    x1, y1, x2, y2 = detection.bbox
    return Detection(
        (
            x1 + offset_x,
            y1 + offset_y,
            x2 + offset_x,
            y2 + offset_y,
        ),
        detection.score,
        detection.class_id,
        detection.label,
        detection.appearance,
    )


def _bbox_iou(a: Detection, b: Detection) -> float:
    ax1, ay1, ax2, ay2 = a.bbox
    bx1, by1, bx2, by2 = b.bbox
    xx1 = max(ax1, bx1)
    yy1 = max(ay1, by1)
    xx2 = min(ax2, bx2)
    yy2 = min(ay2, by2)
    inter = max(0.0, xx2 - xx1) * max(0.0, yy2 - yy1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    return inter / max(area_a + area_b - inter, 1e-6)


def detections_corroborate(
    raw_candidate: Detection,
    enhanced_candidate: Detection,
    min_iou: float = 0.10,
) -> bool:
    """Check whether raw and enhanced detector evidence support the same object."""
    if raw_candidate.class_id != enhanced_candidate.class_id:
        return False
    if _bbox_iou(raw_candidate, enhanced_candidate) >= min_iou:
        return True

    raw_x, raw_y = raw_candidate.center
    enhanced_x, enhanced_y = enhanced_candidate.center
    raw_w = max(1.0, raw_candidate.bbox[2] - raw_candidate.bbox[0])
    raw_h = max(1.0, raw_candidate.bbox[3] - raw_candidate.bbox[1])
    enhanced_w = max(1.0, enhanced_candidate.bbox[2] - enhanced_candidate.bbox[0])
    enhanced_h = max(1.0, enhanced_candidate.bbox[3] - enhanced_candidate.bbox[1])
    tolerance = 0.55 * max(raw_w, raw_h, enhanced_w, enhanced_h)
    return (raw_x - enhanced_x) ** 2 + (raw_y - enhanced_y) ** 2 <= tolerance**2


def corroborate_enhanced_detections(
    raw_candidates: Iterable[Detection],
    enhanced_candidates: Iterable[Detection],
    min_iou: float = 0.10,
) -> list[Detection]:
    """Keep only enhanced detections that have spatial/class support in raw evidence."""
    raw = list(raw_candidates)
    return [
        candidate
        for candidate in enhanced_candidates
        if any(detections_corroborate(raw_candidate, candidate, min_iou) for raw_candidate in raw)
    ]


def detect_people_with_adaptive_regions(
    frame: np.ndarray,
    regions: Iterable[TileRegion],
    detect_at_confidence: DetectionFn,
    person_conf: float = 0.18,
    probe_conf: float = 0.08,
    corroboration_iou: float = 0.10,
) -> list[Detection]:
    """Run enhancement-aware tiny-person processing over detector-owned tiles.

    The callback receives (tile_bgr, confidence_threshold). Raw tiles are probed
    below the acceptance threshold. An enhanced candidate can be returned only
    when a raw probe supports the same person. This helper intentionally does
    not own tile generation or final NMS/merging; those remain detector policy.
    """
    if not 0.0 < probe_conf <= person_conf <= 1.0:
        raise ValueError("require 0 < probe_conf <= person_conf <= 1")
    if not 0.0 <= corroboration_iou <= 1.0:
        raise ValueError("corroboration_iou must be in [0, 1]")

    accepted: list[Detection] = []
    for region, raw_tile, enhanced_tile, _quality, _operations in iter_adaptive_regions(
        frame,
        regions,
    ):
        x1, y1, _x2, _y2 = region
        raw_probes = [
            detection
            for detection in detect_at_confidence(raw_tile, probe_conf)
            if detection.label.lower() == "person"
        ]

        for detection in raw_probes:
            if detection.score >= person_conf:
                accepted.append(translate_detection(detection, x1, y1))

        if enhanced_tile is None:
            continue

        enhanced = [
            detection
            for detection in detect_at_confidence(enhanced_tile, person_conf)
            if detection.label.lower() == "person"
        ]
        corroborated = corroborate_enhanced_detections(
            raw_probes,
            enhanced,
            min_iou=corroboration_iou,
        )
        accepted.extend(translate_detection(detection, x1, y1) for detection in corroborated)

    return accepted
