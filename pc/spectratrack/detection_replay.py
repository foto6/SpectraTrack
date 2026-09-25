from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable

from .types import BBox, Detection

DETECTION_REPLAY_SCHEMA = "spectratrack-detection-replay-v1"


@dataclass(frozen=True, slots=True)
class ReplayDetection:
    bbox: BBox
    score: float
    class_id: int
    label: str
    appearance: tuple[float, ...] | None = None

    def to_detection(self) -> Detection:
        return Detection(
            bbox=tuple(self.bbox),
            score=float(self.score),
            class_id=int(self.class_id),
            label=str(self.label),
            appearance=None if self.appearance is None else tuple(self.appearance),
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "bbox": [float(value) for value in self.bbox],
            "score": float(self.score),
            "class_id": int(self.class_id),
            "label": self.label,
            "appearance": None if self.appearance is None else [float(value) for value in self.appearance],
        }


@dataclass(frozen=True, slots=True)
class ReplayMetadata:
    source_commit: str
    video: str
    video_sha256: str | None
    detector: str
    model_sha256: str | None
    provider: str
    config: dict[str, Any]
    width: int
    height: int

    def to_json(self) -> dict[str, Any]:
        return {
            "type": "metadata",
            "schema": DETECTION_REPLAY_SCHEMA,
            "source_commit": self.source_commit,
            "video": self.video,
            "video_sha256": self.video_sha256,
            "detector": self.detector,
            "model_sha256": self.model_sha256,
            "provider": self.provider,
            "config": self.config,
            "width": self.width,
            "height": self.height,
        }


@dataclass(frozen=True, slots=True)
class ReplayFrame:
    video: str
    frame: int
    timestamp_s: float | None
    width: int
    height: int
    detections: tuple[ReplayDetection, ...]
    detector_ran: bool = True
    camera_motion: tuple[float, float] = (0.0, 0.0)
    camera_transform: tuple[float, float, float, float, float, float] | None = None

    def fresh_detections(self) -> list[Detection]:
        return [item.to_detection() for item in self.detections]

    def to_json(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "type": "frame",
            "video": self.video,
            "frame": self.frame,
            "timestamp_s": self.timestamp_s,
            "width": self.width,
            "height": self.height,
            "detections": [item.to_json() for item in self.detections],
        }
        if not self.detector_ran:
            payload["detector_ran"] = False
        if self.camera_motion != (0.0, 0.0):
            payload["camera_motion"] = [float(self.camera_motion[0]), float(self.camera_motion[1])]
        if self.camera_transform is not None:
            payload["camera_transform"] = [float(value) for value in self.camera_transform]
        return payload


@dataclass(frozen=True, slots=True)
class DetectionReplay:
    metadata: ReplayMetadata
    frames: tuple[ReplayFrame, ...]
    source_sha256: str | None = None

    def canonical_sha256(self) -> str:
        return hashlib.sha256(canonical_replay_bytes(self)).hexdigest()


def _finite_number(value: Any, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{where} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{where} must be finite")
    return number


def _positive_int(value: Any, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{where} must be a positive integer")
    return int(value)


def _optional_sha(value: Any, where: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise ValueError(f"{where} must be a non-empty string or null")
    return value


def _bbox(value: Any, where: str) -> BBox:
    if not isinstance(value, list) or len(value) != 4:
        raise ValueError(f"{where} must be [x1, y1, x2, y2]")
    x1, y1, x2, y2 = (_finite_number(item, where) for item in value)
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"{where} must have positive width and height")
    return x1, y1, x2, y2


def _vector(value: Any, length: int, where: str) -> tuple[float, ...]:
    if not isinstance(value, list) or len(value) != length:
        raise ValueError(f"{where} must contain exactly {length} numeric values")
    return tuple(_finite_number(item, where) for item in value)


def _parse_detection(value: Any, where: str) -> ReplayDetection:
    if not isinstance(value, dict):
        raise ValueError(f"{where} must be an object")
    score = _finite_number(value.get("score"), f"{where}.score")
    if not 0.0 <= score <= 1.0:
        raise ValueError(f"{where}.score must be in [0, 1]")
    class_id = value.get("class_id")
    if isinstance(class_id, bool) or not isinstance(class_id, int) or class_id < 0:
        raise ValueError(f"{where}.class_id must be a non-negative integer")
    label = value.get("label")
    if not isinstance(label, str) or not label:
        raise ValueError(f"{where}.label must be a non-empty string")
    appearance_raw = value.get("appearance")
    appearance = None
    if appearance_raw is not None:
        if not isinstance(appearance_raw, list) or not appearance_raw:
            raise ValueError(f"{where}.appearance must be null or a non-empty numeric list")
        appearance = tuple(_finite_number(item, f"{where}.appearance") for item in appearance_raw)
    return ReplayDetection(
        bbox=_bbox(value.get("bbox"), f"{where}.bbox"),
        score=score,
        class_id=int(class_id),
        label=label,
        appearance=appearance,
    )


def _parse_metadata(value: Any) -> ReplayMetadata:
    if not isinstance(value, dict) or value.get("type") != "metadata":
        raise ValueError("first replay record must be metadata")
    if value.get("schema") != DETECTION_REPLAY_SCHEMA:
        raise ValueError(f"unsupported detection replay schema: {value.get('schema')!r}")
    source_commit = value.get("source_commit")
    video = value.get("video")
    detector = value.get("detector")
    provider = value.get("provider")
    config = value.get("config")
    for field_name, field_value in (
        ("source_commit", source_commit),
        ("video", video),
        ("detector", detector),
        ("provider", provider),
    ):
        if not isinstance(field_value, str) or not field_value:
            raise ValueError(f"metadata.{field_name} must be a non-empty string")
    if not isinstance(config, dict):
        raise ValueError("metadata.config must be an object")
    return ReplayMetadata(
        source_commit=source_commit,
        video=video,
        video_sha256=_optional_sha(value.get("video_sha256"), "metadata.video_sha256"),
        detector=detector,
        model_sha256=_optional_sha(value.get("model_sha256"), "metadata.model_sha256"),
        provider=provider,
        config=config,
        width=_positive_int(value.get("width"), "metadata.width"),
        height=_positive_int(value.get("height"), "metadata.height"),
    )


def _parse_frame(value: Any, metadata: ReplayMetadata, line_number: int) -> ReplayFrame:
    where = f"line {line_number}"
    if not isinstance(value, dict) or value.get("type") != "frame":
        raise ValueError(f"{where}: expected frame record")
    video = value.get("video")
    if video != metadata.video:
        raise ValueError(f"{where}: frame video must match metadata video")
    frame = value.get("frame")
    if isinstance(frame, bool) or not isinstance(frame, int) or frame < 0:
        raise ValueError(f"{where}: frame must be a non-negative integer")
    timestamp_raw = value.get("timestamp_s")
    timestamp_s = None if timestamp_raw is None else _finite_number(timestamp_raw, f"{where}.timestamp_s")
    if timestamp_s is not None and timestamp_s < 0.0:
        raise ValueError(f"{where}.timestamp_s must be >= 0")
    width = _positive_int(value.get("width"), f"{where}.width")
    height = _positive_int(value.get("height"), f"{where}.height")
    detections_raw = value.get("detections")
    if not isinstance(detections_raw, list):
        raise ValueError(f"{where}.detections must be a list")
    detections = tuple(
        _parse_detection(item, f"{where}.detections[{index}]")
        for index, item in enumerate(detections_raw)
    )
    detector_ran = value.get("detector_ran", True)
    if not isinstance(detector_ran, bool):
        raise ValueError(f"{where}.detector_ran must be boolean when present")
    camera_motion_raw = value.get("camera_motion")
    camera_motion = (0.0, 0.0)
    if camera_motion_raw is not None:
        parsed_motion = _vector(camera_motion_raw, 2, f"{where}.camera_motion")
        camera_motion = (parsed_motion[0], parsed_motion[1])
    transform_raw = value.get("camera_transform")
    camera_transform = None
    if transform_raw is not None:
        parsed_transform = _vector(transform_raw, 6, f"{where}.camera_transform")
        camera_transform = (
            parsed_transform[0],
            parsed_transform[1],
            parsed_transform[2],
            parsed_transform[3],
            parsed_transform[4],
            parsed_transform[5],
        )
    return ReplayFrame(
        video=video,
        frame=frame,
        timestamp_s=timestamp_s,
        width=width,
        height=height,
        detections=detections,
        detector_ran=detector_ran,
        camera_motion=camera_motion,
        camera_transform=camera_transform,
    )


def parse_detection_replay(text: str, *, source_sha256: str | None = None) -> DetectionReplay:
    records: list[tuple[int, Any]] = []
    for line_number, raw in enumerate(text.splitlines(), start=1):
        raw = raw.strip()
        if not raw or raw.startswith("#"):
            continue
        try:
            records.append((line_number, json.loads(raw)))
        except json.JSONDecodeError as exc:
            raise ValueError(f"line {line_number}: invalid JSON") from exc
    if not records:
        raise ValueError("detection replay is empty")
    metadata = _parse_metadata(records[0][1])
    frames = tuple(_parse_frame(value, metadata, line_number) for line_number, value in records[1:])
    if not frames:
        raise ValueError("detection replay contains no frame records")
    previous_frame = -1
    previous_timestamp: float | None = None
    for item in frames:
        if item.frame <= previous_frame:
            raise ValueError("frame indices must be strictly increasing")
        if previous_timestamp is not None and item.timestamp_s is not None and item.timestamp_s < previous_timestamp:
            raise ValueError("timestamps must be non-decreasing")
        previous_frame = item.frame
        if item.timestamp_s is not None:
            previous_timestamp = item.timestamp_s
    return DetectionReplay(metadata=metadata, frames=frames, source_sha256=source_sha256)


def load_detection_replay(path: str | Path) -> DetectionReplay:
    source = Path(path)
    raw = source.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    return parse_detection_replay(raw.decode("utf-8-sig"), source_sha256=digest)


def canonical_replay_lines(replay: DetectionReplay) -> Iterable[str]:
    yield json.dumps(replay.metadata.to_json(), sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    for frame in replay.frames:
        yield json.dumps(frame.to_json(), sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def canonical_replay_bytes(replay: DetectionReplay) -> bytes:
    return ("\n".join(canonical_replay_lines(replay)) + "\n").encode("utf-8")


def write_detection_replay(replay: DetectionReplay, path: str | Path) -> str:
    payload = canonical_replay_bytes(replay)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()
