from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import statistics
import time
from typing import Sequence

import numpy as np

from .detector import _tile_regions


RESOLUTIONS: dict[str, tuple[int, int]] = {
    "720p": (1280, 720),
    "1080p": (1920, 1080),
    "1440p": (2560, 1440),
    "4k": (3840, 2160),
}
POLICIES = ("CURRENT", "COARSE_TO_FINE", "TRACK_GUIDED", "BUDGETED_ADAPTIVE")


@dataclass(frozen=True)
class CurrentComputeModel:
    width: int
    height: int
    tile_size: int
    overlap: float
    tile_regions: tuple[tuple[int, int, int, int], ...]
    full_frame_calls: int
    raw_tile_calls: int
    enhanced_tile_calls: int

    @property
    def total_onnx_calls(self) -> int:
        return self.full_frame_calls + self.raw_tile_calls + self.enhanced_tile_calls

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["tile_count"] = len(self.tile_regions)
        payload["total_onnx_calls"] = self.total_onnx_calls
        return payload


def current_compute_model(
    width: int,
    height: int,
    *,
    tile_size: int = 640,
    overlap: float = 0.20,
    enhanced_tile_calls: int = 0,
) -> CurrentComputeModel:
    """Model the current production people-recall policy using production tile geometry."""
    regions = tuple(_tile_regions(width, height, tile_size, overlap))
    if not 0 <= enhanced_tile_calls <= len(regions):
        raise ValueError("enhanced_tile_calls must be between 0 and tile_count")
    return CurrentComputeModel(
        width=width,
        height=height,
        tile_size=tile_size,
        overlap=overlap,
        tile_regions=regions,
        full_frame_calls=1,
        raw_tile_calls=len(regions),
        enhanced_tile_calls=int(enhanced_tile_calls),
    )


def current_compute_bounds(
    width: int,
    height: int,
    *,
    tile_size: int = 640,
    overlap: float = 0.20,
) -> dict[str, CurrentComputeModel]:
    """Return adaptive minimum/maximum call counts for the current policy."""
    minimum = current_compute_model(
        width,
        height,
        tile_size=tile_size,
        overlap=overlap,
        enhanced_tile_calls=0,
    )
    maximum = current_compute_model(
        width,
        height,
        tile_size=tile_size,
        overlap=overlap,
        enhanced_tile_calls=len(minimum.tile_regions),
    )
    return {"minimum": minimum, "maximum": maximum}


def runtime_rates(
    *,
    frames: int,
    elapsed_s: float,
    source_fps: float,
    onnx_calls: int,
) -> dict[str, float]:
    if frames <= 0:
        raise ValueError("frames must be > 0")
    if elapsed_s <= 0.0:
        raise ValueError("elapsed_s must be > 0")
    if source_fps <= 0.0:
        raise ValueError("source_fps must be > 0")
    if onnx_calls < 0:
        raise ValueError("onnx_calls cannot be negative")

    processing_fps = frames / elapsed_s
    source_seconds = frames / source_fps
    return {
        "processing_fps": processing_fps,
        "processing_seconds_per_source_second": elapsed_s / source_seconds,
        "onnx_calls_per_frame": onnx_calls / frames,
        "onnx_calls_per_source_second": onnx_calls / source_seconds,
    }


@dataclass(frozen=True)
class FrameSignals:
    track_rois: int = 0
    suspect_rois: int = 0
    enhancement_eligible_rois: int = 0
    scene_change: bool = False
    camera_motion_trigger: bool = False

    def __post_init__(self) -> None:
        for name in ("track_rois", "suspect_rois", "enhancement_eligible_rois"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} cannot be negative")


@dataclass(frozen=True)
class SchedulerConfig:
    detector_every: int = 1
    global_period: int = 15
    max_calls_per_frame: int = 4
    max_enhanced_calls_per_frame: int = 1

    def __post_init__(self) -> None:
        if self.detector_every <= 0:
            raise ValueError("detector_every must be > 0")
        if self.global_period <= 0:
            raise ValueError("global_period must be > 0")
        if self.max_calls_per_frame <= 0:
            raise ValueError("max_calls_per_frame must be > 0")
        if self.max_enhanced_calls_per_frame < 0:
            raise ValueError("max_enhanced_calls_per_frame cannot be negative")


@dataclass(frozen=True)
class ScheduleDecision:
    policy: str
    frame_index: int
    full_frame_calls: int
    suspect_roi_calls: int
    track_roi_calls: int
    enhanced_roi_calls: int
    reused_temporal_state: bool
    global_rescan_reason: str | None

    @property
    def raw_roi_calls(self) -> int:
        return self.suspect_roi_calls + self.track_roi_calls

    @property
    def total_calls(self) -> int:
        return self.full_frame_calls + self.raw_roi_calls + self.enhanced_roi_calls

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["raw_roi_calls"] = self.raw_roi_calls
        payload["total_calls"] = self.total_calls
        return payload


def _detector_due(frame_index: int, config: SchedulerConfig) -> bool:
    if frame_index < 0:
        raise ValueError("frame_index cannot be negative")
    return frame_index % config.detector_every == 0


def _global_due(frame_index: int, signals: FrameSignals, config: SchedulerConfig) -> tuple[bool, str | None]:
    if signals.scene_change:
        return True, "scene_change"
    if signals.camera_motion_trigger:
        return True, "camera_motion"
    detector_index = frame_index // config.detector_every
    if detector_index % config.global_period == 0:
        return True, "periodic"
    return False, None


def _allocate_rois(
    *,
    remaining_calls: int,
    suspect_rois: int,
    track_rois: int,
) -> tuple[int, int]:
    """Share a small ROI budget without starving either uncertainty or known tracks."""
    suspect = 0
    track = 0
    if remaining_calls >= 2 and suspect_rois > 0 and track_rois > 0:
        suspect = 1
        track = 1
        remaining_calls -= 2
        suspect_rois -= 1
        track_rois -= 1

    extra_suspect = min(suspect_rois, remaining_calls)
    suspect += extra_suspect
    remaining_calls -= extra_suspect
    track += min(track_rois, remaining_calls)
    return suspect, track


def schedule_frame(
    policy: str,
    frame_index: int,
    signals: FrameSignals,
    *,
    tile_count: int,
    config: SchedulerConfig = SchedulerConfig(),
) -> ScheduleDecision:
    """Simulate inference-call scheduling only; no production detector behavior is changed."""
    policy = policy.upper()
    if policy not in POLICIES:
        raise ValueError(f"unknown policy: {policy}")
    if tile_count < 0:
        raise ValueError("tile_count cannot be negative")

    triggered = signals.scene_change or signals.camera_motion_trigger
    if not _detector_due(frame_index, config) and not (policy != "CURRENT" and triggered):
        return ScheduleDecision(policy, frame_index, 0, 0, 0, 0, True, None)

    if policy == "CURRENT":
        enhanced = min(tile_count, signals.enhancement_eligible_rois)
        return ScheduleDecision(
            policy,
            frame_index,
            1,
            tile_count,
            0,
            enhanced,
            False,
            "every_detector_frame",
        )

    if policy == "COARSE_TO_FINE":
        full = 1
        trigger_reason = (
            "scene_change"
            if signals.scene_change
            else "camera_motion"
            if signals.camera_motion_trigger
            else "every_detector_frame"
        )
        remaining = max(0, config.max_calls_per_frame - full)
        reserved_enhanced = 0
        if signals.suspect_rois > 0 and signals.enhancement_eligible_rois > 0:
            reserved_enhanced = min(config.max_enhanced_calls_per_frame, remaining)
        raw_budget = max(0, remaining - reserved_enhanced)
        raw_suspects = min(signals.suspect_rois, raw_budget)
        remaining -= raw_suspects
        enhanced = min(
            raw_suspects,
            signals.enhancement_eligible_rois,
            reserved_enhanced,
            remaining,
        )
        return ScheduleDecision(policy, frame_index, full, raw_suspects, 0, enhanced, False, trigger_reason)

    global_scan, reason = _global_due(frame_index, signals, config)
    full = 1 if global_scan else 0
    remaining = max(0, config.max_calls_per_frame - full)

    reserved_enhanced = 0
    if policy == "BUDGETED_ADAPTIVE" and signals.suspect_rois > 0 and signals.enhancement_eligible_rois > 0:
        reserved_enhanced = min(config.max_enhanced_calls_per_frame, remaining)

    raw_budget = max(0, remaining - reserved_enhanced)
    raw_suspects, raw_tracks = _allocate_rois(
        remaining_calls=raw_budget,
        suspect_rois=signals.suspect_rois,
        track_rois=signals.track_rois,
    )
    raw = raw_suspects + raw_tracks
    remaining -= raw

    if policy == "TRACK_GUIDED":
        enhanced = 0
    else:
        enhanced = min(
            raw_suspects,
            signals.enhancement_eligible_rois,
            reserved_enhanced,
            remaining,
        )

    return ScheduleDecision(
        policy,
        frame_index,
        full,
        raw_suspects,
        raw_tracks,
        enhanced,
        False,
        reason,
    )


def global_discovery_bound_frames(policy: str, config: SchedulerConfig) -> int:
    """Worst periodic-global-rescan bound when no scene/motion trigger fires."""
    policy = policy.upper()
    if policy not in POLICIES:
        raise ValueError(f"unknown policy: {policy}")
    if policy in {"CURRENT", "COARSE_TO_FINE"}:
        return config.detector_every
    return config.detector_every * config.global_period


def simulate_scheduler(
    policy: str,
    *,
    width: int,
    height: int,
    frames: int,
    source_fps: float,
    signals: FrameSignals,
    tile_size: int = 640,
    overlap: float = 0.20,
    config: SchedulerConfig = SchedulerConfig(),
) -> dict[str, object]:
    if frames <= 0:
        raise ValueError("frames must be > 0")
    if source_fps <= 0:
        raise ValueError("source_fps must be > 0")

    tile_count = len(_tile_regions(width, height, tile_size, overlap))
    decisions = [
        schedule_frame(policy, index, signals, tile_count=tile_count, config=config) for index in range(frames)
    ]
    total_calls = sum(item.total_calls for item in decisions)
    full_calls = sum(item.full_frame_calls for item in decisions)
    suspect_calls = sum(item.suspect_roi_calls for item in decisions)
    track_calls = sum(item.track_roi_calls for item in decisions)
    raw_calls = suspect_calls + track_calls
    enhanced_calls = sum(item.enhanced_roi_calls for item in decisions)
    max_frame_calls = max(item.total_calls for item in decisions)
    source_seconds = frames / source_fps

    return {
        "policy": policy.upper(),
        "width": width,
        "height": height,
        "frames": frames,
        "source_fps": source_fps,
        "tile_count": tile_count,
        "full_frame_calls": full_calls,
        "raw_roi_calls": raw_calls,
        "suspect_roi_calls": suspect_calls,
        "track_roi_calls": track_calls,
        "enhanced_roi_calls": enhanced_calls,
        "total_onnx_calls": total_calls,
        "onnx_calls_per_frame": total_calls / frames,
        "onnx_calls_per_source_second": total_calls / source_seconds,
        "max_onnx_calls_on_any_frame": max_frame_calls,
        "periodic_global_discovery_bound_frames": global_discovery_bound_frames(policy, config),
        "periodic_global_discovery_bound_seconds": global_discovery_bound_frames(policy, config) / source_fps,
        "config": asdict(config),
        "signals": asdict(signals),
    }


def target_probe_call_plan(
    frame_index: int,
    *,
    tile_count: int,
    config: SchedulerConfig,
) -> tuple[tuple[str, int | None], ...]:
    """Build a deterministic execution-cost call plan for target-PC probing.

    This does not claim ROI-quality behavior. Non-global ROI locations rotate
    deterministically so the probe measures realistic full-frame vs tile costs.
    """
    if frame_index < 0:
        raise ValueError("frame_index cannot be negative")
    if tile_count <= 0:
        raise ValueError("tile_count must be > 0")
    if not _detector_due(frame_index, config):
        return ()

    detector_index = frame_index // config.detector_every
    global_scan = detector_index % config.global_period == 0
    calls: list[tuple[str, int | None]] = []
    if global_scan:
        calls.append(("full", None))
        start = 0
    else:
        start = detector_index % tile_count

    remaining = max(0, config.max_calls_per_frame - len(calls))
    for offset in range(remaining):
        calls.append(("roi", (start + offset) % tile_count))
    return tuple(calls)


def target_scheduler_probe(
    video_path: str | Path,
    model_path: str | Path,
    *,
    input_size: int = 960,
    person_threshold: float = 0.12,
    tile_size: int = 640,
    overlap: float = 0.20,
    config: SchedulerConfig = SchedulerConfig(detector_every=2, global_period=15, max_calls_per_frame=1),
    source_commit: str,
    max_frames: int = 300,
    prefer_gpu: bool = True,
) -> dict[str, object]:
    """Measure a bounded scheduler call plan on the actual detector/provider.

    The ROI sequence is a deterministic cost proxy. This measures execution
    cost only and must not be interpreted as scheduler quality evidence.
    """
    import cv2

    from .detector import YoloOnnxDetector
    from .integrity import sha256_file

    video = Path(video_path)
    model = Path(model_path)
    if not video.is_file():
        raise FileNotFoundError(video)
    if not model.is_file():
        raise FileNotFoundError(model)
    if max_frames <= 0:
        raise ValueError("max_frames must be > 0")
    if input_size <= 0 or tile_size <= 0:
        raise ValueError("input_size and tile_size must be > 0")
    if not 0.0 <= person_threshold <= 1.0:
        raise ValueError("person_threshold must be in [0, 1]")
    if not 0.0 <= overlap < 1.0:
        raise ValueError("overlap must satisfy 0 <= overlap < 1")
    if len(source_commit) != 40 or any(char not in "0123456789abcdef" for char in source_commit.lower()):
        raise ValueError("source_commit must be a full 40-hex commit SHA")

    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open video: {video}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames: list[np.ndarray] = []
    try:
        while len(frames) < max_frames:
            ok, frame = capture.read()
            if not ok or frame is None:
                break
            frames.append(frame)
    finally:
        capture.release()
    if not frames:
        raise RuntimeError("video produced no frames")
    if not np.isfinite(fps) or fps <= 0.0:
        raise RuntimeError("video FPS is unavailable")

    regions = _tile_regions(width, height, tile_size, overlap)
    if not regions:
        raise RuntimeError("tile geometry produced no regions")

    detector = YoloOnnxDetector(
        model,
        input_size=input_size,
        conf_threshold=0.35,
        iou_threshold=0.45,
        prefer_gpu=prefer_gpu,
        class_thresholds={"person": person_threshold},
    )
    detector.detect(frames[0])  # provider/session warm-up; excluded from measurements.

    latencies_ms: list[float] = []
    stage_ms: dict[str, float] = {}
    full_calls = 0
    roi_calls = 0
    onnx_calls = 0
    started = time.perf_counter()
    for frame_index, frame in enumerate(frames):
        plan = target_probe_call_plan(frame_index, tile_count=len(regions), config=config)
        for kind, tile_index in plan:
            if kind == "full":
                image = frame
                full_calls += 1
            else:
                if tile_index is None:
                    raise RuntimeError("ROI call is missing tile index")
                x1, y1, x2, y2 = regions[tile_index]
                image = frame[y1:y2, x1:x2]
                roi_calls += 1
            call_started = time.perf_counter()
            detector.detect(image)
            latencies_ms.append((time.perf_counter() - call_started) * 1000.0)
            onnx_calls += int(detector.last_inference_calls)
            for name, value in detector.last_stage_ms.items():
                stage_ms[name] = stage_ms.get(name, 0.0) + float(value)
    wall_seconds = time.perf_counter() - started

    source_seconds = len(frames) / fps
    ordered = sorted(latencies_ms)
    p95_index = max(0, min(len(ordered) - 1, int(np.ceil(len(ordered) * 0.95)) - 1))
    discovery_frames = config.detector_every * config.global_period
    return {
        "schema": "spectratrack-vnext-target-scheduler-probe-v1",
        "scope": "execution-cost-only",
        "quality_evidence": False,
        "source_commit": source_commit,
        "quality_note": (
            "ROI locations are deterministic rotating proxy tiles. "
            "Use a frozen QA corpus/replay to evaluate scheduler quality."
        ),
        "video": str(video),
        "video_sha256": sha256_file(video),
        "model": str(model),
        "model_sha256": sha256_file(model),
        "providers": detector.providers,
        "input_size": input_size,
        "person_threshold": person_threshold,
        "width": width,
        "height": height,
        "frames": len(frames),
        "source_fps": fps,
        "source_seconds": source_seconds,
        "tile_size": tile_size,
        "overlap": overlap,
        "tile_count": len(regions),
        "config": asdict(config),
        "full_frame_calls": full_calls,
        "raw_roi_calls": roi_calls,
        "onnx_inference_calls": onnx_calls,
        "onnx_calls_per_source_second": onnx_calls / source_seconds,
        "wall_seconds": wall_seconds,
        "processing_seconds_per_source_second": wall_seconds / source_seconds,
        "latency_ms": {
            "median": statistics.median(latencies_ms),
            "p95": ordered[p95_index],
            "mean": statistics.fmean(latencies_ms),
        },
        "stage_ms": stage_ms,
        "periodic_global_discovery_bound_frames": discovery_frames,
        "periodic_global_discovery_bound_seconds": discovery_frames / fps,
        "gpu_vram_mb": None,
        "gpu_vram_reason": "not measured by this probe",
    }


def batching_probe(
    model_path: str | Path,
    *,
    prefer_gpu: bool = True,
    batch_sizes: Sequence[int] = (1, 2, 4),
    warmup: int = 2,
    repeats: int = 10,
) -> dict[str, object]:
    """Benchmark model batching in isolation. This never changes production inference."""
    import onnxruntime as ort

    if warmup < 0 or repeats <= 0:
        raise ValueError("warmup must be >= 0 and repeats must be > 0")

    available = ort.get_available_providers()
    providers: list[str] = []
    if prefer_gpu and "DmlExecutionProvider" in available:
        providers.append("DmlExecutionProvider")
    if "CPUExecutionProvider" in available:
        providers.append("CPUExecutionProvider")
    if not providers:
        providers = available

    options = ort.SessionOptions()
    if "DmlExecutionProvider" in providers:
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.enable_mem_pattern = False

    session = ort.InferenceSession(str(model_path), sess_options=options, providers=providers)
    input_meta = session.get_inputs()[0]
    shape = list(input_meta.shape)
    if len(shape) != 4:
        return {
            "provider": session.get_providers(),
            "input_shape": shape,
            "supported": False,
            "reason": "expected 4D NCHW detector input",
            "results": [],
            "gpu_vram_mb": None,
        }

    fixed_batch = shape[0] if isinstance(shape[0], int) else None
    spatial = shape[1:]
    if not all(isinstance(value, int) and value > 0 for value in spatial):
        return {
            "provider": session.get_providers(),
            "input_shape": shape,
            "supported": False,
            "reason": "dynamic/non-integer channel or spatial dimensions are not benchmarked by this probe",
            "results": [],
            "gpu_vram_mb": None,
        }

    results: list[dict[str, object]] = []
    for batch_size in batch_sizes:
        if batch_size <= 0:
            continue
        if fixed_batch is not None and batch_size != fixed_batch:
            results.append(
                {
                    "batch_size": batch_size,
                    "supported": False,
                    "reason": f"model has fixed batch dimension {fixed_batch}",
                }
            )
            continue

        tensor = np.zeros((batch_size, *spatial), dtype=np.float32)
        feeds = {input_meta.name: tensor}
        for _ in range(warmup):
            session.run(None, feeds)

        samples_ms: list[float] = []
        for _ in range(repeats):
            started = time.perf_counter()
            session.run(None, feeds)
            samples_ms.append((time.perf_counter() - started) * 1000.0)

        median_ms = statistics.median(samples_ms)
        results.append(
            {
                "batch_size": batch_size,
                "supported": True,
                "median_latency_ms": median_ms,
                "items_per_second": (batch_size * 1000.0 / median_ms) if median_ms > 0 else None,
                "samples_ms": samples_ms,
            }
        )

    supported_batches = [row for row in results if row.get("supported")]
    supports_multi_batch = any(int(row["batch_size"]) > 1 for row in supported_batches)
    return {
        "provider": session.get_providers(),
        "input_shape": shape,
        "supported": bool(supported_batches),
        "supports_multi_batch": supports_multi_batch,
        "results": results,
        "gpu_vram_mb": None,
        "gpu_vram_reason": "not measured by this probe; ONNX Runtime DirectML exposes no portable per-process VRAM counter",
    }


def analyze_perf_report(report: dict[str, object], source_fps: float) -> dict[str, object]:
    frames = int(report["frames"])
    elapsed_s = float(report["elapsed_s"])
    onnx_calls = int(report.get("onnx_inference_calls", 0))
    return {
        "source_report": report,
        "derived": runtime_rates(
            frames=frames,
            elapsed_s=elapsed_s,
            source_fps=source_fps,
            onnx_calls=onnx_calls,
        ),
    }


def _parse_resolution(value: str) -> tuple[int, int]:
    normalized = value.lower()
    if normalized in RESOLUTIONS:
        return RESOLUTIONS[normalized]
    if "x" not in normalized:
        raise argparse.ArgumentTypeError("resolution must be 720p/1080p/1440p/4k or WIDTHxHEIGHT")
    width, height = normalized.split("x", 1)
    try:
        parsed = int(width), int(height)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("resolution must contain integers") from exc
    if parsed[0] <= 0 or parsed[1] <= 0:
        raise argparse.ArgumentTypeError("resolution dimensions must be > 0")
    return parsed


def _dump(payload: object, output: str | None) -> None:
    encoded = json.dumps(payload, indent=2, sort_keys=True)
    if output:
        target = Path(output)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)


def main() -> int:
    parser = argparse.ArgumentParser(description="SpectraTrack vNext A4 research-only performance tools")
    sub = parser.add_subparsers(dest="command", required=True)

    compute = sub.add_parser("compute-model")
    compute.add_argument("--tile-size", type=int, default=640)
    compute.add_argument("--overlap", type=float, default=0.20)
    compute.add_argument("--source-fps", type=float, default=30.0)
    compute.add_argument("--output")

    sched = sub.add_parser("scheduler-sim")
    sched.add_argument("--policy", choices=POLICIES, required=True)
    sched.add_argument("--resolution", type=_parse_resolution, default=RESOLUTIONS["1080p"])
    sched.add_argument("--frames", type=int, default=300)
    sched.add_argument("--source-fps", type=float, default=30.0)
    sched.add_argument("--track-rois", type=int, default=2)
    sched.add_argument("--suspect-rois", type=int, default=2)
    sched.add_argument("--enhancement-eligible-rois", type=int, default=1)
    sched.add_argument("--detector-every", type=int, default=1)
    sched.add_argument("--global-period", type=int, default=15)
    sched.add_argument("--max-calls", type=int, default=4)
    sched.add_argument("--max-enhanced-calls", type=int, default=1)
    sched.add_argument("--output")

    analyze = sub.add_parser("analyze-report")
    analyze.add_argument("--report", required=True)
    analyze.add_argument("--source-fps", required=True, type=float)
    analyze.add_argument("--output")

    target = sub.add_parser("target-probe")
    target.add_argument("--video", required=True)
    target.add_argument("--model", required=True)
    target.add_argument("--source-commit", required=True)
    target.add_argument("--input-size", type=int, default=960)
    target.add_argument("--person-conf", type=float, default=0.12)
    target.add_argument("--tile-size", type=int, default=640)
    target.add_argument("--overlap", type=float, default=0.20)
    target.add_argument("--detector-every", type=int, default=2)
    target.add_argument("--global-period", type=int, default=15)
    target.add_argument("--max-calls", type=int, default=1)
    target.add_argument("--max-frames", type=int, default=300)
    target.add_argument("--cpu", action="store_true")
    target.add_argument("--output")

    batch = sub.add_parser("batch-probe")
    batch.add_argument("--model", required=True)
    batch.add_argument("--cpu", action="store_true")
    batch.add_argument("--batch-sizes", default="1,2,4")
    batch.add_argument("--warmup", type=int, default=2)
    batch.add_argument("--repeats", type=int, default=10)
    batch.add_argument("--output")

    args = parser.parse_args()

    if args.command == "compute-model":
        rows = []
        for name, (width, height) in RESOLUTIONS.items():
            bounds = current_compute_bounds(width, height, tile_size=args.tile_size, overlap=args.overlap)
            minimum = bounds["minimum"]
            maximum = bounds["maximum"]
            rows.append(
                {
                    "name": name,
                    "width": width,
                    "height": height,
                    "tile_count": len(minimum.tile_regions),
                    "tile_regions": minimum.tile_regions,
                    "full_frame_calls": 1,
                    "raw_tile_calls": minimum.raw_tile_calls,
                    "enhanced_tile_calls_min": minimum.enhanced_tile_calls,
                    "enhanced_tile_calls_max": maximum.enhanced_tile_calls,
                    "total_calls_min": minimum.total_onnx_calls,
                    "total_calls_max": maximum.total_onnx_calls,
                    "calls_per_source_second_min": minimum.total_onnx_calls * args.source_fps,
                    "calls_per_source_second_max": maximum.total_onnx_calls * args.source_fps,
                }
            )
        _dump(
            {
                "tile_size": args.tile_size,
                "overlap": args.overlap,
                "source_fps_reference": args.source_fps,
                "rows": rows,
            },
            args.output,
        )
        return 0

    if args.command == "scheduler-sim":
        width, height = args.resolution
        result = simulate_scheduler(
            args.policy,
            width=width,
            height=height,
            frames=args.frames,
            source_fps=args.source_fps,
            signals=FrameSignals(
                track_rois=args.track_rois,
                suspect_rois=args.suspect_rois,
                enhancement_eligible_rois=args.enhancement_eligible_rois,
            ),
            config=SchedulerConfig(
                detector_every=args.detector_every,
                global_period=args.global_period,
                max_calls_per_frame=args.max_calls,
                max_enhanced_calls_per_frame=args.max_enhanced_calls,
            ),
        )
        _dump(result, args.output)
        return 0

    if args.command == "analyze-report":
        report = json.loads(Path(args.report).read_text(encoding="utf-8"))
        _dump(analyze_perf_report(report, args.source_fps), args.output)
        return 0

    if args.command == "target-probe":
        result = target_scheduler_probe(
            args.video,
            args.model,
            input_size=args.input_size,
            person_threshold=args.person_conf,
            tile_size=args.tile_size,
            overlap=args.overlap,
            source_commit=args.source_commit,
            config=SchedulerConfig(
                detector_every=args.detector_every,
                global_period=args.global_period,
                max_calls_per_frame=args.max_calls,
                max_enhanced_calls_per_frame=0,
            ),
            max_frames=args.max_frames,
            prefer_gpu=not args.cpu,
        )
        _dump(result, args.output)
        return 0

    sizes = tuple(int(item.strip()) for item in args.batch_sizes.split(",") if item.strip())
    result = batching_probe(
        args.model,
        prefer_gpu=not args.cpu,
        batch_sizes=sizes,
        warmup=args.warmup,
        repeats=args.repeats,
    )
    _dump(result, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
