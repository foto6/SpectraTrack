from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time
from typing import Any, Iterable, Mapping

import cv2

from ..detector import YoloOnnxDetector
from ..integrity import sha256_file
from ..qa_benchmark import GroundTruthFrame, ground_truth_sha256, load_ground_truth
from .vnext_detection_fusion import (
    FusionConfig,
    ResearchFrame,
    collect_prefusion_people_recall,
    convert_prefusion_to_replay,
    evaluate_fusion,
    read_prefusion_dump,
    write_prefusion_dump,
)
from .vnext_detector_backend_lab import (
    _direct_image_path,
    _image_sequence_path,
    _load_benchmark_source_records,
)


METHODS = ("hard-nms", "conservative-nmm", "weighted", "evidence-aware")


def _timestamp_s(record: Mapping[str, Any], frame_index: int) -> float:
    fps = record.get("source_fps")
    if isinstance(fps, (int, float)) and not isinstance(fps, bool) and float(fps) > 0.0:
        return float(frame_index) / float(fps)
    return float(frame_index)


def _source_image_path(video_root: str | Path, record: dict[str, Any]) -> tuple[Path, str]:
    direct = _direct_image_path(video_root, record)
    if direct is not None:
        return direct, "image"
    sequence = _image_sequence_path(video_root, record)
    if sequence is not None:
        return sequence, "image_sequence"
    video = record.get("video")
    frame = record.get("frame")
    raise FileNotFoundError(f"no canonical image source for {video!r}#{frame!r}")


def collect_corpus_video_frames(
    detector: YoloOnnxDetector,
    annotations: Iterable[GroundTruthFrame],
    source_records: Mapping[tuple[str, int], dict[str, Any]],
    *,
    video_root: str | Path,
    person_threshold: float,
    tile_size: int,
    tile_overlap: float,
    frame_limit: int = 0,
) -> tuple[list[ResearchFrame], dict[str, Any]]:
    ordered = sorted(annotations, key=lambda item: item.frame)
    if frame_limit > 0:
        ordered = ordered[:frame_limit]
    if not ordered:
        raise ValueError("no annotations selected")

    frames: list[ResearchFrame] = []
    inference_calls = 0
    stage_ms: dict[str, float] = {}
    source_modes: set[str] = set()
    started = time.perf_counter()

    for annotation in ordered:
        key = (annotation.video, annotation.frame)
        record = source_records.get(key)
        if record is None:
            raise ValueError(f"missing source record for {annotation.video}#{annotation.frame}")
        image_path, source_mode = _source_image_path(video_root, record)
        source_modes.add(source_mode)
        frame_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if frame_bgr is None:
            raise RuntimeError(f"cannot read benchmark image: {image_path}")

        candidates = collect_prefusion_people_recall(
            detector,
            frame_bgr,
            person_threshold=person_threshold,
            tile_size=tile_size,
            tile_overlap=tile_overlap,
        )
        inference_calls += int(detector.last_inference_calls)
        for name, value in detector.last_stage_ms.items():
            stage_ms[name] = stage_ms.get(name, 0.0) + float(value)
        height, width = frame_bgr.shape[:2]
        frames.append(
            ResearchFrame(
                annotation.video,
                annotation.frame,
                _timestamp_s(record, annotation.frame),
                width,
                height,
                tuple(candidates),
                annotation.objects,
                annotation.tags,
            )
        )

    return frames, {
        "policy_runs": len(frames),
        "inference_calls": inference_calls,
        "wall_time_s": time.perf_counter() - started,
        "stage_ms": stage_ms,
        "source_modes": sorted(source_modes),
    }


def _per_video_performance(summary: Mapping[str, Any], *, reused: bool) -> dict[str, Any]:
    wall_time_s = float(summary.get("wall_time_s", 0.0))
    inference_calls = int(summary.get("inference_calls", 0))
    return {
        "policy_runs": int(summary.get("policy_runs", 0)),
        "inference_calls": inference_calls,
        "new_inference_calls": 0 if reused else inference_calls,
        "reused_inference_calls": inference_calls if reused else 0,
        "represented_detector_wall_s": wall_time_s,
        "new_detector_wall_s": 0.0 if reused else wall_time_s,
        "reused_detector_wall_s": wall_time_s if reused else 0.0,
        "stage_ms": {name: float(value) for name, value in dict(summary.get("stage_ms", {})).items()},
        "source_modes": sorted(summary.get("source_modes", [])),
        "evidence_source": "reused_prefusion" if reused else "new_inference",
    }


def attach_ground_truth(
    frames: Iterable[ResearchFrame],
    annotations: Mapping[tuple[str, int], GroundTruthFrame],
) -> list[ResearchFrame]:
    attached: list[ResearchFrame] = []
    for frame in frames:
        annotation = annotations.get((frame.video, frame.frame))
        if annotation is None:
            raise ValueError(f"missing ground truth for {frame.video}#{frame.frame}")
        attached.append(
            ResearchFrame(
                frame.video,
                frame.frame,
                frame.timestamp_s,
                frame.width,
                frame.height,
                frame.candidates,
                annotation.objects,
                annotation.tags,
            )
        )
    return attached


def evaluate_methods(
    frames: Iterable[ResearchFrame],
    *,
    methods: Iterable[str],
    config: FusionConfig,
    match_iou: float,
    per_video: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    items = list(frames)
    requested = tuple(dict.fromkeys(methods))
    unknown = [method for method in requested if method not in METHODS]
    if unknown:
        raise ValueError(f"unknown fusion methods: {unknown}")
    if not requested:
        raise ValueError("at least one fusion method is required")

    aggregate = {
        method: evaluate_fusion(items, method=method, config=config, match_iou=match_iou)
        for method in requested
    }
    by_video: dict[str, Any] = {}
    if per_video:
        grouped: dict[str, list[ResearchFrame]] = {}
        for frame in items:
            grouped.setdefault(frame.video, []).append(frame)
        for video, video_frames in sorted(grouped.items()):
            by_video[video] = {
                method: evaluate_fusion(video_frames, method=method, config=config, match_iou=match_iou)
                for method in requested
            }
    return aggregate, by_video


def _safe_stem(video: str) -> str:
    stem = "".join(char if char.isalnum() or char in "-_." else "_" for char in video)
    return stem.strip("._") or "video"


def export_canonical_replays(
    *,
    prefusion_dir: str | Path,
    replay_dir: str | Path,
    videos: Iterable[str],
    methods: Iterable[str],
    config: FusionConfig,
) -> dict[str, dict[str, str]]:
    """Convert frozen per-video prefusion evidence to canonical A2 replays.

    This performs no detector inference. The caller is responsible for ensuring
    the prefusion artifacts were validated/reused under the corpus runner's
    provenance contract.
    """
    source = Path(prefusion_dir)
    target = Path(replay_dir)
    if not source.is_dir():
        raise FileNotFoundError(f"prefusion directory is missing: {source}")
    target.mkdir(parents=True, exist_ok=True)

    requested_methods = tuple(dict.fromkeys(methods))
    unknown = [method for method in requested_methods if method not in METHODS]
    if unknown:
        raise ValueError(f"unknown fusion methods: {unknown}")
    if not requested_methods:
        raise ValueError("at least one fusion method is required")

    outputs: dict[str, dict[str, str]] = {}
    for video in sorted(set(videos)):
        stem = _safe_stem(video)
        prefusion_path = source / f"{stem}.jsonl"
        if not prefusion_path.is_file():
            raise FileNotFoundError(f"prefusion artifact is missing for {video}: {prefusion_path}")
        per_method: dict[str, str] = {}
        for method in requested_methods:
            replay_path = target / f"{stem}.{method}.replay.jsonl"
            convert_prefusion_to_replay(
                prefusion_path,
                replay_path,
                method=method,
                config=config,
            )
            per_method[method] = str(replay_path)
        outputs[video] = per_method
    return outputs


def _prefusion_metadata(
    *,
    args: argparse.Namespace,
    detector: YoloOnnxDetector,
    video: str,
    frame: ResearchFrame,
    model_sha256: str,
    gt_sha256: str,
) -> dict[str, Any]:
    return {
        "source_commit": args.source_commit,
        "corpus_revision": args.corpus_revision,
        "ground_truth_sha256": gt_sha256,
        "video": video,
        "video_sha256": None,
        "detector": "current-yolo-onnx-prefusion",
        "model_sha256": model_sha256,
        "provider": ",".join(detector.providers),
        "config": {
            "input_size": args.input_size,
            "actual_input_width": detector.input_w,
            "actual_input_height": detector.input_h,
            "conf": args.conf,
            "decoder_iou": args.decoder_iou,
            "person_conf": args.person_conf,
            "tile_size": args.tile_size,
            "tile_overlap": args.tile_overlap,
            "decoder_local_nms": True,
            "final_cross_pass_fusion": "not_applied",
            "enhancement": "off",
            "frame_limit_per_video": args.frame_limit_per_video,
        },
        "width": frame.width,
        "height": frame.height,
    }


def _validate_reusable_prefusion(
    metadata: Mapping[str, Any],
    *,
    args: argparse.Namespace,
    video: str,
    model_sha256: str,
    gt_sha256: str,
) -> None:
    if metadata.get("video") != video:
        raise ValueError(f"prefusion video mismatch for {video}")
    if metadata.get("source_commit") != args.source_commit:
        raise ValueError(f"prefusion source_commit mismatch for {video}")
    if metadata.get("corpus_revision") != args.corpus_revision:
        raise ValueError(f"prefusion corpus revision mismatch for {video}")
    if metadata.get("ground_truth_sha256") != gt_sha256:
        raise ValueError(f"prefusion ground-truth hash mismatch for {video}")
    if metadata.get("model_sha256") != model_sha256:
        raise ValueError(f"prefusion model hash mismatch for {video}")
    config = metadata.get("config")
    if not isinstance(config, dict):
        raise ValueError(f"prefusion config missing for {video}")
    expected = {
        "input_size": args.input_size,
        "conf": args.conf,
        "decoder_iou": args.decoder_iou,
        "person_conf": args.person_conf,
        "tile_size": args.tile_size,
        "tile_overlap": args.tile_overlap,
        "frame_limit_per_video": args.frame_limit_per_video,
    }
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(f"prefusion config mismatch for {video}: {key}")


def run_corpus_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    if getattr(args, "replay_dir", None) and not getattr(args, "prefusion_dir", None):
        raise ValueError("--replay-dir requires --prefusion-dir so replay provenance stays tied to frozen prefusion evidence")

    model_path = Path(args.model)
    if not model_path.is_file():
        raise FileNotFoundError(f"model not found: {model_path}")

    annotations = load_ground_truth(args.ground_truth)
    source_records = _load_benchmark_source_records(args.ground_truth)
    selected = set(args.include_video or [])
    available = {item.video for item in annotations}
    missing = selected - available
    if missing:
        raise ValueError(f"requested videos not present in ground truth: {sorted(missing)}")
    if selected:
        annotations = [item for item in annotations if item.video in selected]
    if not annotations:
        raise ValueError("no ground-truth frames selected")

    by_video: dict[str, list[GroundTruthFrame]] = {}
    for annotation in annotations:
        by_video.setdefault(annotation.video, []).append(annotation)

    detector = YoloOnnxDetector(
        model_path,
        input_size=args.input_size,
        conf_threshold=args.conf,
        iou_threshold=args.decoder_iou,
        prefer_gpu=not args.cpu,
    )
    model_sha = sha256_file(model_path)
    gt_sha = ground_truth_sha256(args.ground_truth)
    annotation_map = {(item.video, item.frame): item for item in annotations}
    prefusion_dir = Path(args.prefusion_dir) if args.prefusion_dir else None
    if prefusion_dir is not None:
        prefusion_dir.mkdir(parents=True, exist_ok=True)

    all_frames: list[ResearchFrame] = []
    performance = {
        "policy_runs": 0,
        "inference_calls": 0,
        "new_inference_calls": 0,
        "reused_inference_calls": 0,
        "stage_ms": {},
        "source_modes": set(),
        "new_videos": 0,
        "reused_videos": 0,
        "represented_detector_wall_s": 0.0,
        "new_detector_wall_s": 0.0,
        "reused_detector_wall_s": 0.0,
        "per_video": {},
    }
    benchmark_started = time.perf_counter()

    for video, video_annotations in sorted(by_video.items()):
        prefusion_path = None if prefusion_dir is None else prefusion_dir / f"{_safe_stem(video)}.jsonl"
        if args.resume and prefusion_path is not None and prefusion_path.is_file():
            metadata, cached_frames, summary = read_prefusion_dump(prefusion_path)
            _validate_reusable_prefusion(
                metadata,
                args=args,
                video=video,
                model_sha256=model_sha,
                gt_sha256=gt_sha,
            )
            frames = attach_ground_truth(cached_frames, annotation_map)
            performance["reused_videos"] += 1
            performance["reused_inference_calls"] += int(summary.get("inference_calls", 0))
        else:
            frames, summary = collect_corpus_video_frames(
                detector,
                video_annotations,
                source_records,
                video_root=args.video_root,
                person_threshold=args.person_conf,
                tile_size=args.tile_size,
                tile_overlap=args.tile_overlap,
                frame_limit=args.frame_limit_per_video,
            )
            performance["new_videos"] += 1
            performance["new_inference_calls"] += int(summary.get("inference_calls", 0))
            if prefusion_path is not None:
                write_prefusion_dump(
                    prefusion_path,
                    _prefusion_metadata(
                        args=args,
                        detector=detector,
                        video=video,
                        frame=frames[0],
                        model_sha256=model_sha,
                        gt_sha256=gt_sha,
                    ),
                    frames,
                    summary,
                )

        reused = bool(args.resume and prefusion_path is not None and prefusion_path.is_file())
        video_performance = _per_video_performance(summary, reused=reused)
        performance["per_video"][video] = video_performance
        performance["represented_detector_wall_s"] += video_performance["represented_detector_wall_s"]
        performance["new_detector_wall_s"] += video_performance["new_detector_wall_s"]
        performance["reused_detector_wall_s"] += video_performance["reused_detector_wall_s"]
        performance["policy_runs"] += int(summary.get("policy_runs", len(frames)))
        performance["inference_calls"] += int(summary.get("inference_calls", 0))
        for name, value in dict(summary.get("stage_ms", {})).items():
            stage_ms = performance["stage_ms"]
            stage_ms[name] = stage_ms.get(name, 0.0) + float(value)
        performance["source_modes"].update(summary.get("source_modes", []))
        all_frames.extend(frames)

    config = FusionConfig(
        iou_threshold=args.fusion_iou,
        center_ratio=args.center_ratio,
        size_ratio=args.size_ratio,
        score_power=args.score_power,
        full_weight=args.full_weight,
        tile_weight=args.tile_weight,
        evidence_weak_score=args.evidence_weak_score,
        evidence_solo_score=args.evidence_solo_score,
        evidence_strong_score=args.evidence_strong_score,
        evidence_min_sources=args.evidence_min_sources,
    )
    config.validate()
    methods, per_video = evaluate_methods(
        all_frames,
        methods=args.method,
        config=config,
        match_iou=args.match_iou,
        per_video=args.per_video,
    )
    replay_outputs: dict[str, dict[str, str]] = {}
    if getattr(args, "replay_dir", None):
        replay_outputs = export_canonical_replays(
            prefusion_dir=args.prefusion_dir,
            replay_dir=args.replay_dir,
            videos=by_video,
            methods=args.method,
            config=config,
        )
    return {
        "schema": "spectratrack-vnext-fusion-corpus-v1",
        "source_commit": args.source_commit,
        "corpus_revision": args.corpus_revision,
        "ground_truth_sha256": gt_sha,
        "model": str(model_path),
        "model_sha256": model_sha,
        "providers": detector.providers,
        "provider_priority": detector.providers[0] if detector.providers else None,
        "selected_videos": sorted(by_video),
        "frame_count": len(all_frames),
        "settings": {
            "input_size": args.input_size,
            "conf": args.conf,
            "decoder_iou": args.decoder_iou,
            "person_conf": args.person_conf,
            "tile_size": args.tile_size,
            "tile_overlap": args.tile_overlap,
            "match_iou": args.match_iou,
            "fusion": asdict(config),
            "methods": list(dict.fromkeys(args.method)),
            "frame_limit_per_video": args.frame_limit_per_video,
        },
        "performance": {
            **performance,
            "source_modes": sorted(performance["source_modes"]),
            "benchmark_wall_s": time.perf_counter() - benchmark_started,
        },
        "methods": methods,
        "per_video": per_video,
        "replays": replay_outputs,
        "note": "Research corpus evidence only; no production behavior changed.",
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SpectraTrack vNext A1 public-corpus fusion benchmark")
    parser.add_argument("--model", required=True)
    parser.add_argument("--ground-truth", required=True)
    parser.add_argument("--video-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--corpus-revision", required=True)
    parser.add_argument("--include-video", action="append", default=[])
    parser.add_argument("--prefusion-dir")
    parser.add_argument("--replay-dir")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--per-video", action="store_true")
    parser.add_argument("--frame-limit-per-video", type=int, default=0)
    parser.add_argument("--input-size", type=int, default=960)
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--decoder-iou", type=float, default=0.45)
    parser.add_argument("--person-conf", type=float, default=0.12)
    parser.add_argument("--tile-size", type=int, default=640)
    parser.add_argument("--tile-overlap", type=float, default=0.20)
    parser.add_argument("--match-iou", type=float, default=0.50)
    parser.add_argument("--fusion-iou", type=float, default=0.55)
    parser.add_argument("--center-ratio", type=float, default=0.20)
    parser.add_argument("--size-ratio", type=float, default=1.80)
    parser.add_argument("--score-power", type=float, default=1.0)
    parser.add_argument("--full-weight", type=float, default=1.0)
    parser.add_argument("--tile-weight", type=float, default=1.0)
    parser.add_argument("--evidence-weak-score", type=float, default=0.12)
    parser.add_argument("--evidence-solo-score", type=float, default=0.20)
    parser.add_argument("--evidence-strong-score", type=float, default=0.35)
    parser.add_argument("--evidence-min-sources", type=int, default=2)
    parser.add_argument("--method", action="append", choices=METHODS)
    parser.add_argument("--cpu", action="store_true")
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    if not args.method:
        args.method = list(METHODS)
    report = run_corpus_benchmark(args)
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report["methods"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
