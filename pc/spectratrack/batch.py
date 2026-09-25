from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path

import cv2
import numpy as np

from .appearance import attach_appearance, crossvideo_descriptor
from .capture import CaptureConfig, RobustCapture
from .crossvideo import TrackletSummary, build_cross_video_graph, normalize_descriptor
from .detector import YoloOnnxDetector
from .enhance import crop_with_margin
from .integrity import sha256_file
from .motion import GlobalMotionEstimator
from .tracker import MultiObjectTracker


VIDEO_SUFFIXES = {".mp4", ".mov", ".mkv", ".avi", ".m4v", ".webm"}


@dataclass
class _Accumulator:
    video: str
    local_track_id: int
    class_id: int
    label: str
    first_frame: int
    last_frame: int
    observations: int = 0
    score_sum: float = 0.0
    quality_sum: float = 0.0
    best_value: float = -1.0
    best_frame: int = 0
    descriptor_sum: np.ndarray | None = None
    best_crop: np.ndarray | None = None
    gallery: list[tuple[float, ...]] | None = None

    def add(
        self,
        frame_index: int,
        score: float,
        quality: float,
        descriptor: tuple[float, ...],
        frame: np.ndarray,
        bbox: tuple[float, float, float, float],
    ) -> None:
        arr = np.asarray(descriptor, dtype=np.float64)
        if self.gallery is None:
            self.gallery = []
        if self.descriptor_sum is None:
            self.descriptor_sum = np.zeros_like(arr)
        if self.descriptor_sum.shape != arr.shape:
            return
        self.descriptor_sum += arr
        if len(self.gallery) < 6:
            keep = True
            for existing in self.gallery:
                a = np.asarray(existing, dtype=np.float64)
                denom = float(np.linalg.norm(a) * np.linalg.norm(arr))
                similarity = float(np.dot(a, arr) / denom) if denom > 1e-12 else 0.0
                if similarity >= 0.985:
                    keep = False
                    break
            if keep:
                self.gallery.append(tuple(float(v) for v in arr))
        self.last_frame = frame_index
        self.observations += 1
        self.score_sum += float(score)
        self.quality_sum += float(quality)
        value = float(score) * 0.65 + float(quality) * 0.35
        if value > self.best_value:
            self.best_value = value
            self.best_frame = frame_index
            self.best_crop = crop_with_margin(frame, bbox, margin=0.12)

    def finish(self, fps: float, preview_path: str | None = None) -> TrackletSummary | None:
        if self.observations <= 0 or self.descriptor_sum is None:
            return None
        descriptor = normalize_descriptor(self.descriptor_sum.tolist())
        return TrackletSummary(
            video=self.video,
            local_track_id=self.local_track_id,
            class_id=self.class_id,
            label=self.label,
            first_frame=self.first_frame,
            last_frame=self.last_frame,
            observations=self.observations,
            mean_score=self.score_sum / self.observations,
            mean_quality=self.quality_sum / self.observations,
            best_frame=self.best_frame,
            fps=float(fps),
            descriptor=descriptor,
            preview_path=preview_path,
            gallery=tuple(self.gallery or ()),
        )


def _looks_generated(path: Path) -> bool:
    stem = path.stem.lower()
    return stem.endswith("_spectratrack") or stem in {"result_max", "result_stabilized", "analyzed"}


def discover_videos(
    input_dir: str | Path,
    recursive: bool = False,
    include_derived: bool = False,
) -> list[Path]:
    root = Path(input_dir)
    if not root.exists() or not root.is_dir():
        raise FileNotFoundError(f"Input directory not found: {root}")
    iterator = root.rglob("*") if recursive else root.glob("*")
    return sorted(
        (
            p for p in iterator
            if p.is_file()
            and p.suffix.lower() in VIDEO_SUFFIXES
            and (include_derived or not _looks_generated(p))
        ),
        key=lambda p: str(p).lower(),
    )


def _parse_classes(text: str) -> set[str]:
    return {part.strip().lower() for part in text.split(",") if part.strip()}


def video_identifier(input_dir: str | Path, path: str | Path) -> str:
    root = Path(input_dir).resolve()
    candidate = Path(path).resolve()
    try:
        return candidate.relative_to(root).as_posix()
    except ValueError:
        return candidate.name


def analyze_video(
    path: Path,
    detector: YoloOnnxDetector,
    detect_every: int,
    class_filter: set[str],
    min_observations: int,
    max_frames: int = 0,
    preview_dir: Path | None = None,
    video_id: str | None = None,
) -> list[TrackletSummary]:
    capture = RobustCapture(str(path), CaptureConfig(backend="auto", reconnect_attempts=0))
    if not capture.is_opened():
        capture.release()
        raise RuntimeError(f"Cannot open video: {path}")

    props = capture.actual_properties()
    fps = float(props["fps"])
    identifier = video_id or path.name
    tracker = MultiObjectTracker()
    motion = GlobalMotionEstimator()
    accumulators: dict[int, _Accumulator] = {}
    frame_index = 0

    try:
        while True:
            read = capture.read()
            if not read.ok or read.frame is None:
                break
            frame = read.frame
            frame_index += 1
            cam = motion.update(frame)
            camera_shift = (cam.dx, cam.dy) if cam.valid else (0.0, 0.0)
            camera_transform = cam.affine if cam.valid else None
            should_detect = ((frame_index - 1) % detect_every) == 0
            if should_detect:
                detections = detector.detect(frame)
                if class_filter:
                    detections = [d for d in detections if d.label.lower() in class_filter]
                attach_appearance(frame, detections)
                tracks = tracker.update(
                    detections,
                    camera_motion=camera_shift,
                    camera_transform=camera_transform,
                )
            else:
                tracks = tracker.predict_only(
                    camera_motion=camera_shift,
                    camera_transform=camera_transform,
                )

            if should_detect:
                for tr in tracks:
                    if not tr.confirmed or tr.missed != 0 or tr.appearance is None:
                        continue
                    acc = accumulators.get(tr.track_id)
                    if acc is None:
                        acc = _Accumulator(
                            video=identifier,
                            local_track_id=tr.track_id,
                            class_id=tr.class_id,
                            label=tr.label,
                            first_frame=frame_index,
                            last_frame=frame_index,
                        )
                        accumulators[tr.track_id] = acc
                    descriptor = crossvideo_descriptor(frame, tr.bbox)
                    if descriptor is None:
                        continue
                    acc.add(
                        frame_index,
                        tr.last_detection_score or tr.score,
                        tr.quality,
                        descriptor,
                        frame,
                        tr.bbox,
                    )

            if max_frames > 0 and frame_index >= max_frames:
                break
    finally:
        capture.release()

    summaries = []
    if preview_dir is not None:
        preview_dir.mkdir(parents=True, exist_ok=True)
    safe_stem = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in identifier)
    for acc in accumulators.values():
        summary = acc.finish(fps)
        if summary is None or summary.observations < min_observations:
            continue
        if preview_dir is not None and acc.best_crop is not None:
            candidate = preview_dir / f"{safe_stem}_T{acc.local_track_id:03d}.jpg"
            if cv2.imwrite(str(candidate), acc.best_crop):
                summary.preview_path = str(candidate)
        summaries.append(summary)
    summaries.sort(key=lambda t: (t.class_id, t.local_track_id))
    return summaries


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Analyze a folder of videos and build a conservative cross-video appearance graph.",
        allow_abbrev=False,
    )
    parser.add_argument("--model", required=True, help="Compatible fixed-size YOLO ONNX model")
    parser.add_argument("--input-dir", required=True, help="Folder containing videos")
    parser.add_argument("--output", default="cross_video_graph.json")
    parser.add_argument("--input-size", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--detect-every", type=int, default=1)
    parser.add_argument("--classes", default="")
    parser.add_argument("--candidate-threshold", type=float, default=0.86)
    parser.add_argument("--strong-threshold", type=float, default=0.94)
    parser.add_argument("--min-observations", type=int, default=3)
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument(
        "--include-derived",
        action="store_true",
        help="Include SpectraTrack-generated video outputs in the input scan",
    )
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--max-frames-per-video", type=int, default=0)
    parser.add_argument("--no-previews", action="store_true", help="Do not save one best crop per tracklet")
    parser.add_argument("--no-html", action="store_true", help="Do not generate the local HTML review report")
    parser.add_argument("--review", default="", help="Optional exported review JSON from a previous HTML report")
    args = parser.parse_args()

    if args.detect_every < 1:
        raise SystemExit("--detect-every must be >= 1")
    if args.min_observations < 1:
        raise SystemExit("--min-observations must be >= 1")

    videos = discover_videos(
        args.input_dir,
        recursive=args.recursive,
        include_derived=args.include_derived,
    )
    if not videos:
        raise SystemExit(f"No supported videos found in: {args.input_dir}")

    output = Path(args.output)
    preview_dir = None if args.no_previews else output.with_name(output.stem + "_samples")

    detector = YoloOnnxDetector(
        args.model,
        input_size=args.input_size,
        conf_threshold=args.conf,
        iou_threshold=args.iou,
        prefer_gpu=not args.cpu,
    )
    class_filter = _parse_classes(args.classes)
    all_tracklets: list[TrackletSummary] = []
    failures: list[dict[str, str]] = []

    for index, path in enumerate(videos, start=1):
        print(f"[{index}/{len(videos)}] analyzing {path.name}")
        try:
            tracklets = analyze_video(
                path,
                detector,
                detect_every=args.detect_every,
                class_filter=class_filter,
                min_observations=args.min_observations,
                max_frames=args.max_frames_per_video,
                preview_dir=preview_dir,
                video_id=video_identifier(args.input_dir, path),
            )
            all_tracklets.extend(tracklets)
            print(f"  tracklets={len(tracklets)}")
        except Exception as exc:
            failures.append({"video": path.name, "error": str(exc)})
            print(f"  FAILED: {exc}")

    review_decisions = {}
    if args.review:
        review_payload = json.loads(Path(args.review).read_text(encoding="utf-8"))
        for item in review_payload.get("decisions", []):
            left = str(item.get("left", ""))
            right = str(item.get("right", ""))
            decision = str(item.get("decision", "")).lower()
            if left and right and decision in {"same", "different", "unsure"}:
                review_decisions[tuple(sorted((left, right)))] = decision

    graph = build_cross_video_graph(
        all_tracklets,
        candidate_threshold=args.candidate_threshold,
        strong_threshold=args.strong_threshold,
        review_decisions=review_decisions,
    )
    graph["run"] = {
        "model": str(Path(args.model)),
        "model_sha256": sha256_file(args.model),
        "providers": detector.providers,
        "input_dir": str(Path(args.input_dir)),
        "videos_seen": len(videos),
        "videos_failed": len(failures),
        "detect_every": args.detect_every,
        "classes": sorted(class_filter),
        "failures": failures,
        "review_file": args.review or None,
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(graph, ensure_ascii=False, indent=2), encoding="utf-8")
    if not args.no_html:
        from .crossvideo_report import write_html_report

        report_path = output.with_suffix(".html")
        write_html_report(graph, report_path)
        print(f"report={report_path}")
    print(
        f"done videos={len(videos)} failed={len(failures)} "
        f"tracklets={len(graph['tracklets'])} entities={len(graph['entities'])} edges={len(graph['edges'])}"
    )
    print(f"graph={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
