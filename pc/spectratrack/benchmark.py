from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

from .capture import open_capture
from .cmc import CameraMotionEstimator
from .detector import YoloOnnxDetector
from .metrics import RollingProfiler
from .tracker import MultiObjectTracker, TrackerConfig


def main() -> int:
    p = argparse.ArgumentParser(description="Headless SpectraTrack PC benchmark")
    p.add_argument("--model", required=True)
    p.add_argument("--source", default="0")
    p.add_argument("--frames", type=int, default=300)
    p.add_argument("--input-size", type=int, default=640)
    p.add_argument("--conf", type=float, default=0.35)
    p.add_argument("--iou", type=float, default=0.45)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--no-cmc", action="store_true")
    p.add_argument("--output", default="")
    args = p.parse_args()

    detector = YoloOnnxDetector(args.model, args.input_size, args.conf, args.iou, prefer_gpu=not args.cpu)
    tracker = MultiObjectTracker(config=TrackerConfig())
    cmc = CameraMotionEstimator()
    profiler = RollingProfiler(window=max(120, args.frames))
    cap = open_capture(args.source)
    tracks = []
    started = perf_counter()
    processed = 0

    try:
        while processed < args.frames:
            ok, frame = cap.read()
            if not ok:
                break
            with profiler.measure("cmc"):
                motion = cmc.update(frame, [t.bbox for t in tracks if t.confirmed]) if not args.no_cmc else None
            with profiler.measure("detect"):
                detections = detector.detect(frame)
            with profiler.measure("track"):
                tracks = tracker.update(
                    detections,
                    camera_affine=motion.matrix if motion is not None and motion.valid else None,
                )
            processed += 1
    finally:
        cap.release()

    elapsed = max(perf_counter() - started, 1e-9)
    result = {
        "frames": processed,
        "wall_seconds": elapsed,
        "throughput_fps": processed / elapsed,
        "providers": detector.providers,
        "model": str(Path(args.model)),
        "input_size": args.input_size,
        "cmc": not args.no_cmc,
        "metrics": profiler.summary(),
    }
    payload = json.dumps(result, indent=2, sort_keys=True)
    print(payload)
    if args.output:
        Path(args.output).write_text(payload + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
