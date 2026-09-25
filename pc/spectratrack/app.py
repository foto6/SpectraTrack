from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2

from .calibration import CameraGeometry
from .capture import open_capture
from .cmc import CameraMotionEstimator, MotionEstimate
from .detector import YoloOnnxDetector
from .enhance import ENHANCE_MODES, apply_enhancement, crop_with_margin, next_enhancement_mode, run_realesrgan_snapshot
from .hud import compose_hud
from .mask import TargetMaskCache
from .metrics import RollingProfiler
from .model_security import sha256_file, verify_sha256
from .recording import SessionRecorder
from .scheduler import DetectionScheduler
from .stabilize import VideoStabilizer
from .tracker import MultiObjectTracker, TrackerConfig


class UiState:
    def __init__(self) -> None:
        self.tracks = []
        self.selected_id: int | None = None
        self.frame_width = 0


def main() -> int:
    parser = argparse.ArgumentParser(description="SpectraTrack PC v0.2 local object detection/tracking HUD")
    parser.add_argument("--model", required=True, help="Path to YOLOv8/YOLO11-style ONNX model")
    parser.add_argument("--model-sha256", default="", help="Optional expected SHA-256; abort on mismatch")
    parser.add_argument("--source", default="0", help="Camera index or video path")
    parser.add_argument("--input-size", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--cpu", action="store_true", help="Disable DirectML preference")
    parser.add_argument("--enhance-mode", choices=ENHANCE_MODES, default="off")
    parser.add_argument("--stabilize", action="store_true", help="Start with optical stabilization enabled")
    parser.add_argument("--cmc", action=argparse.BooleanOptionalAction, default=True, help="Camera-motion compensation")
    parser.add_argument("--mask", action="store_true", help="Start with classical selected-target GrabCut outline")
    parser.add_argument("--hfov", type=float, default=None, help="Calibrated/known horizontal camera FOV in degrees")
    parser.add_argument("--track-high", type=float, default=0.45)
    parser.add_argument("--track-low", type=float, default=0.12)
    parser.add_argument("--new-track", type=float, default=0.55)
    parser.add_argument("--max-missed", type=int, default=24)
    parser.add_argument("--detect-every", type=int, default=1, help="Run detector once every N frames")
    parser.add_argument("--adaptive-detect", action="store_true", help="Adapt detector cadence to target FPS")
    parser.add_argument("--target-fps", type=float, default=30.0)
    parser.add_argument("--max-detect-interval", type=int, default=4)
    parser.add_argument("--session-dir", default="", help="Write JSONL telemetry sessions under this directory")
    parser.add_argument("--realesrgan", default="", help="Optional path to official realesrgan-ncnn-vulkan executable")
    parser.add_argument("--record", default="", help="Optional annotated output video path")
    args = parser.parse_args()

    model = Path(args.model)
    if not model.exists():
        raise SystemExit(f"Model not found: {model}")
    try:
        model_digest = verify_sha256(model, args.model_sha256) if args.model_sha256 else sha256_file(model)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    print(f"model sha256: {model_digest}")

    detector = YoloOnnxDetector(model, args.input_size, args.conf, args.iou, prefer_gpu=not args.cpu)
    tracker = MultiObjectTracker(config=TrackerConfig(
        high_conf=args.track_high,
        low_conf=args.track_low,
        new_track_conf=args.new_track,
        max_missed=args.max_missed,
    ))
    scheduler = DetectionScheduler(
        fixed_interval=args.detect_every,
        adaptive=args.adaptive_detect,
        target_fps=args.target_fps,
        max_interval=args.max_detect_interval,
    )
    stabilizer = VideoStabilizer()
    cmc = CameraMotionEstimator()
    masker = TargetMaskCache()
    profiler = RollingProfiler()
    geometry = CameraGeometry(args.hfov)

    try:
        cap = open_capture(args.source)
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc

    state = UiState()
    hud_enabled = True
    enhancement_mode = args.enhance_mode
    stabilization = bool(args.stabilize)
    cmc_enabled = bool(args.cmc)
    mask_enabled = bool(args.mask)
    last_tick = time.perf_counter()
    fps = 0.0
    writer = None
    snapshots = Path("snapshots")
    snapshots.mkdir(exist_ok=True)
    frame_index = 0
    recorder = None
    if args.session_dir:
        recorder = SessionRecorder(args.session_dir, {
            "source": args.source,
            "model": str(model),
            "model_sha256": model_digest,
            "providers": detector.providers,
            "input_size": args.input_size,
            "horizontal_fov_deg": args.hfov,
            "adaptive_detect": args.adaptive_detect,
            "detect_every": args.detect_every,
        })
        print(f"session: {recorder.directory}")

    window = "SpectraTrack"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    def on_mouse(event, x, y, flags, userdata):
        if event != cv2.EVENT_LBUTTONDOWN or x >= state.frame_width:
            return
        hit = None
        for tr in reversed(state.tracks):
            x1, y1, x2, y2 = tr.bbox
            if x1 <= x <= x2 and y1 <= y <= y2:
                hit = tr.track_id
                break
        state.selected_id = None if hit == state.selected_id else hit
        masker.clear()

    cv2.setMouseCallback(window, on_mouse)

    try:
        while True:
            frame_start = time.perf_counter()
            ok, frame = cap.read()
            if not ok:
                break
            frame_index += 1
            state.frame_width = frame.shape[1]

            if stabilization:
                with profiler.measure("stabilize"):
                    base_frame = stabilizer.apply(frame)
            else:
                base_frame = frame

            previous_boxes = [tr.bbox for tr in state.tracks if tr.confirmed]
            if cmc_enabled:
                with profiler.measure("cmc"):
                    motion = cmc.update(base_frame, previous_boxes)
            else:
                motion = MotionEstimate.identity()
                cmc.reset()

            with profiler.measure("enhance"):
                analysis_frame = apply_enhancement(base_frame, enhancement_mode)

            force_detect = not any(tr.confirmed for tr in state.tracks)
            detector_ran = scheduler.should_detect(frame_index, force=force_detect)
            if detector_ran:
                det_start = time.perf_counter()
                detections = detector.detect(analysis_frame)
                det_ms = (time.perf_counter() - det_start) * 1000.0
                profiler.add("detect", det_ms)
                scheduler.observe_detector_ms(det_ms)
            else:
                detections = []

            with profiler.measure("track"):
                tracks = tracker.update(
                    detections,
                    camera_affine=motion.matrix if (cmc_enabled and motion.valid) else None,
                    detector_ran=detector_ran,
                )

            state.tracks = tracks
            if state.selected_id is not None and all(t.track_id != state.selected_id for t in tracks):
                state.selected_id = None
                masker.clear()

            selected = next((t for t in tracks if t.track_id == state.selected_id), None)
            if mask_enabled and selected is not None:
                with profiler.measure("mask"):
                    target_mask = masker.update(analysis_frame, selected, frame_index)
            else:
                target_mask = None
                if not mask_enabled:
                    masker.clear()

            now = time.perf_counter()
            inst = 1.0 / max(now - last_tick, 1e-6)
            fps = inst if fps <= 0 else fps * 0.88 + inst * 0.12
            last_tick = now
            profiler.add("pipeline", (now - frame_start) * 1000.0)
            summary = profiler.summary()

            if hud_enabled:
                output = compose_hud(
                    analysis_frame,
                    tracks,
                    state.selected_id,
                    fps,
                    "+".join(detector.providers),
                    enhancement_mode,
                    metrics=summary,
                    motion=motion,
                    geometry=geometry,
                    cmc_enabled=cmc_enabled,
                    stabilization_enabled=stabilization,
                    detector_interval=scheduler.interval,
                    detector_ran=detector_ran,
                    target_mask=target_mask,
                    mask_enabled=mask_enabled,
                )
            else:
                output = analysis_frame

            if recorder is not None:
                recorder.write_frame(
                    frame_index,
                    now,
                    tracks,
                    state.selected_id,
                    motion.to_dict() if cmc_enabled else None,
                    summary,
                    extra={
                        "detector_ran": detector_ran,
                        "detector_interval": scheduler.interval,
                        "enhancement_mode": enhancement_mode,
                        "mask_enabled": mask_enabled,
                    },
                )

            if args.record:
                if writer is None:
                    Path(args.record).parent.mkdir(parents=True, exist_ok=True)
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    writer = cv2.VideoWriter(args.record, fourcc, max(10.0, fps), (output.shape[1], output.shape[0]))
                writer.write(output)

            cv2.imshow(window, output)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("e"):
                enhancement_mode = next_enhancement_mode(enhancement_mode)
            elif key == ord("h"):
                hud_enabled = not hud_enabled
            elif key == ord("c"):
                cmc_enabled = not cmc_enabled
                cmc.reset()
            elif key == ord("m"):
                mask_enabled = not mask_enabled
                masker.clear()
            elif key == ord("z"):
                stabilization = not stabilization
                stabilizer.reset()
                cmc.reset()
            elif key in (ord("s"), ord("u")) and state.selected_id is not None:
                tr = next((t for t in tracks if t.track_id == state.selected_id), None)
                if tr is not None:
                    crop = crop_with_margin(base_frame, tr.bbox)
                    if crop is not None:
                        stamp = time.strftime("%Y%m%d-%H%M%S")
                        raw_path = snapshots / f"T{tr.track_id:03d}-{stamp}.png"
                        cv2.imwrite(str(raw_path), crop)
                        print(f"saved {raw_path}")
                        if key == ord("u"):
                            if not args.realesrgan:
                                print("AI upscale skipped: pass --realesrgan path\\to\\realesrgan-ncnn-vulkan.exe")
                            else:
                                out_path = raw_path.with_name(raw_path.stem + "-x4.png")
                                try:
                                    run_realesrgan_snapshot(args.realesrgan, raw_path, out_path, 4)
                                    print(f"upscaled {out_path}")
                                except Exception as exc:
                                    print(f"upscale failed: {exc}")
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if recorder is not None:
            recorder.close()
        cv2.destroyAllWindows()

    print("performance:", profiler.summary())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
