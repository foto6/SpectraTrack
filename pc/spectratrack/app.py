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
from .metrics import RollingProfiler
from .recording import SessionRecorder
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
    parser.add_argument("--source", default="0", help="Camera index or video path")
    parser.add_argument("--input-size", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--cpu", action="store_true", help="Disable DirectML preference")
    parser.add_argument("--enhance-mode", choices=ENHANCE_MODES, default="off")
    parser.add_argument("--stabilize", action="store_true", help="Start with optical stabilization enabled")
    parser.add_argument("--cmc", action=argparse.BooleanOptionalAction, default=True, help="Camera-motion compensation")
    parser.add_argument("--hfov", type=float, default=None, help="Calibrated/known horizontal camera FOV in degrees")
    parser.add_argument("--track-high", type=float, default=0.45)
    parser.add_argument("--track-low", type=float, default=0.12)
    parser.add_argument("--new-track", type=float, default=0.55)
    parser.add_argument("--max-missed", type=int, default=24)
    parser.add_argument("--session-dir", default="", help="Write JSONL telemetry sessions under this directory")
    parser.add_argument("--realesrgan", default="", help="Optional path to official realesrgan-ncnn-vulkan executable")
    parser.add_argument("--record", default="", help="Optional annotated output video path")
    args = parser.parse_args()

    model = Path(args.model)
    if not model.exists():
        raise SystemExit(f"Model not found: {model}")

    detector = YoloOnnxDetector(model, args.input_size, args.conf, args.iou, prefer_gpu=not args.cpu)
    tracker = MultiObjectTracker(config=TrackerConfig(
        high_conf=args.track_high,
        low_conf=args.track_low,
        new_track_conf=args.new_track,
        max_missed=args.max_missed,
    ))
    stabilizer = VideoStabilizer()
    cmc = CameraMotionEstimator()
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
            "providers": detector.providers,
            "input_size": args.input_size,
            "horizontal_fov_deg": args.hfov,
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

            with profiler.measure("detect"):
                detections = detector.detect(analysis_frame)

            with profiler.measure("track"):
                tracks = tracker.update(
                    detections,
                    camera_affine=motion.matrix if (cmc_enabled and motion.valid) else None,
                )

            state.tracks = tracks
            if state.selected_id is not None and all(t.track_id != state.selected_id for t in tracks):
                state.selected_id = None

            now = time.perf_counter()
            inst = 1.0 / max(now - last_tick, 1e-6)
            fps = inst if fps <= 0 else fps * 0.88 + inst * 0.12
            last_tick = now
            profiler.add("frame", (now - frame_start) * 1000.0)
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
