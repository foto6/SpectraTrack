from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import cv2

from .appearance import attach_appearance
from .calibration import CameraCalibration
from .detector import YoloOnnxDetector
from .enhance import DISPLAY_MODES, apply_display_mode, crop_with_margin, enhance_visibility, run_realesrgan_snapshot
from .hud import compose_hud
from .integrity import sha256_file, verify_sha256
from .metrics import StageTimer
from .model_manifest import ModelManifest
from .motion import GlobalMotionEstimator
from .session import SessionRecorder
from .stabilize import VideoStabilizer
from .tracker import MultiObjectTracker


class UiState:
    def __init__(self) -> None:
        self.tracks = []
        self.selected_id: int | None = None
        self.frame_width = 0


def parse_source(text: str):
    return int(text) if text.isdigit() else text


def parse_class_filter(text: str) -> set[str]:
    return {part.strip().lower() for part in text.split(",") if part.strip()}


def main() -> int:
    parser = argparse.ArgumentParser(description="SpectraTrack PC v0.2 local object detection/tracking HUD")
    parser.add_argument("--model", required=True, help="Path to YOLOv8/YOLO11-style ONNX model")
    parser.add_argument("--model-sha256", default="", help="Expected model SHA-256; abort on mismatch")
    parser.add_argument("--model-manifest", default="", help="Optional JSON model manifest with hash/provenance/input size")
    parser.add_argument("--source", default="0", help="Camera index or video path")
    parser.add_argument("--camera-width", type=int, default=0, help="Requested webcam width; camera sources only")
    parser.add_argument("--camera-height", type=int, default=0, help="Requested webcam height; camera sources only")
    parser.add_argument("--camera-fps", type=float, default=0.0, help="Requested webcam FPS; camera sources only")
    parser.add_argument("--input-size", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--classes", default="", help="Comma-separated labels to retain, e.g. person,car,truck")
    parser.add_argument("--profile", choices=("quality", "balanced", "speed"), default="balanced")
    parser.add_argument("--detect-every", type=int, default=0, help="Run detector every N frames; 0 uses profile default")
    parser.add_argument("--cpu", action="store_true", help="Disable DirectML preference")
    parser.add_argument("--enhance", action="store_true", help="Enhance the detector analysis image with non-generative clarity processing")
    parser.add_argument("--view", choices=DISPLAY_MODES, default="normal", help="Operator display mode; pseudo-thermal is false-color only")
    parser.add_argument("--stabilize", action="store_true", help="Start with optical stabilization enabled")
    parser.add_argument("--no-cmc", action="store_true", help="Disable camera-motion compensation for tracking")
    parser.add_argument("--no-appearance", action="store_true", help="Disable non-biometric color appearance cue used for same-class association")
    parser.add_argument("--calibration", default="", help="Optional camera calibration JSON with width/height/HFOV")
    parser.add_argument("--session-log", default="", help="Optional JSONL metadata/session log")
    parser.add_argument("--realesrgan", default="", help="Optional path to official realesrgan-ncnn-vulkan executable")
    parser.add_argument("--record", default="", help="Optional output video path")
    parser.add_argument("--headless", action="store_true", help="Do not create an OpenCV window; useful for batch video processing")
    parser.add_argument("--max-frames", type=int, default=0, help="Stop after N processed frames; 0 means unlimited")
    args = parser.parse_args()

    model = Path(args.model)
    if not model.exists():
        raise SystemExit(f"Model not found: {model}")

    manifest = ModelManifest.load(args.model_manifest) if args.model_manifest else None
    if manifest is not None:
        model_hash = manifest.verify(model)
        if args.input_size == 640:
            args.input_size = manifest.input_size
        elif args.input_size != manifest.input_size:
            raise SystemExit(
                f"--input-size {args.input_size} conflicts with manifest input_size {manifest.input_size}"
            )
        if args.model_sha256:
            verify_sha256(model, args.model_sha256)
    else:
        model_hash = verify_sha256(model, args.model_sha256) if args.model_sha256 else sha256_file(model)

    calibration = CameraCalibration.from_json(args.calibration) if args.calibration else None
    class_filter = parse_class_filter(args.classes)
    profile_detect_every = {"quality": 1, "balanced": 2, "speed": 3}[args.profile]
    detect_every = args.detect_every if args.detect_every > 0 else profile_detect_every

    detector = YoloOnnxDetector(model, args.input_size, args.conf, args.iou, prefer_gpu=not args.cpu)
    tracker = MultiObjectTracker()
    stabilizer = VideoStabilizer()
    motion = GlobalMotionEstimator()
    timings = StageTimer()

    source = parse_source(args.source)
    if os.name == "nt" and isinstance(source, int):
        cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)
    else:
        cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open source: {args.source}")

    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if isinstance(source, int):
        if args.camera_width > 0:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.camera_width)
        if args.camera_height > 0:
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.camera_height)
        if args.camera_fps > 0:
            cap.set(cv2.CAP_PROP_FPS, args.camera_fps)

    capture_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    capture_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    capture_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    requested_capture = {
        "width": args.camera_width or None,
        "height": args.camera_height or None,
        "fps": args.camera_fps or None,
    }
    actual_capture = {
        "width": capture_width,
        "height": capture_height,
        "fps": round(capture_fps, 3),
    }
    if isinstance(source, int) and any(v is not None for v in requested_capture.values()):
        print(f"camera_requested={requested_capture}")
        print(f"camera_actual={actual_capture}")
    if calibration is not None and capture_width > 0 and capture_height > 0:
        if (calibration.width, calibration.height) != (capture_width, capture_height):
            print(
                "WARNING calibration resolution differs from capture: "
                f"cal={calibration.width}x{calibration.height} capture={capture_width}x{capture_height}. "
                "Angular mapping is scaled, but recalibration is recommended if crop/zoom/FOV changed."
            )

    state = UiState()
    hud_enabled = True
    enhancement = bool(args.enhance)
    stabilization = bool(args.stabilize)
    view_mode = args.view
    last_tick = time.perf_counter()
    fps = 0.0
    writer = None
    recorder = SessionRecorder(args.session_log, {
        "model": str(model),
        "model_sha256": model_hash,
        "model_manifest": args.model_manifest or None,
        "model_name": manifest.name if manifest else None,
        "model_source": manifest.source if manifest else None,
        "providers": detector.providers,
        "source": str(args.source),
        "class_filter": sorted(class_filter),
        "calibration": args.calibration or None,
        "headless": bool(args.headless),
        "capture_requested": requested_capture,
        "capture_actual": actual_capture,
        "calibration_resolution_match": (
            calibration is None
            or capture_width <= 0
            or capture_height <= 0
            or (calibration.width == capture_width and calibration.height == capture_height)
        ),
        "view_mode": view_mode,
        "analysis_enhance": enhancement,
        "profile": args.profile,
        "detect_every": detect_every,
        "appearance_cue": not args.no_appearance,
    }) if args.session_log else None

    snapshots = Path("snapshots")
    snapshots.mkdir(exist_ok=True)
    window = "SpectraTrack"
    frame_index = 0
    previous_track_state: dict[int, tuple[bool, int, str]] = {}

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
        if recorder:
            recorder.event("target_lock", selected_id=state.selected_id)

    if not args.headless:
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(window, on_mouse)

    try:
        while True:
            ok, raw_frame = cap.read()
            if not ok:
                break
            frame_index += 1
            state.frame_width = raw_frame.shape[1]

            with timings.measure("camera_motion"):
                cam = motion.update(raw_frame) if not args.no_cmc else None

            frame = raw_frame
            if stabilization:
                with timings.measure("stabilize"):
                    frame = stabilizer.apply(frame)

            with timings.measure("enhance"):
                analysis_frame = enhance_visibility(frame) if enhancement else frame
                display_frame = apply_display_mode(frame, view_mode)

            should_detect = ((frame_index - 1) % detect_every) == 0
            detections = []
            if should_detect:
                with timings.measure("detect"):
                    detections = detector.detect(analysis_frame)
                    if class_filter:
                        detections = [d for d in detections if d.label.lower() in class_filter]
                    if not args.no_appearance:
                        attach_appearance(frame, detections)
            else:
                timings.add("detect", 0.0)

            # Stabilization already compensates image motion; do not compensate twice.
            camera_shift = (0.0, 0.0)
            camera_transform = None
            if not stabilization and cam is not None and cam.valid:
                camera_shift = (cam.dx, cam.dy)
                camera_transform = cam.affine

            with timings.measure("track"):
                if should_detect:
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

            state.tracks = tracks

            if recorder:
                current_ids = {t.track_id for t in tracks}
                previous_ids = set(previous_track_state)
                for tr in tracks:
                    prev = previous_track_state.get(tr.track_id)
                    if prev is None:
                        recorder.event("track_created", track_id=tr.track_id, label=tr.label, confirmed=tr.confirmed)
                    else:
                        prev_confirmed, prev_missed, _ = prev
                        if not prev_confirmed and tr.confirmed:
                            recorder.event("track_confirmed", track_id=tr.track_id, label=tr.label)
                        if prev_missed > 0 and tr.missed == 0:
                            recorder.event("track_recovered", track_id=tr.track_id, label=tr.label, recoveries=tr.recoveries)
                for ended_id in previous_ids - current_ids:
                    prev_confirmed, prev_missed, prev_label = previous_track_state[ended_id]
                    recorder.event("track_ended", track_id=ended_id, label=prev_label, confirmed=prev_confirmed, last_missed=prev_missed)
                previous_track_state = {
                    t.track_id: (t.confirmed, t.missed, t.label)
                    for t in tracks
                }

            if state.selected_id is not None and all(t.track_id != state.selected_id for t in tracks):
                if recorder:
                    recorder.event("target_lost", selected_id=state.selected_id)
                state.selected_id = None

            now = time.perf_counter()
            inst = 1.0 / max(now - last_tick, 1e-6)
            fps = inst if fps <= 0 else fps * 0.88 + inst * 0.12
            last_tick = now

            camera_motion_info = None
            if cam is not None and cam.valid:
                camera_motion_info = (cam.dx, cam.dy, cam.rotation_rad)

            timing_snapshot = {
                "detect": round(timings.latest("detect"), 3),
                "track": round(timings.latest("track"), 3),
                "cmc": round(timings.latest("camera_motion"), 3),
                "enhance": round(timings.latest("enhance"), 3),
                "stabilize": round(timings.latest("stabilize"), 3),
                "hud": round(timings.latest("hud"), 3),
                "detector_ran": 1.0 if should_detect else 0.0,
            }

            if hud_enabled:
                with timings.measure("hud"):
                    output = compose_hud(
                        display_frame, tracks, state.selected_id, fps,
                        "+".join(detector.providers), enhancement,
                        timings_ms=timing_snapshot,
                        camera_motion=camera_motion_info,
                        calibration=calibration,
                        view_mode=view_mode,
                    )
            else:
                output = display_frame

            if recorder:
                recorder.frame(
                    frame_index, tracks, state.selected_id, fps,
                    timings_ms=timing_snapshot,
                    camera_motion=camera_motion_info,
                )

            if args.record:
                if writer is None:
                    Path(args.record).parent.mkdir(parents=True, exist_ok=True)
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    writer = cv2.VideoWriter(args.record, fourcc, max(10.0, fps), (output.shape[1], output.shape[0]))
                    if not writer.isOpened():
                        raise RuntimeError(f"Cannot create video writer: {args.record}")
                writer.write(output)

            if not args.headless:
                cv2.imshow(window, output)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                if key == ord("e"):
                    enhancement = not enhancement
                    if recorder:
                        recorder.event("enhance", enabled=enhancement)
                elif key == ord("h"):
                    hud_enabled = not hud_enabled
                elif key == ord("m"):
                    view_mode = DISPLAY_MODES[(DISPLAY_MODES.index(view_mode) + 1) % len(DISPLAY_MODES)]
                    if recorder:
                        recorder.event("view_mode", mode=view_mode)
                elif key == ord("z"):
                    stabilization = not stabilization
                    stabilizer.reset()
                    motion.reset()
                    tracker.reset()
                    state.selected_id = None
                    if recorder:
                        recorder.event("stabilize", enabled=stabilization)
                elif key == ord("r"):
                    tracker.reset()
                    state.selected_id = None
                    if recorder:
                        recorder.event("tracker_reset")
                elif key in (ord("s"), ord("u")) and state.selected_id is not None:
                    tr = next((t for t in tracks if t.track_id == state.selected_id), None)
                    if tr is not None:
                        crop = crop_with_margin(frame, tr.bbox)
                        if crop is not None:
                            stamp = time.strftime("%Y%m%d-%H%M%S")
                            raw_path = snapshots / f"T{tr.track_id:03d}-{stamp}.png"
                            cv2.imwrite(str(raw_path), crop)
                            print(f"saved {raw_path}")
                            if recorder:
                                recorder.event("snapshot", track_id=tr.track_id, path=str(raw_path))
                            if key == ord("u"):
                                if not args.realesrgan:
                                    print("AI upscale skipped: pass --realesrgan path\\to\\realesrgan-ncnn-vulkan.exe")
                                else:
                                    out_path = raw_path.with_name(raw_path.stem + "-x4.png")
                                    try:
                                        run_realesrgan_snapshot(args.realesrgan, raw_path, out_path, 4)
                                        print(f"upscaled {out_path}")
                                        if recorder:
                                            recorder.event("ai_upscale", track_id=tr.track_id, path=str(out_path))
                                    except Exception as exc:
                                        print(f"upscale failed: {exc}")

            if args.max_frames > 0 and frame_index >= args.max_frames:
                break
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if recorder is not None:
            recorder.close()
        if not args.headless:
            cv2.destroyAllWindows()

    print(
        f"processed_frames={frame_index} capture={capture_width}x{capture_height}@{capture_fps:.2f} "
        f"profile={args.profile} detect_every={detect_every} model_sha256={model_hash} "
        f"detect_avg_ms={timings.average('detect'):.2f} track_avg_ms={timings.average('track'):.2f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
