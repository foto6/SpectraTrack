from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import cv2

from .calibration import CameraCalibration
from .detector import YoloOnnxDetector
from .enhance import crop_with_margin, enhance_visibility, run_realesrgan_snapshot
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
    parser.add_argument("--input-size", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--classes", default="", help="Comma-separated labels to retain, e.g. person,car,truck")
    parser.add_argument("--cpu", action="store_true", help="Disable DirectML preference")
    parser.add_argument("--enhance", action="store_true", help="Start with visibility enhancement enabled")
    parser.add_argument("--stabilize", action="store_true", help="Start with optical stabilization enabled")
    parser.add_argument("--no-cmc", action="store_true", help="Disable camera-motion compensation for tracking")
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
    state = UiState()
    hud_enabled = True
    enhancement = bool(args.enhance)
    stabilization = bool(args.stabilize)
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
    }) if args.session_log else None

    snapshots = Path("snapshots")
    snapshots.mkdir(exist_ok=True)
    window = "SpectraTrack"
    frame_index = 0

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
                input_frame = enhance_visibility(frame) if enhancement else frame

            with timings.measure("detect"):
                detections = detector.detect(input_frame)
                if class_filter:
                    detections = [d for d in detections if d.label.lower() in class_filter]

            # Stabilization already compensates image motion; do not compensate twice.
            camera_shift = (0.0, 0.0)
            camera_transform = None
            if not stabilization and cam is not None and cam.valid:
                camera_shift = (cam.dx, cam.dy)
                camera_transform = cam.affine

            with timings.measure("track"):
                tracks = tracker.update(
                    detections,
                    camera_motion=camera_shift,
                    camera_transform=camera_transform,
                )

            state.tracks = tracks
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
            }

            if hud_enabled:
                with timings.measure("hud"):
                    output = compose_hud(
                        input_frame, tracks, state.selected_id, fps,
                        "+".join(detector.providers), enhancement,
                        timings_ms=timing_snapshot,
                        camera_motion=camera_motion_info,
                        calibration=calibration,
                    )
            else:
                output = input_frame

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
        f"processed_frames={frame_index} model_sha256={model_hash} "
        f"detect_avg_ms={timings.average('detect'):.2f} track_avg_ms={timings.average('track'):.2f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
