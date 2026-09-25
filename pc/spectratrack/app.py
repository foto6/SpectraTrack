from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import cv2

from .detector import YoloOnnxDetector
from .enhance import crop_with_margin, enhance_visibility, run_realesrgan_snapshot
from .hud import compose_hud
from .tracker import MultiObjectTracker
from .stabilize import VideoStabilizer


class UiState:
    def __init__(self) -> None:
        self.tracks = []
        self.selected_id: int | None = None
        self.frame_width = 0


def parse_source(text: str):
    return int(text) if text.isdigit() else text


def main() -> int:
    parser = argparse.ArgumentParser(description="SpectraTrack local object detection/tracking HUD")
    parser.add_argument("--model", required=True, help="Path to YOLOv8/YOLO11-style ONNX model")
    parser.add_argument("--source", default="0", help="Camera index or video path")
    parser.add_argument("--input-size", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--cpu", action="store_true", help="Disable DirectML preference")
    parser.add_argument("--enhance", action="store_true", help="Start with visibility enhancement enabled")
    parser.add_argument("--stabilize", action="store_true", help="Start with optical stabilization enabled")
    parser.add_argument("--realesrgan", default="", help="Optional path to official realesrgan-ncnn-vulkan executable")
    parser.add_argument("--record", default="", help="Optional output video path")
    args = parser.parse_args()

    model = Path(args.model)
    if not model.exists():
        raise SystemExit(f"Model not found: {model}")

    detector = YoloOnnxDetector(model, args.input_size, args.conf, args.iou, prefer_gpu=not args.cpu)
    tracker = MultiObjectTracker()
    stabilizer = VideoStabilizer()

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
    snapshots = Path("snapshots")
    snapshots.mkdir(exist_ok=True)

    window = "SpectraTrack"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    def on_mouse(event, x, y, flags, userdata):
        if event != cv2.EVENT_LBUTTONDOWN or x >= state.frame_width:
            return
        hit = None
        for tr in state.tracks:
            x1, y1, x2, y2 = tr.bbox
            if x1 <= x <= x2 and y1 <= y <= y2:
                hit = tr.track_id
                break
        state.selected_id = None if hit == state.selected_id else hit

    cv2.setMouseCallback(window, on_mouse)

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            state.frame_width = frame.shape[1]
            if stabilization:
                frame = stabilizer.apply(frame)
            input_frame = enhance_visibility(frame) if enhancement else frame
            detections = detector.detect(input_frame)
            tracks = tracker.update(detections)
            state.tracks = tracks
            if state.selected_id is not None and all(t.track_id != state.selected_id for t in tracks):
                state.selected_id = None

            now = time.perf_counter()
            inst = 1.0 / max(now - last_tick, 1e-6)
            fps = inst if fps <= 0 else fps * 0.88 + inst * 0.12
            last_tick = now

            if hud_enabled:
                output = compose_hud(input_frame, tracks, state.selected_id, fps, "+".join(detector.providers), enhancement)
            else:
                output = input_frame

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
                enhancement = not enhancement
            elif key == ord("h"):
                hud_enabled = not hud_enabled
            elif key == ord("z"):
                stabilization = not stabilization
                stabilizer.reset()
            elif key in (ord("s"), ord("u")) and state.selected_id is not None:
                tr = next((t for t in tracks if t.track_id == state.selected_id), None)
                if tr is not None:
                    crop = crop_with_margin(frame, tr.bbox)
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
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
