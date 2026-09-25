# SpectraTrack v0.2.0

First packaged PC + Android release.

## Downloads

- **SpectraTrack-Windows-x64.zip** — ready-to-run Windows x64 build. No Python installation required.
- **SpectraTrack-Android.apk** — Android sideload build.
- **SHA256SUMS.txt** — SHA-256 checksums for the release binaries.
- GitHub also provides source ZIP/TAR automatically.

## PC highlights

- YOLO ONNX inference with DirectML preference for AMD GPUs and CPU fallback.
- Two-stage multi-object tracking with stable IDs and short-loss prediction.
- Affine camera-motion compensation using optical flow + RANSAC quality gates.
- Quality / balanced / speed inference profiles.
- Non-biometric appearance cue for same-class association.
- Target lock, trajectory, crop preview and target telemetry.
- Normal / clarity / low-light / edges / explicitly false-color display modes.
- Camera calibration support for angular offset and angular size.
- Robust webcam capture with bounded reconnects and telemetry.
- JSONL session logging, CSV export, reports and replay overlays.
- Model SHA-256 verification and provenance manifests.
- Snapshot provenance sidecars.
- Standalone Windows executable is smoke-tested in CI.

## Android

- CameraX live camera pipeline.
- ONNX Runtime local inference.
- Local-only object tracking and tap-to-lock HUD.
- Compatible ONNX model can be imported from phone storage.
- CAMERA permission only; no INTERNET permission.

## Model weights

Neural-network weights are intentionally **not bundled** in this release. Use a compatible fixed-size YOLO ONNX model such as a locally exported 640x640 YOLO11/YOLOv8 model. This keeps model provenance and licensing separate from the application.

For Windows, put the ONNX file next to the executable and run:

```powershell
.\SpectraTrack-PC.exe --model .\yolo11n.onnx --source .\video.mp4 --profile balanced --record .\result.mp4
```

For Android, install the APK, open SpectraTrack, press **IMPORT MODEL**, and select the compatible ONNX file.

## Data semantics

Pseudo-thermal is false-color RGB visualization, not thermal sensing. Metric range is not fabricated from a single ordinary RGB camera. AI super-resolution is marked as enhanced output and may invent fine detail.
