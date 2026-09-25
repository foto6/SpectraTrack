# SpectraTrack v0.2.1

Hotfix release for video-file processing.

## Fixed
- Processed video recording now preserves the source video's FPS instead of using processing throughput. A 16-second input therefore keeps the expected duration in the rendered result.

## Assets
- **SpectraTrack-Windows-x64.zip** — standalone Windows x64 build, no Python required for inference.
- **SpectraTrack-Android.apk** — Android sideload build.
- **SHA256SUMS.txt** — checksums.

Model weights remain separate. Use a compatible fixed-size YOLO ONNX model.
