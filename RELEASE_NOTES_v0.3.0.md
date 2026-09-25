# SpectraTrack v0.3.0

PC-focused release with multi-video analysis, safer target-lock refinement, and substantially stronger regression coverage. Android is rebuilt and version-aligned as 0.3.0.

## Windows: easiest usage

Put a compatible fixed-size YOLO ONNX model next to the executable. The launchers search for:

1. `model.onnx`
2. `yolo11x.onnx`
3. the first `*.onnx` in the application folder

Then:

- drag one video onto **PROCESS_VIDEO.bat** for a high-quality single-video pass;
- drag a folder onto **ANALYZE_VIDEO_FOLDER.bat** for cross-video analysis;
- double-click **START_SPECTRATRACK.bat** for camera mode.

## Cross-video graph

The folder analyzer now:

- builds local tracklets per video;
- uses affine camera-motion compensation while tracking;
- collects a small gallery of spatial HSV / grayscale / edge-orientation descriptors;
- saves one best preview crop per retained tracklet;
- compares only compatible detector classes;
- uses conservative complete-link grouping to avoid weak transitive A≈B≈C merges;
- creates a local HTML report with entity branches and review candidates;
- supports **SAME / DIFFERENT / UNSURE** decisions;
- exports `spectratrack_review.json` and can apply it on a later run;
- skips SpectraTrack-generated output videos by default to prevent recursive self-analysis;
- uses relative paths as video IDs, so duplicate filenames in subfolders do not collide.

For people, links mean **similar visible appearance in this batch only**. They are not face recognition, biometric identity, or named-person identification.

For vehicles, animals, and other objects, links are visual same-object candidates and can still be wrong. Ambiguous links are intentionally reviewable.

## Tracking / lock improvements

- Selected-target sparse optical-flow refinement is integrated into the live pipeline.
- Flow is re-anchored on fresh detector observations and used between detections or short misses.
- The refiner transforms the previous target box, avoiding double-applying motion to an already-current detector box.
- HUD/session logging expose lock-flow status.

## Reliability fixes

- `--conf 0.20` can no longer be misparsed as `--config 0.20`.
- Recorded processed video preserves source FPS.
- Video files ignore camera-only DirectShow/MSMF backend flags.
- Runtime presets and all one-click launchers are included in packaged Windows builds.
- Batch validates model/review paths and thresholds before processing.
- Batch prints periodic frame/track progress.
- A full-video batch failure now returns an error instead of a misleading empty success.
- Local HTML review still works when browser `file://` localStorage is unavailable.
- Recursive folders with identical basenames use collision-safe relative video IDs.

## Verification

The release workflow reruns:

- Ruff;
- Python compile checks;
- full pytest suite, including an end-to-end synthetic AVI → capture → CMC → tracker → cross-video descriptor → preview test;
- synthetic tracker benchmark;
- diagnostics and self-check;
- PyInstaller Windows build;
- standalone `SpectraTrack-PC.exe --help`;
- standalone `SpectraTrack-PC.exe batch --help`;
- Android SDK/Gradle APK build.

## Android

- APK metadata is aligned to `versionName 0.3.0`, `versionCode 3`.
- CameraX / ONNX Runtime local camera pipeline remains included.
- No INTERNET permission is added.

## Model weights

Model weights are not bundled. Use a compatible fixed-size YOLOv8/YOLO11-style ONNX model.

## Data semantics

- Pseudo-thermal remains false-color RGB visualization, not thermal sensing.
- SpectraTrack does not fabricate metric target range from an ordinary monocular RGB camera.
- Optional neural super-resolution output may invent fine detail and is marked as AI-enhanced.
