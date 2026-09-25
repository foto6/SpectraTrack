# SpectraTrack

Local-only camera object detection, multi-object tracking, target lock, digital target view, visibility enhancement, optical stabilization, and optional snapshot super-resolution for Windows and Android.

This is an engineering-oriented civilian computer-vision project, not a claim of military sensor capability. It operates on ordinary camera pixels and does not invent calibrated range, bearing, thermal data, or sensor measurements that the hardware did not capture.

## What is in v0.1

### Windows / AMD-friendly

- webcam or video input;
- YOLOv8/YOLO11-style ONNX detection;
- ONNX Runtime DirectML acceleration with CPU fallback;
- multi-object stable IDs with short-loss prediction;
- click-to-lock target;
- target trajectory;
- enlarged target window;
- non-generative local-contrast / denoise / sharpen mode;
- optical-flow camera stabilization;
- target snapshots;
- optional Real-ESRGAN ncnn/Vulkan x4 snapshot hook;
- synthetic visual demo that needs no model;
- unit tests.

### Android / Samsung

- native Kotlin application;
- CameraX 1.6.2;
- ONNX Runtime Android 1.30.0;
- NNAPI attempt + CPU fallback;
- same YOLO output convention as PC;
- local multi-object tracking;
- tap-to-lock HUD;
- target thumbnail and motion/status panel;
- no server, login, telemetry, or cloud processing.

## Fastest route

### PC

```powershell
cd pc
.\run_demo.bat
```

That verifies the HUD and tracking without downloading any model.

To use the real detector, first follow `models/README.md`, then:

```powershell
cd pc
.\run_camera.bat
```

### Samsung / Android

1. Generate `yolo11n.onnx` with `tools/export_yolo.py`.
2. Confirm the file exists at `android/app/src/main/assets/yolo11n.onnx`.
3. Open `android/` in Android Studio.
4. Build/install `app`.
5. Grant camera permission.
6. Tap a box to lock it.

See `android/README.md` for performance tuning.

## Why it is split this way

Real-time detection/tracking and neural super-resolution compete for the same compute budget. Running a heavy generative upscaler on every frame increases latency and can make tracking worse. SpectraTrack keeps the real-time path responsive and uses AI SR only on a selected PC snapshot in v0.1.

## Privacy / security

Read `SECURITY.md`. Runtime clients contain no networking code. They do not auto-download models or executables.

## Repository layout

```text
pc/        Windows/Python application and tests
android/   Native CameraX Android application
models/    Model format/setup notes (weights are gitignored)
tools/     Model export and hashing utilities
docs/      Architecture and test plan
```

## Current status

- PC core: syntax checked; tracker unit tests pass.
- Android: source tree prepared for AGP 9.4 / SDK 36 and intended to be compiled in CI/Android Studio.
- Neural weights are intentionally excluded from Git.
