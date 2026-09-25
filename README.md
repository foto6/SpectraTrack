# SpectraTrack

Local-only camera object detection, multi-object tracking, target lock, digital target view, visibility enhancement, optical stabilization, and optional snapshot super-resolution for Windows and Android.

This is an engineering-oriented civilian computer-vision project, not a claim of military sensor capability. It operates on ordinary camera pixels and does not invent calibrated range, bearing, thermal data, or sensor measurements that the hardware did not capture.

## What is in v0.3

### Windows / AMD-friendly v0.3

See [pc/README.md](pc/README.md) for the Windows guide, [ARCHITECTURE.md](ARCHITECTURE.md) for the current system map, and [TASKS.md](TASKS.md) for active engineering work.



- webcam or video input;
- YOLOv8/YOLO11-style ONNX detection;
- ONNX Runtime DirectML acceleration with CPU fallback;
- two-stage multi-object stable IDs with tentative/confirmed/predicted lifecycle;
- affine optical-flow camera-motion compensation with RANSAC quality gates;
- detector cadence profiles (quality/balanced/speed) with prediction-only skipped frames;
- click-to-lock target;
- target trajectory;
- enlarged target window;
- non-generative local-contrast / denoise / sharpen mode;
- optical-flow camera stabilization;
- normal / clarity / low-light / edges / explicitly false-color operator views;
- model SHA-256 verification + model provenance manifests;
- headless video processing, class filters, diagnostics and synthetic benchmarks;
- calibrated angular offset/size display without fabricating monocular metric range;
- JSONL session logging + post-run stability/performance reports;
- target snapshots;
- optional Real-ESRGAN ncnn/Vulkan x4 snapshot hook;
- synthetic visual demo that needs no model;
- selected-target sparse optical-flow refinement between detector observations;
- multi-video batch analysis with conservative cross-video track graphs;
- best-tracklet preview crops and a local HTML review report;
- exportable SAME / DIFFERENT / UNSURE review decisions;
- richer spatial HSV/grayscale/edge descriptor galleries for cross-video candidates;
- generated-video exclusion to avoid batch feedback loops;
- one-click Windows launchers for one video or a whole video folder;
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

Real-time detection/tracking and neural super-resolution compete for the same compute budget. Running a heavy generative upscaler on every frame increases latency and can make tracking worse. SpectraTrack keeps the real-time path responsive and uses AI SR only on a selected PC snapshot.

## Parallel agent development

Before changing code, AI agents and human contributors should read:

- [AGENTS.md](AGENTS.md) — branch, testing, coordination, and scope rules;
- [ARCHITECTURE.md](ARCHITECTURE.md) — canonical current architecture and code ownership map;
- [TASKS.md](TASKS.md) — completed, partial, known-problem, debt, and next-work lists;
- [DECISIONS.md](DECISIONS.md) — architectural decisions and their reasons.

The intended multi-agent flow after the setup branch is merged is:

```text
main
 ↑
integration
 ↑
├── agent/detection
├── agent/tracking
├── agent/enhancement
├── agent/performance
└── agent/qa
```

Agents should not commit directly to `main`.

Current highest-priority PC problem: high-recall detection of small people in high-angle, night-time, compressed video. See [docs/PC_V03_PLAN.md](docs/PC_V03_PLAN.md).

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

PC v0.3 is the active development target. Windows CI compiles/tests the package,
runs a synthetic benchmark and diagnostics, builds a standalone PyInstaller artifact,
and smoke-tests the generated EXE. Android remains maintained separately and is not
part of the current PC v0.3 feature work.



- PC core: syntax checked; tracker unit tests pass.
- Android: source tree prepared for AGP 9.4 / SDK 36 and intended to be compiled in CI/Android Studio.
- Neural weights are intentionally excluded from Git.
