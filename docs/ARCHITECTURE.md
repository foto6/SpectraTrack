# Architecture

## Shared pipeline

`camera/video -> resize/letterbox -> ONNX detector -> class-aware motion tracker -> target lock -> HUD`

The two clients deliberately use the same detector output convention and very similar tracker logic, which makes behavior easier to compare.

## Windows client

- OpenCV: capture, drawing, local contrast enhancement, optical-flow stabilization.
- ONNX Runtime DirectML: detector inference on DirectX 12 hardware, including AMD GPUs.
- CPU fallback: used automatically if DirectML is unavailable or `--cpu` is passed.
- Local tracker: velocity prediction + IoU + center-distance gating.
- Target view: click a detected object to lock it and show a zoomed crop.
- Real-ESRGAN hook: selected-target snapshots can be handed to a separately installed ncnn/Vulkan executable.

Why SR is snapshot-only by default: doing a heavy neural upscale on every live frame wastes GPU budget and increases latency. Detection/tracking should remain responsive; SR is more useful on a selected crop.

## Android client

- CameraX Preview + ImageAnalysis with `STRATEGY_KEEP_ONLY_LATEST`.
- ONNX Runtime Android.
- NNAPI is attempted first and CPU is the fallback. NNAPI is not guaranteed to be faster for every model because unsupported graph fragments can cause expensive CPU/accelerator hand-offs.
- A lightweight tracker mirrors the desktop matching strategy.
- `HudOverlayView` maps detector coordinates over a FIT_CENTER camera preview and supports tap-to-lock.

## What is intentionally not implemented

- biometric identification;
- cross-camera person re-identification;
- remote camera access;
- covert/background capture;
- weapons targeting or fire-control integration;
- fabricated range/bearing values without calibrated sensors.

If GPS/IMU is added later, the HUD should distinguish **measured sensor data** from values derived only from image pixels.
