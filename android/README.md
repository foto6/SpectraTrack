# Android client

The Android client is a native CameraX + Kotlin application. It does not need Python or Pydroid.

## Requirements

- Android Studio compatible with Android Gradle Plugin 9.4.
- JDK 17.
- Android SDK 36.
- Gradle 9.6.0 (Android Studio can provision the matching Gradle version).
- A fixed-size YOLO11/YOLOv8 COCO ONNX model. You can either bundle it at `app/src/main/assets/yolo11n.onnx` before building, or import it from phone storage at first launch.

CameraX is pinned to 1.6.2 and ONNX Runtime Android to 1.30.0 in this snapshot.

## Build

Open the `android` directory in Android Studio, allow Gradle sync, connect the phone with USB debugging enabled, and run the `app` configuration.

Command line, if Gradle 9.6 is installed:

```powershell
gradle :app:assembleDebug
```

APK:

`app/build/outputs/apk/debug/app-debug.apk`

## Runtime behavior

- Back camera opens through CameraX.
- The analyzer keeps only the latest frame, avoiding an ever-growing latency queue.
- The image is rotated to display orientation before inference.
- ONNX Runtime tries NNAPI and falls back to CPU.
- Tap a tracked box to lock/unlock it.
- The target panel shows a crop, confidence, motion in image pixels/frame, age, and missed-frame count.

## Performance tuning on Samsung

Start with `yolo11n` at 640. If inference time is too high, export at 512 or 416 and change `inputSize` in `OnnxDetector.kt` to match. A smaller model often gives a better live experience than forcing a heavyweight network and dropping most analysis frames.

NNAPI is not automatically fastest. If the HUD shows high latency, test CPU by temporarily commenting out `options.addNnapi()` in `OnnxDetector.kt` and compare the displayed inference milliseconds.

## AI super-resolution on Android

This first build deliberately does **not** run generative SR on every camera frame. A live detector/tracker should get the compute budget first. The target crop is currently high-quality scaled for display; a native ncnn/Vulkan SR module can be added as an on-demand snapshot action later without redesigning the detector or tracker.
