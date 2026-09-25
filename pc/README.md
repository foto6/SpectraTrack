# SpectraTrack PC v0.2

Windows-first local computer-vision HUD. The real-time path is deliberately non-cloud and does not perform biometric identification.

## What changed from v0.1

- 8-state Kalman box prediction.
- Two-stage high/low-confidence association inspired by ByteTrack's core idea.
- Tentative / LOCK / COAST track states.
- Sparse optical-flow camera-motion compensation (CMC) before association.
- Runtime performance counters.
- JSONL session telemetry for reproducible analysis.
- Headless benchmark command.
- Optional calibrated horizontal FOV for relative angle and angular-rate estimates.
- Four live enhancement modes: off, visibility, lowlight, detail.
- Existing DirectML preference remains suitable for AMD GPUs on Windows.

No ReID claim is made in v0.2. Real ReID needs a separate appearance embedding model and will be added only with an explicit model contract and provenance.

## Install

Use 64-bit Python 3.12.

~~~
cd pc
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
pip install -r requirements-win.txt
~~~

## Test without a model

~~~
python -m spectratrack.demo
~~~

## Run camera

Put a compatible YOLO ONNX at ../models/yolo11n.onnx, then:

~~~
python -m spectratrack.app --model ..\models\yolo11n.onnx --source 0 --cmc
~~~

With optical stabilization and a known/verified 70 degree horizontal FOV:

~~~
python -m spectratrack.app --model ..\models\yolo11n.onnx --source 0 --cmc --stabilize --hfov 70
~~~

The --hfov value must come from camera specifications or calibration. It is used only for image-angle geometry. SpectraTrack does not infer metric range from a single uncalibrated RGB camera.

## Session logging

~~~
python -m spectratrack.app --model ..\models\yolo11n.onnx --source 0 --session-dir .\sessions
~~~

Each session gets meta.json and append-only frames.jsonl with tracks, CMC, selection state and rolling performance data.

## Benchmark

~~~
python -m spectratrack.benchmark --model ..\models\yolo11n.onnx --source "D:\video.mp4" --frames 500 --output benchmark.json
~~~

Compare CPU explicitly with the same command plus --cpu.

## Controls

- Left click: lock/unlock a tracked target.
- E: cycle enhancement OFF -> VISIBILITY -> LOWLIGHT -> DETAIL.
- C: toggle camera-motion compensation.
- Z: toggle optical stabilization.
- H: toggle HUD.
- S: save selected raw crop.
- U: run optional Real-ESRGAN snapshot upscale when configured.
- Q / Esc: quit.

## Data labels

- [CAL]: derived from supplied camera calibration/specification.
- [EST]: estimated from image/tracker geometry.
- No metric range, GPS, thermal reading or physical target velocity is fabricated.

## Optional Real-ESRGAN

Pass a locally installed, trusted realesrgan-ncnn-vulkan.exe with --realesrgan. SpectraTrack contains no downloader for that executable and never runs it through a shell.


## Model provenance manifest

Create a manifest after exporting a trusted ONNX model:

~~~
python ..\tools\make_model_manifest.py ..\models\yolo11n.onnx --name yolo11n --source "Ultralytics export" --input-size 640
~~~

Then run with both the model and manifest. SpectraTrack verifies SHA-256 before loading:

~~~
python -m spectratrack.app --model ..\models\yolo11n.onnx --model-manifest ..\models\yolo11n.onnx.manifest.json --source 0
~~~

## Calibration capture and lens correction

Capture checkerboard views:

~~~
python -m spectratrack.capture_calibration --source 0 --output .\calib --cols 9 --rows 6
~~~

Calibrate:

~~~
python -m spectratrack.calibrate --images ".\calib\*.png" --cols 9 --rows 6 --square 25 --output camera-calibration.json
~~~

Run with distortion correction:

~~~
python -m spectratrack.app --model ..\models\yolo11n.onnx --source 0 --calibration camera-calibration.json --undistort
~~~

## Self-contained replay session

Add --session-video together with --session-dir. SpectraTrack stores tracking-input.mp4 in the same session directory, in the exact coordinate space used by the tracker.

~~~
python -m spectratrack.app --model ..\models\yolo11n.onnx --source 0 --session-dir .\sessions --session-video
python -m spectratrack.replay --session .\sessions\session-YYYYMMDDTHHMMSSZ
~~~
