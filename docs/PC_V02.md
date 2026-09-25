# SpectraTrack PC v0.2

## Goal

A local-only computer-vision workstation for ordinary cameras and video files:
object detection, stable IDs, camera-motion compensation, target lock, honest
image-space kinematics, calibrated angular offsets, operator enhancement views,
session recording and reproducible model provenance.

It is not a thermal imager, physical rangefinder, biometric identifier, or fire-control system.

## Recommended start

From `pc/`:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
pip install -r requirements-win.txt
python -m spectratrack.diagnostics
```

Generate/export a compatible ONNX model as described in `../models/README.md`.

Run:

```powershell
python -m spectratrack.app ^
  --model ..\models\yolo11n.onnx ^
  --source 0 ^
  --profile balanced ^
  --session-log sessions\camera.jsonl
```

For PowerShell, write the command on one line or use backticks instead of `^`.

## Profiles

- `quality`: detector every frame.
- `balanced`: detector every 2nd frame; prediction-only between detector frames.
- `speed`: detector every 3rd frame.

Override explicitly with `--detect-every N`.

Skipped detector frames do not increment `missed`. The tracker advances with its
motion model and camera-motion compensation until the next scheduled detector frame.

## Camera-motion compensation

When stabilization is off, SpectraTrack estimates a RANSAC affine/similarity transform
from optical flow. Translation, rotation and small scale changes are applied to tracked
boxes before target residual motion is estimated.

Bad transforms are rejected by:

- minimum inlier count;
- minimum inlier ratio;
- translation sanity bounds;
- rotation sanity bounds;
- scale sanity bounds.

When optical stabilization is enabled, the tracker does not apply CMC a second time.

## Operator views

Use `--view MODE` or press `M`:

- `normal`
- `clarity`
- `lowlight`
- `edges`
- `pseudo-thermal`

`pseudo-thermal` is false-color luminance only. It is not IR/thermal data.

The detector analysis image is separate from the operator display image. Changing the
view does not feed false-color imagery into YOLO.

Press `E` to toggle non-generative clarity enhancement on the detector analysis image.

## Controls

- left click: select/unselect a tracked object;
- `E`: analysis enhancement;
- `M`: cycle operator view;
- `Z`: toggle optical stabilization; tracker resets to avoid mixing coordinate spaces;
- `R`: reset tracker IDs;
- `H`: toggle HUD;
- `S`: save selected original crop;
- `U`: save crop and optionally send it to a locally supplied Real-ESRGAN ncnn/Vulkan executable;
- `Q` / `Esc`: quit.

## Model integrity

Direct hash:

```powershell
python -m spectratrack.app --model ..\models\yolo11n.onnx --model-sha256 <SHA256>
```

Manifest:

```powershell
python -m spectratrack.app ^
  --model ..\models\yolo11n.onnx ^
  --model-manifest ..\models\my-model.json
```

See `../models/model-manifest.example.json`.

The app aborts before creating the ONNX session if the hash does not match.

## Class filters

```powershell
python -m spectratrack.app --model ..\models\yolo11n.onnx --classes person,car,truck
```

This filters post-inference detections before tracking.

## Calibration and angular measurements

A calibration JSON gives image width/height and camera field of view.

Example:

```powershell
python -m spectratrack.calibrate_fov ^
  --image-width 1920 ^
  --image-height 1080 ^
  --object-px 420 ^
  --object-m 1.0 ^
  --distance-m 3.0 ^
  --output calibration.json
```

Then:

```powershell
python -m spectratrack.app ^
  --model ..\models\yolo11n.onnx ^
  --calibration calibration.json
```

The HUD can then show target angular offset and angular size. These are labeled
estimated/calibrated quantities. No metric target distance is invented from a single
ordinary camera.

Calibration is valid only for the same camera, resolution, crop and zoom state.

## Session recording

```powershell
python -m spectratrack.app ^
  --model ..\models\yolo11n.onnx ^
  --session-log sessions\run01.jsonl
```

Each frame records IDs, boxes, confidence, residual image velocity, lifecycle state,
quality, prediction/recovery information, CMC data and stage timings.

Generate a report:

```powershell
python -m spectratrack.session_report sessions\run01.jsonl --json sessions\run01-report.json
```

## Headless video processing

```powershell
python -m spectratrack.app ^
  --model ..\models\yolo11n.onnx ^
  --source input.mp4 ^
  --headless ^
  --record output.mp4 ^
  --session-log output.jsonl
```

Use `--max-frames N` for bounded tests.

## Diagnostics

```powershell
python -m spectratrack.diagnostics
```

This prints Python/OpenCV/NumPy/ONNX Runtime versions and available execution providers.
It does not open cameras unless you explicitly add `--probe-cameras`.

## Synthetic tracker benchmark

```powershell
python -m spectratrack.benchmark --frames 3000 --targets 32
```

CI requires a minimum synthetic tracker throughput, which catches severe performance
regressions independently of the neural model.

## Data labels

- `MEASURED`: direct sensor data.
- `CALIBRATED`: derived using a calibration.
- `EST`: image/model estimate.
- `AI ENHANCED`: neural reconstruction which may invent fine details.

Never treat neural super-resolution as forensic recovery of text, faces, plates or other
information that was absent from the source frame.
