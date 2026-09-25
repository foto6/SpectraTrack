# Windows client

## 1. Install

Use 64-bit Python 3.12 on Windows 11.

```powershell
cd pc
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
pip install -r requirements-win.txt
```

## 2. Test immediately

```powershell
python -m spectratrack.demo
```

No camera and no neural model are required for this test.

## 3. Put the model in place

Follow `../models/README.md` so `../models/yolo11n.onnx` exists.

## 4. Run a camera

```powershell
python -m spectratrack.app --model ..\models\yolo11n.onnx --source 0 --stabilize
```

For a video file:

```powershell
python -m spectratrack.app --model ..\models\yolo11n.onnx --source "D:\video.mp4"
```

Controls: `Q/Esc` quit, `E` enhancement, `Z` stabilization, `H` HUD, mouse click target lock, `S` save selected crop, `U` run optional neural SR on the selected crop.

## 5. Optional Real-ESRGAN / Vulkan

Install an official `realesrgan-ncnn-vulkan` build yourself. Do not replace it with a random repack. Then pass the exact executable path:

```powershell
python -m spectratrack.app `
  --model ..\models\yolo11n.onnx `
  --source 0 `
  --realesrgan "C:\Tools\realesrgan-ncnn-vulkan.exe"
```

Press `U` while a target is locked. SpectraTrack saves the raw crop first and then asks the executable for an x4 image.

## AMD note

The Python client uses `onnxruntime-directml`, so it does not require CUDA/NVIDIA. On Windows the detector will display its active ONNX providers in the HUD. Pass `--cpu` to compare performance.

## 6. Cross-video batch graph

Standalone Windows build:

- drag one video onto `PROCESS_VIDEO.bat`;
- drag a folder of videos onto `ANALYZE_VIDEO_FOLDER.bat`.

Direct CLI:

```powershell
SpectraTrack-PC.exe batch `
  --model .\yolo11x.onnx `
  --input-dir "D:\Videos" `
  --output "D:\Videos\spectratrack_cross_video.json" `
  --detect-every 1 `
  --conf 0.20 `
  --recursive
```

Outputs:

- `spectratrack_cross_video.json` — machine-readable track/entity graph;
- `spectratrack_cross_video.html` — local visual review;
- `spectratrack_cross_video_samples\` — one best crop per retained tracklet.

The HTML supports local SAME / DIFFERENT / UNSURE review and can export
`spectratrack_review.json`. Pass it back with `--review` to rebuild groups
with manual decisions applied.

For `person`, cross-video links mean similar visible appearance in the current
batch only. SpectraTrack does not perform face recognition or claim biometric identity.


## 7. High-recall person mode

This mode is opt-in. Standard detection remains the default.

```powershell
python -m spectratrack.app `
  --model ..\models\yolo11n.onnx `
  --source "D:\video.mp4" `
  --detector-mode people-recall `
  --person-conf 0.12 `
  --person-tile-size 640 `
  --person-tile-overlap 0.20
```

Or use the provided preset:

```powershell
python -m spectratrack.app --model ..\models\yolo11n.onnx --source "D:\video.mp4" --config presets\people-recall.json
```

The mode runs a full-frame pass plus overlapping tiled passes and merges duplicate person boxes. It is intentionally more expensive than standard mode.

Do not treat the default `0.12/640/0.20` values as calibrated. Measure them against representative annotations first.
