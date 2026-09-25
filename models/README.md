# Model setup

SpectraTrack does **not** ship or auto-download neural-network weights. This is intentional:
the runtime has no model downloader, and the model hash/provenance can be verified before ONNX
Runtime creates a session.

## Recommended baseline

Use a fixed-size YOLO-style ONNX export. The PC decoder currently supports:

- raw `xywh + class scores`, e.g. `[1,84,N]` / `[1,N,84]` for COCO-80;
- common end-to-end post-NMS `xyxy, score, class` rows.

## Export locally

From the repository root:

```powershell
py -3.12 -m venv .export-env
.\.export-env\Scripts\Activate.ps1
python -m pip install -U pip
pip install ultralytics onnx onnxslim

python .\tools\export_yolo.py ^
  --model yolo11n.pt ^
  --imgsz 640 ^
  --opset 17 ^
  --output models\yolo11n.onnx ^
  --name yolo11n-coco-640
```

The exporter now creates two files:

```text
models/yolo11n.onnx
models/yolo11n.manifest.json
```

The manifest contains SHA-256, input size and provenance. PC export is the default; pass
`--android-copy` only when you intentionally want to copy the weight into Android assets.

## Inspect an ONNX model

From `pc/` after installing `requirements-win.txt`:

```powershell
python -m spectratrack.model_inspect ..\models\yolo11n.onnx
```

To generate a fresh manifest from an already-existing ONNX:

```powershell
python -m spectratrack.model_inspect ..\models\yolo11n.onnx ^
  --manifest-out ..\models\yolo11n.manifest.json ^
  --name yolo11n-coco-640 ^
  --source "local verified export"
```

## Run with manifest verification

```powershell
python -m spectratrack.app ^
  --model ..\models\yolo11n.onnx ^
  --model-manifest ..\models\yolo11n.manifest.json
```

If the file hash differs from the manifest, SpectraTrack exits before inference.

## Unsupported layouts

Segmentation masks, pose/keypoints, explicit objectness layouts and custom multi-output
architectures are not silently guessed. Add/verify a decoder for those layouts first.
