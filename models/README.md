# Model setup

SpectraTrack does **not** ship or auto-download neural-network weights. This is intentional: you can see exactly what files are added to the project and hash them yourself.

## Recommended baseline

Use a fixed-size `640x640` YOLOv8/YOLO11-style ONNX export with an output shaped like `[1, 84, N]` or `[1, N, 84]` for COCO-80.

From the repository root on Windows:

```powershell
py -3.12 -m venv .export-env
.\.export-env\Scripts\Activate.ps1
python -m pip install -U pip
pip install ultralytics onnx onnxslim
python .\tools\export_yolo.py --model yolo11n.pt --imgsz 640 --opset 17
```

The script copies the result to both:

- `models/yolo11n.onnx` for the PC client;
- `android/app/src/main/assets/yolo11n.onnx` for the Android client.

It also prints SHA-256. Save that value if you want to verify the file later.

## Other models

The decoder is intentionally simple. A different model is fine if it emits the same `xywh + class scores` layout. If your model has objectness as a separate field, end-to-end NMS, segmentation masks, or another output layout, adapt the decoder first.
