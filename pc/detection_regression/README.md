# Person-detection regression set

This directory defines the on-disk contract for the detector-level benchmark.

A representative real regression set is intentionally **not fabricated** here. Add legally usable frames from the actual failure domain (high-angle, night, compression, blur, partial occlusion) and annotate every clearly visible person before using benchmark numbers for engineering decisions.

## Manifest format

Paths are relative to the manifest file.

```json
{
  "samples": [
    {
      "image": "frames/camera_a_000123.jpg",
      "persons": [
        [412, 188, 438, 251]
      ],
      "tags": ["night", "high-angle", "compressed"]
    }
  ]
}
```

Boxes are `[x1, y1, x2, y2]` in source-image pixels.

## Run baseline

```powershell
python -m spectratrack.detection_benchmark ^
  --model ..\models\yolo11.onnx ^
  --manifest detection_regression\manifest.json ^
  --mode standard ^
  --output baseline.json
```

## Run people-recall mode

```powershell
python -m spectratrack.detection_benchmark ^
  --model ..\models\yolo11.onnx ^
  --manifest detection_regression\manifest.json ^
  --mode people-recall ^
  --person-conf 0.12 ^
  --person-tile-size 640 ^
  --person-tile-overlap 0.20 ^
  --output people-recall.json
```

The output records model SHA-256, providers, thresholds, recall, precision, false positives/frame, latency/FPS, recall by person height, and recall by tags. DirectML VRAM is reported as unavailable rather than guessed.
