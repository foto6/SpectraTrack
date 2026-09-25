# Detection engineering

This document is the current contract for the PC detection subsystem on the `agent/detection` workstream.

## Scope

Primary code:

- `pc/spectratrack/detector.py`
- detector call paths in `pc/spectratrack/app.py` and `pc/spectratrack/batch.py`
- `pc/spectratrack/runtime_config.py`
- `pc/presets/people-recall.json`
- `pc/spectratrack/detection_benchmark.py`
- detector and benchmark tests
- `pc/detection_regression/`

Do not change `Detection`, tracker semantics, CMC, cross-video descriptors, or UI/backend contracts from this workstream without a separate measured reason and coordination.

## Baseline contract

`YoloOnnxDetector.detect(frame)` is the stable standard path.

Defaults remain:

- input size request: 640
- confidence: 0.35
- NMS IoU: 0.45
- DirectML preferred on Windows, CPU fallback
- fixed-size YOLO-compatible ONNX
- raw YOLO or known end-to-end rows only

Standard mode behavior is intentionally unchanged by people-recall work.

## People-recall mode

Enable with:

```text
--detector-mode people-recall
```

The mode uses the same ONNX session and returns the same `list[Detection]` contract.

Pipeline:

```text
source frame
  ├─ full-frame inference
  │    └─ person-specific lower threshold
  └─ overlapping source-pixel tiles
       └─ same ONNX detector
            └─ keep person results
                 ↓
       offset boxes to source coordinates
                 ↓
       class-aware NMS merge with full-frame results
                 ↓
             Detection[]
```

Current uncalibrated defaults:

- `person_conf = 0.12`
- `person_tile_size = 640`
- `person_tile_overlap = 0.20`
- `person_merge_iou = 0.55`

These are feature defaults, not evidence that they are optimal.

## Why tiling is the first implementation

Large CCTV frames are downscaled into the fixed detector input. A small person can therefore collapse to only a few model-input pixels. Source-image tiles reduce that downscaling and are an additive strategy that does not require replacing the detector model or changing tracker types.

The implementation intentionally does not add SAHI as a dependency. The required slice/detect/remap/merge behavior is small enough to keep inside the existing detector and avoids another runtime package.

## Class-specific threshold behavior

The standard decoder now supports per-label threshold overrides while retaining the global confidence threshold for other classes.

This works for raw YOLO outputs and for boxes that remain present in supported end-to-end output tensors. If an exported end-to-end ONNX graph already removed weak boxes internally, external threshold changes cannot recover them.

## Merge strategy

Current merge is the existing class-aware hard NMS.

Soft-NMS and Weighted Boxes Fusion are not implemented yet because there is no representative benchmark proving they improve the failure domain. Compare them only after the annotation set exists.

## Benchmark

`python -m spectratrack.detection_benchmark` consumes annotated still frames.

Manifest example:

```json
{
  "samples": [
    {
      "image": "frames/camera_a_000123.jpg",
      "persons": [[412, 188, 438, 251]],
      "tags": ["night", "high-angle", "compressed"]
    }
  ]
}
```

Metrics:

- recall
- precision
- false positives/frame
- true/false positives and false negatives
- mean and median detector latency
- effective processed frames/s
- recall by apparent person height: <32 px, 32-63 px, >=64 px
- recall by annotation tag
- exact model SHA-256, providers and detector settings

DirectML VRAM is currently `null` with an explicit note. Do not substitute process RAM or a guessed GPU value.

## Required real validation set

Before changing defaults or claiming improvement, annotate representative frames that include:

- small/distant people
- high camera angle
- night/poor illumination
- backlight
- motion blur
- JPEG/video compression
- partial occlusion
- unusual pose
- negative/confusing regions without people

Every clearly visible person in each selected frame must be annotated, including misses from the current detector. Split tuning and holdout samples before freezing thresholds.

## Current known limitation at detector ↔ tracker boundary

`MultiObjectTracker` currently uses:

- `low_conf = 0.12`
- `high_conf = 0.45`

Detections below `high_conf` can help associate/recover existing tracks but cannot create a new track ID by themselves. That is intentional existing tracker behavior.

Do not lower the tracker creation threshold globally from the detection branch. The planned solution is temporal confirmation of spatially consistent weak person evidence, benchmarked separately.

## Experiment matrix once data is available

Record each run with the same holdout annotations and model hash.

Compare:

1. standard full-frame baseline;
2. people-recall with person threshold sweep;
3. tile size and overlap sweep;
4. original RGB vs existing non-generative analysis enhancement;
5. input/model variants;
6. hard NMS vs Soft-NMS/WBF only after the above baseline;
7. current YOLO11 family variants and any newer candidate only after ONNX output compatibility is verified;
8. dedicated/fine-tuned overhead person model if generic weights remain the bottleneck.

For every run save recall, precision, false positives/frame, latency/FPS and trustworthy VRAM if a measurement path is available.

## Non-goals

- no face recognition or biometric identity
- no generative SR as ground-truth evidence
- no silent model downloads
- no fabricated recall/FPS/VRAM numbers
- no tracker rewrite inside detection work
