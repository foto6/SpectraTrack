# Cross-video graph

SpectraTrack v0.3 can analyze several local videos with one detector and build
conservative candidate links between tracklets of the same object class.

## Pipeline

1. YOLO detects objects.
2. The per-video tracker builds stable local tracklets.
3. Affine camera-motion compensation reduces fragmentation during camera motion.
4. Each confirmed tracklet collects a small gallery of non-biometric visual descriptors.
5. One best-quality crop is saved for review.
6. Cross-video similarity is computed only within the same detector class.
7. Strong links form conservative complete-link groups. Weak links stay in REVIEW.
8. Manual SAME / DIFFERENT / UNSURE decisions can be exported and applied on a later run.

## Important semantics

- Vehicle/animal/object links are visual same-object candidates, not guaranteed identity.
- Person links mean similar visible appearance in that batch only.
- No face detector, face embedding, biometric template, or named-person recognition is used.
- Two tracklets from the same source video are not automatically merged.
- Generated SpectraTrack output videos are skipped by default to avoid recursive re-analysis.

## Windows standalone

Drag a folder onto `ANALYZE_VIDEO_FOLDER.bat`.

The launcher auto-detects `model.onnx`, then `yolo11x.onnx`, then the first
ONNX model in the application folder.

## Direct CLI

```powershell
SpectraTrack-PC.exe batch `
  --model .\yolo11x.onnx `
  --input-dir "D:\Videos" `
  --output "D:\Videos\spectratrack_cross_video.json" `
  --detect-every 1 `
  --conf 0.20 `
  --iou 0.50 `
  --recursive
```

Optional review round-trip:

```powershell
SpectraTrack-PC.exe batch `
  --model .\yolo11x.onnx `
  --input-dir "D:\Videos" `
  --output "D:\Videos\spectratrack_cross_video.json" `
  --review "D:\Videos\spectratrack_review.json" `
  --recursive
```

The current visual descriptor is a model-free spatial HSV + grayscale +
edge-orientation descriptor with a small per-tracklet viewpoint gallery. It is a
conservative baseline, not a claim of perfect object re-identification.
