# Changelog

## 0.3.0 - in development

### Added
- selected-target sparse optical-flow lock refinement between detector observations;
- cross-video batch analyzer and conservative tracklet/entity graph;
- spatial HSV/grayscale/edge descriptor galleries for cross-video candidate matching;
- best-tracklet preview crops and local HTML review report;
- exportable SAME / DIFFERENT / UNSURE decisions with review-file round trip;
- affine camera-motion compensation in folder batch analysis;
- one-click single-video and folder-analysis Windows launchers;
- automatic ONNX discovery in Windows launchers;
- generated-video exclusion to prevent recursive self-analysis.

### Reliability
- video files ignore camera-only DirectShow/MSMF backend flags;
- source video FPS is preserved when recording processed output;
- CLI abbreviation is disabled so --conf cannot be misparsed as --config;
- runtime presets are included in standalone/source CI packages;
- complete-link clustering prevents A≈B≈C chains from automatically collapsing into A=C;
- manual DIFFERENT decisions block automatic grouping.

### Safety / data semantics
- person links are explicitly same-appearance candidates, not biometric identity;
- no face recognition or facial embedding is used;
- object links remain reviewable visual candidates, not guaranteed identity.

## 0.2.0 - in development

### Added
- two-stage object association;
- affine camera-motion compensation with RANSAC quality gates;
- tentative/tracked/predicted lifecycle and track quality;
- detector cadence profiles with prediction-only skipped frames;
- calibrated angular offset/size HUD;
- camera FOV calibration helper;
- model SHA-256 verification and provenance manifests;
- JSONL session recorder and session report;
- headless batch processing and class filters;
- pipeline timings, diagnostics and synthetic benchmark;
- independent normal/clarity/low-light/edge/false-color operator views;
- support for common end-to-end YOLO ONNX output;
- standalone Windows artifact build in CI.

### Changed
- PC and Android CI workflows are separated.
- Operator display transforms no longer alter the detector input image.
- Camera motion is modeled separately from target residual image motion.

### Safety / data semantics
- false-color mode is explicitly not thermal imaging;
- calibrated angular values are distinguished from measured data;
- no monocular metric target range is fabricated;
- neural SR remains an optional selected-crop operation.
