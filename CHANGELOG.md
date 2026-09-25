# Changelog

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
