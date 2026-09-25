# PC v0.2 engineering plan

This checklist is intentionally finite. New work is added as a new milestone rather
than reopening completed milestones indefinitely.

## Milestone A — repository / build
- [x] Dedicated repository
- [x] PC-only CI separated from Android CI
- [x] Python compile + unit tests
- [x] Synthetic tracker benchmark in CI
- [x] Environment diagnostics in CI
- [x] Standalone Windows/PyInstaller build added to CI
- [ ] Confirm standalone artifact smoke test and publish green artifact

## Milestone B — tracking
- [x] Two-stage high/low-confidence association
- [x] Tentative/confirmed/predicted lifecycle
- [x] Recovery counting
- [x] Track quality score
- [x] Detector cadence + prediction-only frames
- [x] Affine camera-motion compensation
- [x] RANSAC CMC quality gates
- [x] Crossing / dropout / pan / affine regression tests

## Milestone C — imaging / HUD
- [x] Separate detector analysis image from operator display image
- [x] Normal / clarity / low-light / edges / false-color modes
- [x] Explicit false-color labeling
- [x] Stage timing overlay
- [x] Target lifecycle / quality overlay
- [x] Snapshot + optional external Real-ESRGAN hook

## Milestone D — reproducibility
- [x] SHA-256 model verification
- [x] Model manifest with provenance
- [x] JSONL session logging
- [x] Track lifecycle events
- [x] Session report tool
- [x] Headless processing
- [x] Class filters

## Milestone E — calibrated geometry
- [x] Camera calibration JSON
- [x] HFOV calibration helper
- [x] Calibrated angular offset
- [x] Angular target size
- [x] No fabricated monocular metric range

## Milestone F — compatibility
- [x] Raw YOLO ONNX xywh+class-score output
- [x] End-to-end xyxy+score+class output
- [ ] Confirm end-to-end decoder CI
- [ ] Run one real-camera validation session and inspect report

## Next v0.3 candidates
Only after v0.2 is green and tested on real footage:
- optional appearance/ReID association;
- optional selected-target segmentation backend;
- calibrated stereo/multi-camera depth;
- pluggable restoration ONNX backend;
- GPU/backend benchmarks on representative hardware.
