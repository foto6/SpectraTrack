# PC v0.2 execution plan

The work is staged so CI failures cannot turn into blind retry loops.

## Milestone A - tracking core
- [x] 8-state bounding-box Kalman filter.
- [x] Two-stage high/low confidence association.
- [x] Tentative / LOCK / COAST / PREDICT states.
- [x] Camera-motion affine compensation.
- [x] Preserve lightweight dependency footprint.

## Milestone B - observability and reproducibility
- [x] Rolling per-stage profiler.
- [x] JSONL session telemetry with per-track state.
- [x] Headless benchmark command.
- [x] Horizontal-FOV geometry for relative angles only.
- [x] Explicit CAL/EST labels; no fake metric range.
- [x] Session analyzer for track/performance summaries.
- [x] Environment/model/camera diagnostics CLI.
- [x] Optional model SHA-256 verification.
- [x] Model provenance manifest with runtime hash verification.
- [x] Optional synchronized tracking-input session video.
- [x] Offline telemetry replay.

## Milestone C - live UX and throughput
- [x] HUD performance counters.
- [x] CMC state and global-motion readout.
- [x] Target association/state readout.
- [x] Enhancement modes: off / visibility / lowlight / detail.
- [x] Runtime toggles for CMC, stabilization and enhancement.
- [x] Fixed detector cadence with prediction on skipped frames.
- [x] Adaptive detector cadence with hysteresis.
- [x] Optional classical GrabCut outline for selected target.
- [x] Checkerboard capture helper.
- [x] Checkerboard camera calibration.
- [x] Optional lens distortion correction before CMC/detection.

## Milestone D - verification
- [x] Crossing-target regression test.
- [x] Camera-motion synthetic regression test.
- [x] Low-confidence recovery regression test.
- [x] Enhancement and geometry tests.
- [x] JSONL recorder test.
- [x] Windows and Linux PC-only CI green after core layer.
- [x] Explicit regression for scheduled detector skips.
- [x] Model hash / mask / scheduler / session-analyzer tests.
- [x] Windows and Linux PC-only CI green after throughput layer.
- [ ] Calibration/replay/provenance layer CI green.
- [ ] Standalone Windows artifact build green.
- [ ] Main branch fast-forward only after final branch CI is green.

## Later candidates
- Optional appearance/ReID model as a separate, explicitly sourced model.
- Optional neural target segmentation model with a documented ONNX output contract.
- Stereo depth only with calibrated two-camera geometry.
- Rich offline timeline viewer and export.
- Performance preset auto-tuner based on benchmark results.

## Anti-loop rule

If the same CI stage fails twice for the same reason, do not retry a cosmetic variant.
Read the failing log, isolate the smallest reproduction, add a regression test when
possible, then make one targeted change.
