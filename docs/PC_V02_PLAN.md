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

## Milestone C - live UX and throughput
- [x] HUD performance counters.
- [x] CMC state and global-motion readout.
- [x] Target association/state readout.
- [x] Enhancement modes: off / visibility / lowlight / detail.
- [x] Runtime toggles for CMC, stabilization and enhancement.
- [x] Fixed detector cadence with prediction on skipped frames.
- [x] Adaptive detector cadence with hysteresis.
- [x] Optional classical GrabCut outline for selected target.

## Milestone D - verification
- [x] Crossing-target regression test.
- [x] Camera-motion synthetic regression test.
- [x] Low-confidence recovery regression test.
- [x] Enhancement and geometry tests.
- [x] JSONL recorder test.
- [x] First Windows and Linux PC-only CI green.
- [x] Explicit regression for scheduled detector skips.
- [x] Model hash / mask / scheduler / session-analyzer tests.
- [ ] Second Windows and Linux CI green after throughput/diagnostics layer.
- [ ] Main branch fast-forward only after final branch CI is green.

## Later, after v0.2 is green
- Optional appearance/ReID model as a separate, explicitly sourced model.
- Optional neural target segmentation model with a documented ONNX output contract.
- Stereo depth only with calibrated two-camera geometry.
- Checkerboard calibration CLI and distortion correction.
- Offline video + telemetry replay viewer.
- Model manifest with provenance + SHA-256.

## Anti-loop rule

If the same CI stage fails twice for the same reason, do not retry a cosmetic variant.
Read the failing log, isolate the smallest reproduction, add a regression test when
possible, then make one targeted change.
