# PC v0.2 execution plan

This plan is intentionally staged so CI failures cannot turn into blind retry loops.

## Milestone A - tracking core
- [x] 8-state bounding-box Kalman filter.
- [x] Two-stage high/low confidence association.
- [x] Tentative/confirmed/coasting states.
- [x] Camera-motion affine compensation.
- [x] Preserve lightweight dependency footprint.

## Milestone B - observability and reproducibility
- [x] Rolling per-stage profiler.
- [x] JSONL session telemetry with per-track state.
- [x] Headless benchmark command.
- [x] Horizontal-FOV geometry for relative angles only.
- [x] Explicit CAL/EST labels; no fake metric range.

## Milestone C - live UX
- [x] HUD performance counters.
- [x] CMC state and global-motion readout.
- [x] Target association/state readout.
- [x] Enhancement modes: off / visibility / lowlight / detail.
- [x] Runtime toggles for CMC, stabilization and enhancement.

## Milestone D - verification
- [ ] Crossing-target regression test.
- [ ] Camera-motion synthetic regression test.
- [ ] Low-confidence recovery regression test.
- [ ] Enhancement and geometry tests.
- [ ] JSONL recorder test.
- [ ] Windows and Linux PC-only CI green.
- [ ] Main branch fast-forward only after branch CI is green.

## Later, after v0.2 is green
- Optional appearance/ReID model as a separate, explicitly sourced model.
- Optional target segmentation model with a documented ONNX output contract.
- Stereo depth only with calibrated two-camera geometry.
- Offline replay viewer for JSONL sessions.
- Model manifest with provenance + SHA-256.

## Anti-loop rule

If the same CI stage fails twice for the same reason, do not retry a cosmetic variant.
Read the failing log, isolate the smallest reproduction, add a regression test when
possible, then make one targeted change.
