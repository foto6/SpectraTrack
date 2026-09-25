# TASKS.md

This file is the current engineering task ledger for parallel agents.

Baseline: `main@2012eaae2f4ffe820a66d12e40346d911616cd03`.

## Completed

- Dedicated `foto6/SpectraTrack` repository.
- PC and Android code separated into explicit clients.
- Windows ONNX detector with DirectML preference and CPU fallback.
- Raw YOLO and common end-to-end detector decoding.
- Model SHA-256 verification and provenance manifest support.
- Robust camera/video capture with bounded webcam reconnects.
- Camera-only backend flags no longer break normal video files.
- Source FPS preserved for processed video recording.
- CLI abbreviation disabled; `--conf` no longer collides with `--config`.
- Two-stage high/low confidence tracker with stable IDs and lifecycle states.
- Detector cadence profiles: quality/balanced/speed.
- Prediction-only skipped-frame tracking.
- Affine CMC using optical flow + RANSAC gates.
- Non-biometric HSV appearance cue for same-class tracking.
- Selected-target sparse optical-flow lock refinement.
- Operator display modes separated from detector analysis path.
- Calibration-based angular offset/size display.
- JSONL session logging and lifecycle events.
- Session report / CSV / metadata overlay utilities.
- Snapshot provenance sidecars.
- Optional external Real-ESRGAN snapshot enhancement.
- Headless processing and class filters.
- Synthetic tracker benchmark, diagnostics and self-check.
- Standalone Windows PyInstaller CI with CLI smoke tests.
- Native Android CameraX + ONNX Runtime client.
- Cross-video batch analysis.
- Conservative cross-video entity graph with complete-link grouping.
- Best-tracklet previews and local HTML review.
- SAME / DIFFERENT / UNSURE review round trip.
- Collision-safe recursive video IDs.
- Generated-output exclusion to avoid batch feedback loops.
- End-to-end synthetic AVI batch regression test.
- Published `v0.3.0` Windows/Android release pipeline.
- Feature-flagged PC `people-recall` detector mode with a lower person-specific threshold.
- Overlapping tiled/sliced person pass with full-frame coordinate remapping and class-aware NMS merge.
- Detector-level annotated person benchmark harness with recall, precision, false positives/frame, latency/FPS, pixel-height buckets, tags, model SHA-256, and exact settings.

## Partially implemented / limited

### Selected-target LockRefiner

Implemented and live in `app.py`, but refined bbox currently changes `display_tracks` for HUD rendering only. It does not feed the refined bbox back into the base tracker state or the per-frame JSONL track bbox.

### Appearance/Re-ID

- live tracker uses a simple HSV histogram cue;
- cross-video uses a richer hand-built spatial HSV/grayscale/edge gallery;
- this is not a learned Re-ID system and is expected to fail on visually similar objects or large viewpoint/lighting changes.

### Cross-video grouping

The graph and review workflow work, but similarity thresholds have not been calibrated on a large representative real dataset.

### Android

Maintained and buildable, but PC v0.3 tracking/CMC/cross-video features are not feature-parity implementations on Android.

### Output encoding

Processed PC video uses OpenCV `mp4v`; original audio is not carried into the output.

## Highest-priority active work

### High-recall small-person detection in bad video

Representative failure: high-angle, night-time, compressed CCTV frames where visible people are small and the current generic YOLO pass detects many cars but misses people.

Do not solve this by only lowering global `--conf`.

Follow `docs/PC_V03_PLAN.md`.

Current implementation state:

1. representative annotated validation set — **missing**;
2. real baseline recall/precision/false positives — **not measured yet**;
3. `people-recall` feature flag — implemented;
4. lower person-specific threshold — implemented;
5. overlapping tiled/sliced second pass — implemented;
6. full-frame/tile coordinate merge with class-aware NMS — implemented;
7. detector benchmark harness — implemented;
8. temporal confirmation of weak new-person evidence — not implemented;
9. model/input/tile/enhancement comparison on real data — not measured;
10. DirectML live optimization — intentionally deferred until recall is measured.

Engineering targets for the first real validation set remain in `docs/PC_V03_PLAN.md`. See `docs/detection.md` for the current detector contract and benchmark workflow.

## Known bugs / known product problems

- Small visible people can be completely missed in high-angle night/compressed footage.
- There is no annotated real-world person-recall dataset in the repository yet, so no real recall/precision improvement is claimed.
- `people-recall` adds multiple inference calls per detector frame; live DirectML FPS/VRAM impact is not measured yet.
- End-to-end ONNX exports that already suppress boxes inside the graph can limit how much an external lower threshold can recover.
- The tracker only creates a new ID from detections at or above its existing `high_conf=0.45`; weak person detections can recover existing tracks but do not yet get temporal confirmation into new tracks.
- Cross-video hand-built descriptors can confuse visually similar vehicles/animals.
- Cross-video similarity thresholds are engineering defaults, not dataset-calibrated probabilities.
- Greedy tracker association can make suboptimal assignments in dense crossings.
- Output video currently drops source audio.
- Current HUD can become visually dense with many detections/tracks.
- Lock refinement is display-side and is not fused back into tracker state.
- Batch detection currently analyzes the original frame; it has no dedicated poor-light person-recall preprocessing path.
- Android does not have parity with PC cross-video/CMC/refinement functionality.

## Technical debt

- Replace or benchmark greedy association against a global assignment method.
- Evaluate Kalman/ByteTrack/BoT-SORT-style tracking only with regression evidence.
- Add representative GPU/backend benchmarks, not only synthetic tracker benchmark.
- Add H.264/H.265/FFmpeg output path with optional original-audio mux.
- Add richer progress/ETA and per-stage performance summaries for long offline runs.
- Add model compatibility tests for more verified ONNX exports without guessing unknown layouts.
- Calibrate cross-video similarity thresholds on real multi-view object data.
- Consider a learned non-biometric object embedding for vehicles/animals if the hand-built descriptor ceiling is reached.
- Decide whether LockRefiner should remain display-only or become a controlled tracker measurement source.
- Consolidate older docs under `docs/` with this root architecture so stale descriptions do not diverge.
- Version the session/cross-video schemas explicitly if they evolve beyond compatible additive changes.

## Ideas not implemented

- temporal accumulation of weak person detections before creating tracks;
- benchmark Soft-NMS / Weighted Boxes Fusion against the current class-aware NMS before changing merge semantics;
- multi-scale person passes beyond the current full-frame + fixed-source-tile strategy;
- dedicated small-object/overhead model or fine-tuned person model;
- selected-target segmentation;
- learned generic object embedding for cross-video matching;
- H.264/H.265 hardware encoding;
- automatic original-audio remux;
- drag-and-drop GUI beyond BAT launchers;
- automatic model manager/downloader; current runtime intentionally does not silently download models;
- adaptive tile scheduling based on scene content;
- GPU/VRAM telemetry overlay;
- best-frame/miss-review export specifically for detector benchmark iteration;
- stereo/multi-camera calibrated depth.

## Suggested parallel-agent split

### agent/detection

Own the small-person recall milestone and detector benchmark. Avoid tracker rewrites.

### agent/tracking

Improve association/recovery/LockRefiner integration without changing detector semantics unless coordinated.

### agent/enhancement

Benchmark non-generative preprocessing for poor-light detection and operator display. Do not treat generative SR as source evidence.

### agent/performance

Profile DirectML pipeline, capture, detector cadence, future tiling, encoding and long-run throughput.

### agent/qa

Build real validation assets, benchmark harness, regression tests, branch-integration checks and release verification.
