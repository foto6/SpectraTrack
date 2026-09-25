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
- Detector cadence profiles with canonical fast/balanced/high-quality/max-recall names and legacy speed/quality aliases.
- Stage-level live performance report with true detector-only latency and process CPU sampling.
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
- Annotated detector/tracker QA benchmark harness with saved result comparison and NEW FALSE NEGATIVE regression gates.
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

Required sequence:

1. create a representative annotated validation set;
2. measure current person recall/precision/false positives;
3. add offline `people-recall` / `quality-max` mode;
4. add person-only high-resolution pass;
5. add tiled/sliced inference with overlap;
6. merge tile/full-frame detections without duplicates;
7. support class-specific person thresholds;
8. use temporal confirmation/recovery for weak person evidence;
9. benchmark input sizes, tile sizes, enhancement variants, models;
10. only then optimize for live DirectML performance.

Engineering targets for the first offline validation set are recorded in `docs/PC_V03_PLAN.md`.

## Known bugs / known product problems

- Small visible people can be completely missed in high-angle night/compressed footage.
- Generic detector confidence is global; there is no class-specific `person` threshold yet.
- There is no tiled/sliced detector path.
- There is no multi-scale or dedicated second person pass.
- The annotated QA harness exists, but no representative real-world person-recall corpus has been collected and validated yet.
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
- Add validated Windows GPU utilization/VRAM telemetry across DirectML hardware; do not infer unavailable counters.
- Populate and validate the detector-level benchmark with representative real footage, provenance, and a held-out subset.
- Add H.264/H.265/FFmpeg output path with optional original-audio mux.
- Add richer progress/ETA and per-stage performance summaries for long offline runs.
- Add model compatibility tests for more verified ONNX exports without guessing unknown layouts.
- Calibrate cross-video similarity thresholds on real multi-view object data.
- Consider a learned non-biometric object embedding for vehicles/animals if the hand-built descriptor ceiling is reached.
- Decide whether LockRefiner should remain display-only or become a controlled tracker measurement source.
- Consolidate older docs under `docs/` with this root architecture so stale descriptions do not diverge.
- Version the session/cross-video schemas explicitly if they evolve beyond compatible additive changes.

## Ideas not implemented

- quality-max / people-recall detector mode;
- tiled/sliced inference;
- class-specific thresholds;
- temporal accumulation of weak person detections before creating tracks;
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

Benchmark harness/regression checks are implemented. Next collect real validation assets, establish BASELINE, then use the same result schema for branch-integration and release checks.
