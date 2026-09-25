# Tracking / Re-ID agent state

## Ownership

Branch: `agent/tracking-integration`

Role mapping remains `tracking.md -> agent/tracking`; this branch name is retained because the earlier `agent/tracking` history was not force-rewritten.

Primary scope:

- track association and lifecycle;
- temporary-loss recovery/reactivation;
- CMC interaction with tracking;
- track continuity and ID-switch reduction;
- cross-video multi-signal object linking;
- global object IDs and their semantics;
- tracking/Re-ID tests.

Normally outside scope:

- detector threshold/model strategy;
- frame enhancement;
- performance pipeline refactors;
- QA framework ownership.

## Current objective

Preserve object identity through short/long detector dropouts and provide conservative cross-video object grouping without presenting visual similarity as biometric identity.

## Important invariants

- Same-class association remains mandatory.
- Person cross-video links are appearance candidates, not identity claims.
- Re-ID scores are not probabilities unless calibrated.
- CMC must not be double-applied.
- Dormant/reactivated tracks must not create easy ID hijacking between similar nearby objects.

## Current branch state

Current shared base observed from `origin/integration`: `ee76f4e0c9728af9f94a363c3cee682130599521`.

Original implementation base: `573087cdf60c1a00bbc93b8764733b2ade3cbdd8`.

Current branch: `agent/tracking-integration`.

Validated implementation HEAD before this handoff-file commit: `0379da5e497a10afe8a3462895fcca46185cfe59`.

After review, current `integration @ ee76f4e0c9728af9f94a363c3cee682130599521` was merged into this branch only to import the canonical coordination files and resolve the `docs/agents/tracking.md` add/add conflict. No agent feature branch was merged. The branch is now ahead of `integration` and no longer behind it.

Implemented work:

- bounded dormant pool for confirmed tracks after long detector loss;
- multi-cue dormant reactivation using same class, appearance, shape/size, spatial plausibility and optional direction;
- dormant geometry follows CMC across both detector and `predict_only()` frames;
- deterministic crossing and long-reappearance identity probes in the synthetic benchmark;
- multi-signal cross-video scoring using appearance/gallery, color, shape, relative size, and same-video temporal/direction context;
- explicit `global_object_id` separate from video-local `track_id`;
- schema version 2 for the cross-video graph while preserving legacy `entity_id` and edge `similarity`;
- regression tests for reappearance, false reactivation, missing appearance, expiry/reset, same-video fragments, overlap rejection, CMC during dormancy, multi-signal scoring and temporal gap handling.

## Files changed

Production / benchmark:

- `pc/spectratrack/tracker.py`
- `pc/spectratrack/crossvideo.py`
- `pc/spectratrack/batch.py`
- `pc/spectratrack/benchmark.py`

Tests:

- `pc/tests/test_tracker.py`
- `pc/tests/test_crossvideo.py`
- `pc/tests/test_batch_integration.py`

Shared architecture/task records changed earlier because tracker defaults and the cross-video serialized schema genuinely changed:

- `ARCHITECTURE.md`
- `DECISIONS.md`
- `TASKS.md`

Final coordination update is limited to this file: `docs/agents/tracking.md`.

## Commits owned on current branch

- `ad3564c8b87c691e85d10cc124032cd0d51eff7f` — docs: record tracking re-id design
- `aa8ceaacb64eb9bf11e17cd5480e98fd0587101b` — tracking: reactivate confirmed tracks after occlusion
- `390a64b8aca728594e4bbc1ac30edc6f1651ce49` — reid: add multi-signal global object scoring
- `b4b8d9f7eb3ace44452f90a33a5158cb95d81e88` — tests: cover dormant tracker edge cases
- `650f7358f966bd0df3b8b9f59d23c934ea09269c` — tracking: skip dormant work on empty hot path
- `475f9ce9f6756816f459b31e7cdb13a5cae4e0d6` — tracking: keep dormant geometry aligned with CMC
- `23e741a10ab6804f037fd53aa5a32c0b0f9f0a28` — reid: fix inclusive tracklet gap calculation
- `0379da5e497a10afe8a3462895fcca46185cfe59` — docs: merge tracking re-id docs onto integration
- `74fee97219ff4a2daacd1c946327c70027e06f00` — docs: finalize tracking agent handoff state
- `a4d1aaa000b17fd78d001808eaaf70142c05412d` — merge: sync current integration coordination state

The final metadata-only commit that updates this file is intentionally not self-referenced by SHA; the final branch HEAD is reported in the handoff response.

## Tests actually run

GitHub Actions run `36141844051`, Windows Server 2025 / Python 3.12, PR #6 merge ref for implementation HEAD `0379da5e497a10afe8a3462895fcca46185cfe59`:

- `ruff check spectratrack tests` — passed;
- `python -m compileall -q spectratrack tests` — passed;
- `pytest -q` — **98 passed in 1.02s**;
- `python -m spectratrack.benchmark --frames 500 --targets 24` — **1304.8 tracker FPS**, 24 active tracks;
- deterministic benchmark probes — **crossing_id_switches=0**, **reappearance_id_switches=0**;
- `python -m spectratrack.diagnostics` — passed;
- `python -m spectratrack.selfcheck` — `SELF_CHECK=PASS`;
- PyInstaller standalone Windows build — passed;
- `SpectraTrack-PC.exe --help` — passed;
- `SpectraTrack-PC.exe batch --help` — passed.

The PyInstaller scan emitted the existing optional `onnxruntime.quantization` warning because package `onnx` is not installed; build and smoke tests still completed successfully.

## Real before / after measurements

Baseline CI run `36135278819` on `main@62a9a7e63d31c7627fed8cab22279ed7c1a7c950`:

- **80 passed in 0.95s**;
- synthetic tracker benchmark: **1413.9 FPS**, 24 active tracks.

Repository comparison confirms the path from that baseline to original shared base `573087c` changed documentation/release-workflow files only and did not change `pc/` source.

Current implementation CI run `36141844051`:

- **98 passed in 1.02s**;
- synthetic tracker benchmark: **1304.8 FPS**, 24 active tracks;
- identity probes: **0 crossing switches / 0 reappearance switches**.

The short synthetic FPS sample is runner-sensitive; 1304.8 vs 1413.9 is therefore recorded as observed data, not claimed as a statistically meaningful performance regression. The identity probes did not exist in the baseline run, so there is no fabricated numeric before-value for them.

## Known limitations / unverified claims

- No representative real annotated occlusion/reappearance corpus has been run; dormant thresholds are engineering defaults.
- Dormant reactivation currently considers only unused **high-confidence** detections; low-confidence evidence does not revive a dormant ID.
- Dormant geometry follows camera motion, but target self-motion is not extrapolated while dormant; fast long-gap motion can exceed the spatial gate.
- Visually similar same-class objects can still hijack a dormant ID despite conservative appearance/shape/size gates.
- Live association remains greedy; no claim is made that it matches Hungarian/Kalman/OC-SORT/BoT-SORT quality.
- Cross-video `same_object_score` is not a calibrated probability.
- `global_object_id` is a graph-run grouping ID, not persistent biometric identity and not a named-person identifier.
- Hand-built Re-ID cues can fail under major viewpoint/lighting/occlusion changes.
- No learned Re-ID model was added.
- No real DirectML/GPU Re-ID quality/performance benchmark or representative CCTV benchmark was run.

## Integration hotspots

PR #5 / enhancement overlaps directly in:

- `pc/spectratrack/tracker.py`
- `pc/tests/test_tracker.py`

Its `creation_thresholds` changes the same `MultiObjectTracker.update()` detection-filter/create path. Integration must preserve both behaviors: class-specific weak-track creation and this branch's dormant expiry/reactivation. In particular, do not restore this branch's simple `score >= low_conf` prefilter if PR #5's lower class-specific creation floor is accepted.

PR #1 / detection overlaps in:

- `pc/spectratrack/batch.py`
- `pc/tests/test_batch_integration.py`

Its people-recall detector dispatch is logically independent from this branch's tracklet feature accumulation and both need to survive the merge.

PR #1 and PR #4 also touch shared root architecture/task docs. Resolve documentation text rather than dropping either subsystem's factual state.

Old PR #2 / `agent/tracking` contains the earlier equivalent tracking history and must not be merged in addition to PR #6.

## Handoff checklist

- validated implementation HEAD: `0379da5e497a10afe8a3462895fcca46185cfe59`;
- commits owned: listed above;
- tests run: full Windows PC CI listed above;
- ID-switch evidence: deterministic crossing/reappearance probes both 0 on the validated implementation;
- unverified items: real-video calibration, learned Re-ID, real GPU/CCTV quality benchmarks;
- conflicts/overlaps: PR #5 tracker lifecycle; PR #1 batch path; PR #1/#4 shared docs; obsolete duplicate PR #2;
- current synchronized pre-final-metadata HEAD: `a4d1aaa000b17fd78d001808eaaf70142c05412d`;
- ready for integration: **yes, subject to preserving the noted overlaps during integration and final CI on the final metadata commit**.
