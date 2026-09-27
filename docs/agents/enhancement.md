# Enhancement / tiny-object agent state

## Ownership

Branch: `agent/enhancement-clean`

Primary scope:

- frame quality estimation;
- adaptive non-generative preprocessing;
- low-light/denoise/deblock/mild sharpen routing;
- tiny-object/tiled processing where enhancement is required;
- raw-frame corroboration of enhancement-originated candidates;
- enhancement tests.

Normally outside scope:

- general tracker architecture;
- Re-ID;
- detector tile-grid/NMS/class-threshold policy;
- broad performance refactors;
- QA framework ownership.

## Current objective

Provide integration-ready enhancement primitives for difficult small/poor-quality person footage without allowing enhancement-only structure to become accepted detector evidence.

## Important invariants

- Enhancement is not new ground-truth evidence.
- An enhanced-only person candidate is not accepted without raw-frame detector corroboration.
- Normal/well-lit regions are not forced through processing when the quality router selects no operation.
- Processing preserves image shape/type.
- Detector-owned tile geometry and final merge/NMS policy are not duplicated here.

## Corrected branch/base state

Historical implementation started from `2012eaae2f4ffe820a66d12e40346d911616cd03` and later diverged from the shared integration history.

The cleaned branch was created directly from:

- base branch: `integration`
- base commit: `ee76f4e0c9728af9f94a363c3cee682130599521`
- branch: `agent/enhancement-clean`

Tested code HEAD before this documentation-only handoff commit:

`5d35fe0fa099f5788c87b1d6023afe36176682cc`

The current branch HEAD after the handoff documentation commit is the commit containing this file; its exact SHA is reported in the final handoff.

## Implemented changes

### Frame quality assessment

`pc/spectratrack/enhance.py` now estimates normalized routing heuristics for:

- blur;
- darkness;
- compression/blocking;
- low resolution;
- noise.

These are heuristics, not calibrated perceptual-quality scores.

### Adaptive non-generative preprocessing

`adaptive_analysis_frame()` conditionally applies:

- low-light gamma lift + CLAHE;
- bilateral denoise/deblock;
- mild sharpening when structure/noise/compression gates permit it.

Whole-frame upscaling and generative restoration are not used in accepted detector evidence.

### Enhancement-aware tiny-person regions

New `pc/spectratrack/enhancement_recall.py` preserves the enhancement-specific tiny-person processing without duplicating detector policy.

It accepts detector-owned tile/region coordinates and provides:

- adaptive processing for each supplied region;
- raw region retention;
- tile-local to full-frame detection translation;
- weak raw-probe + enhanced candidate corroboration;
- a detector-callback helper for enhancement-aware tiny-person region processing.

Tile generation, overlap policy, class thresholds and final NMS/merge remain detector-owned.

### Raw-frame corroboration

Enhanced detections are returned only when same-class raw detector evidence spatially corroborates them. Enhanced-only detections are rejected by the helper.

This reduces enhancement-hallucination risk but does not prove that a corroborated raw detector candidate is a true positive.

## Files changed

- `pc/spectratrack/enhance.py`
- `pc/spectratrack/enhancement_recall.py`
- `pc/tests/test_enhance.py`
- `pc/tests/test_enhancement_recall.py`
- `docs/agents/enhancement.md` (handoff state only)

No changes remain in:

- `pc/spectratrack/tracker.py`
- `pc/tests/test_tracker.py`
- `pc/spectratrack/app.py`
- `pc/spectratrack/runtime_config.py`
- other agents' role files.

## Tests actually run

GitHub Actions: SpectraTrack PC CI run #104, workflow run `36144462778`, on code HEAD `5d35fe0fa099f5788c87b1d6023afe36176682cc`.

Results:

- `ruff check spectratrack tests`: success, "All checks passed!"
- `python -m compileall -q spectratrack tests` + `pytest -q`: success, `102 passed in 0.97s`
- `python -m spectratrack.benchmark --frames 500 --targets 24`: success
  - frames: 500
  - targets: 24
  - observations: 11970
  - elapsed: 0.364 s
  - tracker_fps: 1372.1
  - active_tracks: 24
- `python -m spectratrack.diagnostics`: success
- `python -m spectratrack.selfcheck`: success
- PyInstaller standalone Windows build: success
- `SpectraTrack-PC.exe --help`: success
- `SpectraTrack-PC.exe batch --help`: success
- source/Windows package creation and artifact upload: success

The synthetic tracker benchmark is a CI smoke measurement, not evidence that enhancement improves person recall.

## Real before/after measurements

None.

No representative annotated real CCTV corpus/model run was available on this branch, so no real baseline-vs-enhancement person recall, precision, false-positive rate, detector latency, or quality delta is claimed.

The repository QA harness exists, but real poor-quality/high-angle/night/compressed validation footage still needs to be collected and evaluated before tuning these heuristics from evidence.

## Raw-vs-enhanced validation

Automated tests verify:

- dark/low-resolution quality routing;
- shape/dtype preservation;
- no forced preprocessing on a well-lit textured frame;
- invalid frame/region rejection;
- region-local detection coordinate translation;
- enhanced person candidates require same-class spatial raw corroboration;
- an uncorroborated enhanced candidate is excluded;
- a strong raw person candidate remains accepted;
- a weak raw probe can corroborate an enhanced candidate;
- invalid probe/acceptance threshold ordering is rejected.

These are synthetic unit tests, not real detector-quality measurements.

## Tracker changes

None remain in the cleaned branch.

The historical enhancement branch modified `tracker.py` to lower class-specific tentative-track creation thresholds. That policy was deliberately dropped during cleanup because tracker architecture belongs to `agent/tracking` and the current tracking agent has independent tracker work.

No change to `agent/tracking` or `agent/tracking-integration` was made.

## Cross-agent overlaps / conflicts

### `agent/tracking` / `agent/tracking-integration`

- Historical overlap existed in `tracker.py` and `test_tracker.py`.
- Clean branch file-level overlap: none.
- Tracker semantics are handed back to the tracking agent.

### `agent/detection`

The detection branch independently implements tile-grid generation, people-recall thresholds, detector runtime wiring and final merge/NMS.

To avoid duplicating that logic, this branch no longer owns tile-grid generation or NMS. `enhancement_recall.py` consumes detector-owned regions and a detector callback.

Current clean-branch file-level overlap with the observed detection branch: none.

Integration still requires the detection path to call these enhancement primitives if adaptive raw/enhanced corroboration is desired at runtime.

### `agent/performance` / `agent/qa`

No current file-level overlap in this cleaned diff.

## Known limitations / unverified claims

- Quality thresholds are engineering heuristics and are not dataset-calibrated.
- Compression/noise/blur routing has not been validated on a representative real CCTV corpus.
- Raw corroboration lowers the risk of enhancement-only hallucination but cannot eliminate repeatable raw detector false positives.
- Final duplicate suppression/merge is intentionally detector-owned.
- The cleaned enhancement branch does not wire a user-facing runtime mode by itself; runtime integration belongs with the detector path.
- No real recall/precision or enhancement latency before/after benchmark has been run.
- Neural/generative restoration remains excluded from accepted detection evidence.

## Commits owned on the clean branch

- `f7e542f5430b4174df5dabce69366c688c0b9861` — Preserve adaptive enhancement recall primitives
- `5d35fe0fa099f5788c87b1d6023afe36176682cc` — Delegate tile geometry to detector policy
- final handoff documentation commit — `docs/agents/enhancement.md` only (current branch HEAD; exact SHA in final handoff)

Historical commits on old enhancement branches were not force-rewritten or merged into this clean history.

## Readiness for integration

**Yes, as enhancement primitives.**

The branch is cleanly based on current `integration`, has no tracker/detection file-level conflicts in the observed agent branches, and passed the full PC CI workflow.

For user-visible people-recall behavior, the detector integration should explicitly consume `enhancement_recall.py`; that wiring remains detector-owned and is not claimed complete here.
