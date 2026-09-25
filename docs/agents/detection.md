# Detection agent state

## Ownership

Branch: `agent/detection`

Primary scope:

- detector inference and decode;
- high-recall person detection;
- tiled/sliced inference;
- class-specific detector thresholds;
- detector call paths in live and batch modes only as required.

Normally outside scope:

- tracker association/Re-ID;
- broad preprocessing architecture;
- performance pipeline refactors;
- QA framework ownership.

## Current objective

Reduce false negatives for small/distant/poor-quality people without silently degrading standard detector behavior.

## Important invariants

- Standard detector mode remains the default and keeps the existing `Detection[]` contract.
- Tile boxes map back to full-frame coordinates.
- Overlapping tile/full-frame detections are merged class-wise.
- Batch and live detector behavior use the same opt-in people-recall path.
- No recall/precision/FPS/VRAM improvement is claimed without real representative measurements.

## Base / branch state

Canonical coordination files were read from:

- `origin/integration @ ee76f4e0c9728af9f94a363c3cee682130599521`

Original shared base for this branch:

- `integration @ 573087cdf60c1a00bbc93b8764733b2ade3cbdd8`

Branch:

- `agent/detection`

Reviewed implementation HEAD before this handoff-file-only commit:

- `4b59cb68b581df6ee798491a816c91f41cb651fd`

The final handoff commit changes only this state file; the exact final branch HEAD is therefore reported in the handoff report/PR metadata rather than pretending this file can embed its own commit SHA.

## Final implemented changes

- added class-specific confidence threshold support without changing the standard default threshold;
- added opt-in `detect_people_recall()`;
- people-recall runs one full-frame pass plus overlapping source-image tiles;
- only person detections are accepted from tile passes;
- tile detections are offset back into full-frame coordinates;
- combined detections use existing class-aware hard NMS;
- added live CLI/config dispatch through `--detector-mode people-recall`;
- added batch dispatch for the same mode;
- added `people-recall.json` preset;
- added validation so programmatic batch callers cannot silently fall back to standard mode on an unknown detector mode;
- kept standard mode as the default;
- removed the branch-local detector benchmark harness after `integration` introduced the canonical `qa_benchmark.py`, avoiding two independent metric implementations;
- removed earlier routine edits to shared root coordination docs from the final net diff.

## Final net files changed

- `pc/README.md`
- `pc/presets/people-recall.json`
- `pc/spectratrack/app.py`
- `pc/spectratrack/batch.py`
- `pc/spectratrack/detector.py`
- `pc/spectratrack/runtime_config.py`
- `pc/tests/test_batch_integration.py`
- `pc/tests/test_detector_decode.py`
- `pc/tests/test_runtime_config.py`
- `docs/agents/detection.md` (handoff state only)

## Commits owned

Implementation/history commits:

- `10dc3415152d71e7ca37d65dcccb780f98f8f54a` — Add tiled high-recall person detector path
- `9ed2a4e96757910d70e54e742e4f75c73eb9c84b` — Expose people-recall mode in PC workflows
- `7143ebcb20d2de35d2ab196dbe61912033febfed` — Add measurable person-detection benchmark harness
- `89eb25641974742a38e0dd92fb65231025f8f699` — Document people-recall detector contract and limits
- `75d09623ec38e59e209e169e479d6f9067509e7b` — Record actual detector input shape in benchmark
- `9a21c495baa258c55d74ddc6a4cdde6c88e2867a` — Reuse detector IoU in recall benchmark
- `4b59cb68b581df6ee798491a816c91f41cb651fd` — Prepare detection branch for integration

The benchmark/documentation work from commits `7143ebc`, `89eb256`, `75d0962`, and `9a21c49` was intentionally cleaned out of the final net diff where it duplicated the canonical QA harness or shared coordination docs. History is preserved; no force-push/rewrite was used.

## Tests actually run

GitHub Actions PC CI run `36144251342` on `4b59cb68b581df6ee798491a816c91f41cb651fd`: **success**.

Observed results from that run:

- `ruff check spectratrack tests` — success, `All checks passed!`;
- compile + pytest — **96 passed in 1.96s**;
- `python -m spectratrack.benchmark --frames 500 --targets 24` — success;
- diagnostics — success; ONNX Runtime exposed `DmlExecutionProvider,CPUExecutionProvider`, `directml=yes`;
- self-check — `SELF_CHECK=PASS`;
- PyInstaller standalone Windows build — success;
- standalone `SpectraTrack-PC.exe --help` — success;
- standalone `SpectraTrack-PC.exe batch --help` — success;
- PC source / Windows package creation and artifact uploads — success.

Non-blocking build warning observed:

- PyInstaller could not collect optional `onnxruntime.quantization` because Python package `onnx` was not installed. The standalone build and smoke tests still completed successfully.

## Real before/after benchmark data

None.

There is no collected/validated representative annotated high-angle/night/compressed pedestrian corpus in the repository and no real model/footage benchmark was run for this branch. Therefore this branch makes **no** measured claim for:

- person recall;
- precision;
- false positives/frame;
- people-recall latency/FPS;
- DirectML VRAM.

The synthetic benchmark in CI is a smoke/regression check, not evidence of detector quality improvement.

## Unverified / known limitations

- `people-recall` defaults (`person_conf=0.12`, tile size `640`, overlap `0.20`, merge IoU `0.55`) are engineering defaults, not calibrated optima.
- tiled mode adds multiple inference calls per detector frame; real DirectML cost is unmeasured.
- supported end-to-end ONNX exports may suppress weak boxes internally before external class thresholds can see them.
- merge remains hard class-aware NMS; Soft-NMS/WBF were not added without benchmark evidence.
- tracker `high_conf=0.45` still gates creation of new IDs; weak detections can help existing tracks but do not create stable new tracks by themselves. That boundary belongs to tracking/temporal-confirmation work.
- the canonical `pc/spectratrack/qa_benchmark.py` currently executes the standard detector path only. Extending it to run people-recall should be coordinated with QA rather than reintroducing a second benchmark implementation.
- the combined tree with the latest `integration @ ee76f4e...` was not executed locally as a merged checkout. The final detection net diff does not overlap the files newly added to integration by the coordination/QA update, and GitHub currently reports the PR mergeable, but full post-integration CI remains an integrator check.

## Cross-agent overlaps / conflicts

### `agent/enhancement @ 9a9d6a65fc0487753b23c5d3d35f661f60c0d51b`

Direct semantic overlap exists in:

- `pc/spectratrack/detector.py`
- `pc/spectratrack/app.py`
- `pc/spectratrack/runtime_config.py`
- related tests

That branch independently introduced tiled/people-recall detector logic and a different CLI/config shape. These implementations must not both be merged blindly. Detection owns the detector high-recall core per `AGENTS.md`; enhancement-specific preprocessing should be reconciled around one canonical detector API.

### `agent/performance @ 113c1d8d91922da429cd0034304ca2549380b456`

Textual/behavioral overlap exists in:

- `pc/spectratrack/detector.py`
- `pc/spectratrack/app.py`
- `pc/spectratrack/runtime_config.py`
- detector/runtime tests and presets

Performance adds detector timing/profile/runtime changes. Integrator should preserve those measurements/profile changes while retaining the people-recall detector semantics and standard-mode compatibility.

### `agent/tracking @ ccf2b14e1606049a313d3471fc9e762723b1e095`

Overlap exists in:

- `pc/spectratrack/batch.py`
- `pc/tests/test_batch_integration.py`

Integrator should preserve tracking branch association/batch behavior and this branch's explicit detector-mode dispatch/validation.

### `agent/qa @ 5ff444758ad48ec3d5b52bf1226f181da75c5945`

No duplicate detector benchmark remains in the final detection net diff. QA owns the canonical benchmark harness. Future people-recall benchmark dispatch belongs in coordinated QA integration, not a parallel metric implementation here.

## Handoff notes

- No changes were made to tracker association, Re-ID, CMC, appearance semantics, or `Detection`/`Track` shared types.
- No model weights were added and no silent model download was introduced.
- No other agent role file was edited.
- No agent branch was merged into `agent/detection`.
- No merge into `integration` was performed.
- No force-push/history rewrite was performed.

## Ready for integration

**Yes for detection scope, with explicit cross-agent reconciliation required.**

The branch is clean relative to its assigned scope, the final net diff no longer includes routine shared-doc edits or a duplicate QA benchmark, current branch CI is green, and GitHub reports the PR mergeable with `integration`.

The integrator still must reconcile the known enhancement/performance/tracking overlaps and then run the full merged CI on the actual integration result.
