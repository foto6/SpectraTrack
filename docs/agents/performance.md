# Performance / pipeline agent state

## Ownership

Branch: `agent/performance-clean`

This clean branch supersedes the old `agent/performance` history for integration purposes. The old branch contained a foreign QA merge; no QA branch/history was modified while cleaning it.

Primary scope:

- per-stage profiling;
- detector preprocess/inference/postprocess timing;
- FPS/latency/CPU reporting;
- performance profiles;
- capture/runtime performance measurements;
- performance-regression evidence.

Normally outside scope:

- QA benchmark framework ownership;
- detector quality strategy;
- tracker/Re-ID feature work;
- enhancement algorithms.

## Current objective

Make bottlenecks measurable before introducing larger optimization/concurrency changes.

## Important invariants

- Profiling must not change detection/tracking results.
- Performance improvements must not silently reduce recall.
- GPU/VRAM values remain unknown when no trustworthy measurement source exists.
- Legacy runtime profile names remain accepted.

## Branch / base state

Original performance-work base: `integration @ 573087cdf60c1a00bbc93b8764733b2ade3cbdd8`.

Clean integration base: `integration @ ee76f4e0c9728af9f94a363c3cee682130599521`.

Validated code HEAD before this handoff-state commit:
`83290bdbaaaa0815c7a751056a3cf9a2b9c34366`.

Draft PR for the clean branch: #8.

Old PR #4 was closed because its history contained QA work that belongs to `agent/qa`.

## Implemented changes

- `StageTimer` now keeps bounded recent samples plus whole-run count/average/max summaries.
- Added dependency-free process CPU sampling normalized to total logical CPU capacity.
- Added optional machine-readable `--perf-report` JSON with processing FPS, provider list, capture stats, stage latency summaries and process CPU data.
- GPU utilization / VRAM fields remain explicit `null` values with an unavailable reason rather than inferred values.
- Detector timing is split into preprocess, ONNX inference and postprocess without changing detector outputs.
- Live pipeline separately records detector, appearance, tracker, capture, HUD, session-write, record-write and full-frame timing.
- Canonical performance profiles are `fast`, `balanced`, `high-quality`, and `max-recall`; legacy `speed` / `quality` aliases remain accepted.
- Existing preset files were deliberately left unchanged after review because rewriting legacy profile names was unnecessary churn.
- Added performance-focused tests for whole-run timer statistics, CPU reporting/normalization, profile aliases, config validation and detector stage timing.

## Net files changed vs current integration

- `pc/spectratrack/app.py`
- `pc/spectratrack/detector.py`
- `pc/spectratrack/metrics.py`
- `pc/spectratrack/runtime_config.py`
- `pc/tests/test_detector_decode.py`
- `pc/tests/test_metrics.py`
- `pc/tests/test_runtime_config.py`
- `docs/agents/performance.md` in this handoff commit only

No `qa_benchmark.py`, new-FN regression-gate code, QA benchmark tests, or QA role files are part of the performance diff.

## Commits owned on the clean branch

- `0a91b8dd53510dd69f77b9e5fbe7654d3b43bcb2` — perf: add pipeline profiling and stage timings
- `a5ae712391ca7f2c89f402fcc07d31d4d70ef5fa` — perf: remove unnecessary preset churn
- `b94bdcbe0dbb0b7726435b47628803689cec6d95` — perf: remove unnecessary preset churn
- `83290bdbaaaa0815c7a751056a3cf9a2b9c34366` — test: cover process CPU normalization

The two preset-cleanup commits have zero net preset diff versus `integration`; they document the self-review cleanup without rewriting history.

Historical source commits on the superseded old branch were `ee8acb29742b0da85620a2875ba83d6004deaccb` and `f86a78ddd2ff57d88c3a5a00e83a44f02afe27ee`. They are not the integration target.

## Tests actually run

GitHub Actions run `36144206512`, job `108101148857`, against code HEAD `83290bdbaaaa0815c7a751056a3cf9a2b9c34366`:

- dependency install: success;
- `ruff check spectratrack tests`: success;
- `python -m compileall -q spectratrack tests`: success;
- `pytest -q`: **93 passed in 1.03s**;
- `python -m spectratrack.benchmark --frames 500 --targets 24`: success;
  - frames=500;
  - targets=24;
  - observations=11970;
  - elapsed=0.278s;
  - tracker_fps=1798.4;
  - active_tracks=24;
- diagnostics: success;
  - `ort_available=DmlExecutionProvider,CPUExecutionProvider`;
  - `directml=yes`;
- self-check: success;
- PyInstaller standalone Windows build: success;
- `SpectraTrack-PC.exe --help`: success;
- `SpectraTrack-PC.exe batch --help`: success;
- source/app packaging and artifact upload: success.

## Measurements / before-after

There is **no valid real pipeline before/after performance result yet**.

The CI tracker benchmark above is a smoke/performance gate for tracker code, not a detector/DirectML pipeline benchmark. This branch does not change tracker logic, and a single hosted-runner number must not be presented as a speedup.

No fixed representative model + video + hardware workload was run before and after this performance diff. Therefore no detector FPS, DirectML utilization, VRAM, decode throughput, or end-to-end latency improvement is claimed.

The concrete before/after capability change is measurement quality:

- before: the aggregate `detect` timing mixed detector work with appearance and included zero-valued skipped frames;
- after: actual detector runs are counted separately and detector preprocess / inference / postprocess, appearance and other pipeline stages have separate summaries.

## Known limitations / unverified items

- No representative target-PC run with a fixed ONNX model/video was available, so no actual bottleneck ranking has been established.
- No validated portable DirectML per-process GPU utilization or VRAM source exists in this implementation; report fields intentionally remain unknown.
- Process CPU percentage is normalized to total logical CPU capacity; it is not per-core utilization and may not reflect CPU-affinity restrictions.
- `p95_recent` is computed from the bounded recent sample window, while average/count/max are whole-run values.
- `detect` remains a legacy amortized aggregate metric; `detector` and `detector_inference` are the appropriate fields for actual detector-run latency.
- Session event writes outside the timed `recorder.frame()` call are not individually attributed to `session_write`.
- No real-model end-to-end invocation of `--perf-report` was run in CI; CLI/build/tests validate the code path structurally, not representative GPU timings.
- `max-recall` currently uses detector-every-frame cadence like `high-quality`; it does not itself add a higher-recall detector algorithm.

## Cross-agent overlaps / conflicts

No foreign agent branch was merged into this clean branch.

Likely integration overlaps:

- `agent/detection` changes `pc/spectratrack/app.py`, `detector.py`, `runtime_config.py` and related tests for people-recall/tiled detection.
- `agent/enhancement` changes the same three runtime files for enhancement-driven people-recall/tiled paths.
- Their tiled/multi-pass detector implementations introduce additional low-level inference calls. When either branch is integrated, performance timing should be reconciled into the shared low-level detector inference path so tile passes are measured rather than bypassing stage timing.
- `agent/tracking` does not currently overlap the net performance file set.
- `agent/qa` owns `qa_benchmark.py`, new-FN gates and QA benchmark tests; those files are intentionally absent from this branch.

Current PR #8 is mergeable against `integration @ ee76f4e0c9728af9f94a363c3cee682130599521`. Detection/enhancement may create textual or semantic conflicts later if they are merged first; resolve those at integration rather than copying their work into this branch.

## Readiness

Ready for integration against the current `integration` baseline: **yes**.

Recommended integration order/review note: if detection or enhancement lands first, manually reconcile the shared detector/app/runtime-config hunks and preserve both their quality path and these profiling hooks. Do not resolve by dropping tiled inference or by timing only the old single-pass `detect()` wrapper.
