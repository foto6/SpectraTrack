# Performance / pipeline agent state

## Ownership

Branch: `agent/performance`

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
- GPU/VRAM values must remain unknown when no trustworthy measurement source exists.
- Old runtime-config/profile names should remain compatible where practical.

## Current branch state

Original shared base: `integration @ 573087cdf60c1a00bbc93b8764733b2ade3cbdd8`

Observed HEAD when this file was created: `113c1d8d91922da429cd0034304ca2549380b456`

Observed performance work includes:

- pipeline profiling baseline;
- process CPU sampler;
- machine-readable performance report;
- normalized performance profiles;
- detector preprocess/inference/postprocess timing.

The branch also currently contains QA benchmark work through a merge/foreign commit. That QA code is not performance-agent ownership and should be removed from the final performance-only history/diff without modifying the QA branch.

## Required cleanup before integration

Keep performance-owned commits/changes only.

Do not carry `qa_benchmark.py`, new-FN gates, or QA benchmark tests as performance-owned work.

Do not start a large async/TensorRT/concurrency refactor until measured bottlenecks justify it.

## Integration hotspots

Likely overlap with detection/enhancement in:

- `pc/spectratrack/app.py`
- `pc/spectratrack/detector.py`
- `pc/spectratrack/runtime_config.py`

## Handoff checklist

Update before final handoff:

- cleaned HEAD SHA:
- commits owned:
- tests run:
- baseline measurements:
- actual bottleneck findings:
- unverified items:
- conflicts/overlaps:
- ready for integration: yes/no
