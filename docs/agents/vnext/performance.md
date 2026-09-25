# A4 — vNext Performance / Scheduler Research

Branch:

`agent/vnext-performance`

Exact start/base:

`vnext-base @ d03af3ae6425d3ea2e4d52e25389fecc09957394`

Immutable product baseline:

`integration @ 3ebc4d50213593cac62b97399447fccf6bbc1755`

Validated research-code HEAD before this handoff-state commit:

`2c876387f26656ae807c1a9aa23d459b4aa4321b`

No specialist branch was merged into A4. `integration`, `main`, RC and other agent branches were not modified.

## Current bottleneck report

The known real-world blocker remains:

`people-recall + adaptive ~= 1 source second / ~120 processing seconds`

The existing observation does not include enough attached provenance to split that 120x cost by exact resolution, source FPS, model SHA, provider and stage. It is therefore treated as a blocker observation, not a calibrated latency sample.

The structural compute model explains why this path can become extreme: current adaptive people-recall runs one full-frame inference, probes every tile raw, then runs another inference for every tile whose quality router enables enhancement.

For a detector frame:

`ONNX calls = 1 + tile_count + enhanced_tile_count`

with `0 <= enhanced_tile_count <= tile_count`.

The highest-leverage performance target is therefore detector-call elimination/scheduling, not a first-pass 5% Python/OpenCV micro-optimization.

## Exact inference-call model

Production tile geometry is reused directly from `spectratrack.detector._tile_regions()`.

Current `tile_size=640`, `overlap=0.20`:

| Resolution | Tiles | Full-frame calls | Raw tile calls | Enhanced calls | Total ONNX calls / detector frame |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1280x720 | 6 | 1 | 6 | 0..6 | 7..13 |
| 1920x1080 | 8 | 1 | 8 | 0..8 | 9..17 |
| 2560x1440 | 15 | 1 | 15 | 0..15 | 16..31 |
| 3840x2160 | 32 | 1 | 32 | 0..32 | 33..65 |

At a 30 FPS reference and `detect_every=1`, those bounds are respectively:

- 720p: 210..390 ONNX calls/source-second;
- 1080p: 270..510;
- 1440p: 480..930;
- 4K: 990..1950.

These are exact structural call counts, not throughput claims.

### Exact 1080p production regions

For 1920x1080 / 640 / 0.20:

```text
(0,    0,   640,  640)
(512,  0,   1152, 640)
(1024, 0,   1664, 640)
(1280, 0,   1920, 640)
(0,    440, 640,  1080)
(512,  440, 1152, 1080)
(1024, 440, 1664, 1080)
(1280, 440, 1920, 1080)
```

A4 regression tests call the actual production `detect_people_recall()` control path with a fake ONNX session and check `last_inference_calls` against the theoretical model:

- adaptive clean 1080p / no selected enhancement: 9 calls;
- adaptive dark 1080p / every tile enhanced: 17 calls;
- people-recall enhancement-off at 720p/1080p/1440p/4K: 7/9/16/33 calls.

This validates control-flow counters, not real inference latency.

## Stage measurement status

Production `--perf-report` already exposes:

- capture/decode;
- CMC;
- detector policy wall time;
- detector preprocess;
- ONNX inference;
- detector postprocess;
- appearance;
- tracker;
- HUD/rendering;
- session write;
- video write;
- whole-frame time;
- actual ONNX call count;
- process CPU;
- provider list.

The current production report does not split adaptive quality assessment from adaptive operation preprocessing inside people-recall.

A3 now has isolated tooling on `agent/vnext-enhancement` for quality assessment, raw probe, operation preprocessing, enhanced inference and call counts, but its current role handoff publishes no measured result artifact yet. A4 therefore does not invent A3 costs.

Target-PC values still missing:

- real ms/frame by stage;
- actual source FPS and processing FPS from the blocker clip;
- provider for that exact blocker run;
- process CPU for that exact run;
- trustworthy GPU utilization / VRAM.

The A4 `analyze-report` tool derives:

- processing FPS;
- processing seconds/source second;
- ONNX calls/frame;
- ONNX calls/source second

from a real production perf report plus known source FPS.

## Experimental scheduler candidates

The simulator is research-only; production behavior is unchanged.

### CURRENT

- full-frame discovery every detector frame;
- all raw tiles;
- every enhancement-eligible tile;
- no hard call budget.

### COARSE_TO_FINE

- full-frame global discovery every detector frame;
- expensive raw follow-up only for suspect ROIs;
- bounded enhanced follow-up slot;
- new-person discovery remains global every detector cadence.

### TRACK_GUIDED

- known-track ROIs plus suspect ROIs;
- periodic full-frame rediscovery;
- immediate full-frame rescan on scene-change or camera-motion trigger, even on an otherwise cadence-skipped frame;
- no enhanced calls in this candidate.

### BUDGETED_ADAPTIVE

- hard max ONNX calls/frame;
- hard max enhanced calls/frame;
- known-track and suspect ROI work share the raw budget without starving either when budget permits;
- bounded enhanced slot reserved only for enhancement-eligible suspect work;
- periodic global rediscovery;
- immediate scene/camera rescan;
- temporal reuse on non-detector frames.

## CURRENT vs scheduler candidates

Normalized 1080p compute simulation:

- 300 source frames;
- 30 FPS reference;
- `detect_every=1`;
- 8 production tiles.

| Policy | Explicit research signal/config | Avg calls/frame | Max calls/frame | Calls/source-second | Global discovery bound |
| --- | --- | ---: | ---: | ---: | ---: |
| CURRENT min | 0 enhanced tiles | 9.000 | 9 | 270 | 1 frame |
| CURRENT max | 8 enhanced tiles | 17.000 | 17 | 510 | 1 frame |
| COARSE_TO_FINE | 4 suspect ROIs, 1 enhanced, max calls 6 | 6.000 | 6 | 180 | 1 frame |
| TRACK_GUIDED | 2 track ROIs, 1 suspect, max calls 4, period 15 | 3.067 | 4 | 92 | 15 frames / 0.50 s @30 FPS |
| BUDGETED_ADAPTIVE | 2 tracks, 2 suspects, 1 enhanced, max calls 4, period 10 | 4.000 | 4 | 120 | 10 frames / 0.33 s @30 FPS |

This table is compute simulation only. Wall-time speedup is not assumed proportional to call-count reduction.

## Quality / cost table

No frozen, human-confirmed A5 corpus revision exists in the currently published A5 handoff.

Therefore quality fields cannot yet be filled honestly:

| Policy | Recall impact | FN delta | Real processing s/source-s | New-person discovery latency |
| --- | --- | --- | --- | --- |
| CURRENT | pending A5 baseline | pending | one field observation ~120x, provenance incomplete | global every detector frame |
| COARSE_TO_FINE | pending A5 | pending | pending target run | <= detector cadence |
| TRACK_GUIDED | pending A5 | pending | pending target run | <= periodic bound; earlier on trigger |
| BUDGETED_ADAPTIVE | pending A5 | pending | pending target run | <= periodic bound; earlier on trigger |

There is not yet a valid QUALITY-vs-COMPUTE Pareto frontier. The compute axis is ready; the quality axis must wait for the frozen A5 corpus.

A1 and A2 current role handoffs likewise publish no candidate runtime-cost result yet.

## DirectML batching result

Research-only `batch-probe` was implemented. It inspects the actual ONNX input shape/provider and benchmarks supported batch sizes with warmup/repeats.

It records:

- provider;
- actual input shape;
- fixed/dynamic batch support;
- median latency;
- items/second;
- supported/unsupported batch sizes.

It deliberately leaves VRAM unknown without trustworthy telemetry.

Result for this research cycle so far:

`NOT MEASURED`

Reason: the repository does not contain the exact production ONNX model binary, and this branch has no target RX 5700 XT DirectML run. Running a synthetic model or hosted-runner DirectML and calling it representative would violate the research contract.

No batching implementation is recommended without this measurement.

## Recommended scheduler

Recommended **next research candidate**, not production change:

`BUDGETED_ADAPTIVE`

Reason:

- hard ONNX-call ceiling directly attacks the structural blocker;
- bounded enhanced work;
- retains both known-track refresh and uncertainty/suspect work;
- periodic global rediscovery prevents permanent ROI blindness;
- scene/camera triggers force immediate rediscovery;
- compute budget is explicit and testable.

`TRACK_GUIDED` remains the lower-call comparator and may become preferable if A5 shows no recall/discovery regression.

No candidate is ready for production until A5 quality evidence exists.

## Discovery guarantees

Critical invariant: ROI optimization must not permanently hide a new person outside existing tracks.

Research guarantees:

- CURRENT: global scan every detector frame;
- COARSE_TO_FINE: global scan every detector frame;
- TRACK_GUIDED: periodic global scan + immediate scene/camera trigger;
- BUDGETED_ADAPTIVE: periodic global scan + immediate scene/camera trigger.

For TRACK_GUIDED/BUDGETED_ADAPTIVE, the periodic bound is:

`detect_every * global_period` source frames.

Triggers override detector cadence in the simulator.

## Risks

- full-frame discovery may itself miss a tiny person, so a bounded global-scan schedule is a discovery guarantee for the detector policy, not a recall guarantee;
- track/suspect ROI generation quality is not yet measured;
- a hard budget can defer lower-priority ROIs;
- detector fusion semantics remain A1-owned and are not modeled here;
- enhancement activation/cost remains A3-owned;
- tracker association remains A2-owned;
- quality/FN comparison remains A5-owned;
- no target GPU timing means call-count reduction cannot yet be translated into wall-time reduction;
- scene/camera triggers need robust signals before production use.

## Rejected / deferred optimizations

Rejected as first moves:

- micro-optimizing OpenCV/Python before reducing redundant detector calls;
- unconditional batching based on CUDA assumptions;
- ROI-only inference with no periodic/triggered global discovery;
- unlimited enhancement on every tile;
- fabricating GPU/VRAM measurements;
- projecting candidate wall time by blindly scaling the ~120x field observation.

Deferred until benchmark evidence:

- DirectML batching;
- mixed precision changes;
- TensorRT/CUDA-specific paths;
- pinned-memory tuning;
- async decode;
- hardware decode;
- queue/pipeline concurrency.

These may matter later, but they are not justified before the detector-call budget is measured on target hardware.

## A5 corpus revision

Current published A5 role state contains no frozen corpus revision identifier yet.

A4 quality rerun status:

`BLOCKED ON A5 FREEZE`

A4 does not substitute synthetic fixtures for CCTV quality evidence.

## Files added by A4

- `pc/spectratrack/vnext_performance.py`
- `pc/tests/test_vnext_performance.py`
- `pc/benchmarks/vnext/performance/README.md`
- this role-state file

No production app/detector/enhancement/tracker file is changed.

## Validation status

GitHub PR #13 targets `vnext-base` only for CI/review and must not be merged as part of this handoff.

Latest code validation is recorded against:

`2c876387f26656ae807c1a9aa23d459b4aa4321b`

GitHub Actions PC CI:

- workflow run: `36166015554`
- job: `108173819930`
- conclusion: **success**
- `ruff check spectratrack tests`: success, `All checks passed!`
- `python -m compileall -q spectratrack tests` + `pytest -q`: **162 passed in 2.67s**
- synthetic tracker benchmark: `500 frames / 24 targets / 11970 observations`
- synthetic benchmark elapsed: `0.363 s`
- synthetic tracker throughput: `1377.6 tracker_fps`
- diagnostics: success; `ort_available=DmlExecutionProvider,CPUExecutionProvider`, `directml=yes`
- self-check: success
- PyInstaller standalone Windows build: success
- standalone `SpectraTrack-PC.exe --help`: success
- standalone `SpectraTrack-PC.exe batch --help`: success
- source/app packaging and artifact uploads: success

The hosted Windows runner exposed DirectML, but its throughput is not RX 5700 XT evidence. The synthetic tracker benchmark does not measure the people-recall detector path.

The subsequent handoff documentation commits do not modify validated PC code.

## Readiness

Research tooling / compute-model handoff: **ready**.

Production scheduler integration: **not ready**.

Required before production candidacy:

1. frozen A5 corpus revision;
2. same-corpus quality/FN/discovery-latency results for scheduler candidates;
3. target-PC production perf report with model/video/provider provenance;
4. A3 measured enhancement cost artifact;
5. real DirectML batching probe if batching remains under consideration.
