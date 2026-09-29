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

## Round 2 supplement — target RX 5700 XT scheduler probe

Round-2 target-PC execution-cost probing was run on the user's AMD Radeon RX 5700 XT using DirectML with the exact `yolo11x.onnx` model SHA-256 `e84cbad768b218d74ecc85e3e52d84631123719a6951b3ddf6eddc850d5b3f73`, input size 960, person threshold 0.12, and the real 1920x1080 `12345.mp4` test clip (125 frames @ 25 FPS).

The target-probe CLI had one wiring bug discovered by this run: `--source-commit` was parsed but not forwarded to `target_scheduler_probe()`. The round-2 change fixes that before measurement.

Measured execution-cost results:

| detector_every | max calls / detector frame | global period | ONNX calls/source-s | processing s/source-s | median call ms | p95 call ms | periodic global bound |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 1 | 12 | 12.6 | 1.495 | 114.3 | 134.7 | 0.96 s |
| 3 | 1 | 8 | 8.4 | 0.994 | 114.6 | 137.6 | 0.96 s |
| 4 | 1 | 6 | 6.4 | 0.755 | 113.9 | 135.5 | 0.96 s |
| 4 | 2 | 6 | 12.8 | 1.496 | 112.8 | 133.1 | 0.96 s |
| 5 | 2 | 5 | 10.0 | 1.165 | 112.8 | 135.2 | 1.00 s |
| 5 | 3 | 5 | 15.0 | 1.751 | 112.9 | 134.6 | 1.00 s |

All runs reported `DmlExecutionProvider` with CPU fallback present in provider priority. These probes intentionally use deterministic rotating ROI tiles and are **execution-cost-only**; they do not establish scheduler quality or recall.

Round-2 interpretation:

- the <=15 calls/source-second research target is achievable on the measured 1080p clip;
- the <=2 processing-seconds/source-second research target is also achievable for every measured bounded configuration;
- one-call detector frames at `detect_every=3` and `4` measure around real-time or faster on this clip, but their quality is unknown;
- `detect_every=4, max_calls=1` is the lowest-cost measured point (6.4 calls/source-s, 0.755 processing s/source-s) while keeping a 0.96 s periodic-global bound;
- `detect_every=5, max_calls=3` remains below the 2x processing target at 15 calls/source-s and preserves a 1.0 s global bound, giving a larger ROI budget for the future quality comparison.

No production scheduler choice is made from this execution-only table. A5/A1 quality evidence is still required.


## Round 2 public-first scheduler checkpoint — 2026-09-26

Status: **BLOCKED BY QUALITY FINALISTS / NO NEW LARGE SEARCH**

Existing RX 5700 XT + DirectML execution-cost evidence remains valid as cost evidence only.

A4 must not launch a new abstract scheduler search now.

Wait for:

- A1 surviving MOT17/CrowdHuman/NightOwls fusion policy;
- A2 current-vs-ambiguity decision including DanceTrack association evidence;
- A3 enhancement ON/OFF decision including NightOwls.

Then run only 1-3 surviving scheduler configurations and produce the final quality-vs-cost Pareto evidence on the target PC.

Do not select the fastest configuration without quality evidence.

## Round 2 A4 — Smoke400 cost model and bottleneck gate — 2026-09-27

Status: **DONE / NO FULL5000 PERFORMANCE RUN PROMOTED**

This checkpoint obeys the A5 smoke gate and does not alter production thresholds, enhancement policy, frozen
corpus bytes, or the full-corpus selection.

### Frozen input consumed

A4 waited for A5 rather than selecting its own NightOwls frames.

- parent: `nightowls-public-slice5000-r1`
- parent corpus SHA-256:
  `1ba30ef5adad0f6bedba4319d1c3b5f246d4c576c92b26f8b98b68b4b94a0ae8`
- A5 triage revision: `nightowls-public-smoke400-r1`
- A5 smoke corpus SHA-256:
  `70c29ecd91ede9239ebed2949ea46e4b07b63e842aef630c0752ee41b9620162`
- smoke GT SHA-256:
  `45ba10895c8a98f433bc7bad8e488b311a31ad8acf8459656d0cd775fe3fc0e8`
- selection-proof SHA-256:
  `0838ecfd1341eb5f0193ad369516098be5a7d8b53d77302f8e2e4a4089c34940`
- 400 unique frames, 143/143 logical sequences represented, `tracking_supported=false`
- A4 selection-key SHA-256:
  `c3be95af68e0674facfe8f66f242e4dba9a86de0ac077aa7662d2e688fbf3cf9`

The A4 runner consumes A5's selection proof by source identity and rejects unresolved, duplicate, or wrong-count
selections. Candidate-result data is not an input to selection.

### Locked detector configuration

Both measured variants used identical inputs and:

- model: `yolo11x.onnx`
- model SHA-256:
  `e84cbad768b218d74ecc85e3e52d84631123719a6951b3ddf6eddc850d5b3f73`
- provider priority: `DmlExecutionProvider, CPUExecutionProvider`
- input size: 960
- person confidence: 0.12
- tile size: 640
- tile overlap: 0.20
- merge IoU: 0.55
- enhancement: **off**
- source code used for the measured pair:
  `16982bddcf6474052afbdb70ef5bf784fe288c36`

A3's frozen enhancement decision remains authoritative. A4 did not rerun adaptive enhancement and did not
silently re-enable it.

### Instrumentation added

`spectratrack.performance_smoke` records per frame:

- image decode;
- camera-motion estimation;
- detector policy wall time;
- detector preprocess, synchronous `session.run`, and postprocess;
- full-frame pass, raw ROI passes, enhanced ROI passes, enhancement-router residual, and fusion;
- appearance extraction;
- tracker update;
- canonical result serialization;
- frame wall time and process CPU time;
- system CPU idle when `psutil` is available;
- ONNX/full/raw/enhanced invocation counts;
- a synchronous-inference host-wait residual (wall minus process CPU, floored at zero).

That residual intentionally is **not** labeled transfer-only time: with DirectML, `session.run` includes
accelerator execution, host/device movement, synchronization, and provider fallback behavior. DirectML exposes
provider selection here but no trustworthy portable per-process GPU-utilization percentage, so A4 records GPU
utilization as unavailable instead of fabricating a value.

The comparison gate now also rejects mismatched providers, model hashes, frame counts, corpus hashes, selection
hashes, or detector settings.

### CONTROL — exact A5 Smoke400

Measured artifact:

`E:\SpectraTrack-data\runs\a4-nightowls-smoke400-control.json`

Results:

- frames: 400
- wall: **152.009 s**
- throughput: **2.631 frames/s**
- ONNX calls: **1200 = 3.0/frame**
  - full-frame: 400
  - raw ROI: 800
  - enhanced ROI: 0
- quality at the unchanged scorer: TP 135 / FP 126 / FN 48 / GT 183
- precision: **0.517241**
- recall: **0.737705**
- output hash:
  `e09e930084b2384d80860a33693669adb306ad38ee382c4d8e832d7574db76be`

Per-frame stage distributions:

| Stage | Mean ms | p50 ms | p95 ms | Share of measured frame total |
| --- | ---: | ---: | ---: | ---: |
| detector policy | 347.205 | 344.737 | 374.340 | 91.4% |
| ONNX inference (inside detector) | 292.085 | 289.994 | 309.827 | 76.9% |
| raw ROI policy passes (2/frame; overlaps detector internals) | 231.460 | 230.155 | 250.036 | 60.9% |
| full-frame policy pass (1/frame; overlaps detector internals) | 115.438 | 114.151 | 126.283 | 30.4% |
| detector preprocess | 30.387 | 29.245 | 39.072 | 8.0% |
| detector postprocess | 20.234 | 19.800 | 27.520 | 5.3% |
| image decode | 16.686 | 16.726 | 20.349 | 4.4% |
| motion | 14.450 | 19.114 | 28.530 | 3.8% |
| appearance | 0.544 | 0.465 | 1.150 | 0.14% |
| tracker | 0.513 | 0.196 | 1.864 | 0.13% |
| serialization | 0.156 | 0.138 | 0.319 | 0.04% |
| fusion | 0.203 | 0.166 | 0.519 | 0.05% |

The nested rows above are not additive: raw/full-pass rows include their detector internals, and inference /
preprocess / postprocess are components of detector policy.

Additional host evidence:

- frame total: mean **379.982 ms**, p50 **377.072 ms**, p95 **421.460 ms**
- synchronous inference host-wait residual: mean **9.604 ms/frame**, p95 **37.838 ms/frame**
- system CPU idle: mean **21.23%**, p50 **18.78%**, p95 **49.46%**
- GPU utilization: **not observable through the chosen ONNX Runtime DirectML interface**

Linear Smoke400 -> FULL5000 projection, with the smoke-representativeness caveat:

- projected wall: **1899.9 s = 31.67 min = 0.528 h**
- scale factor: **12.5x**

### Dominant-cost conclusion

The first optimization target remains detector invocation/inference, not Python/UI bookkeeping.

On the exact Smoke400 control:

1. detector policy consumed **91.4%** of measured frame time;
2. synchronous ONNX inference alone consumed **76.9%**;
3. the two raw ROI passes consumed **60.9%** of frame time versus **30.4%** for the one full pass.

Preprocess and postprocess were the next detector-internal costs at 8.0% and 5.3%. Decode, motion, tracker,
appearance, fusion, and serialization are not first-order blockers.

### Candidate 1 — person-only tile postprocess

Research toggle:

`tile_person_only_postprocess=true`

This is semantics-preserving for the people-recall path because non-person tile detections were already discarded
before final fusion. Mixed-class unit coverage verifies identical final detections and unchanged ONNX-call count.

It was killed before an expensive candidate Smoke400 pass using the **measured control Smoke400 Amdahl ceiling**:
even deleting *all* detector postprocess time (a stronger improvement than this candidate can provide) would yield
only **1.056x**, below the locked **1.10x** promotion gate.

Decision: **STOP / no FULL5000**.

### Candidate 2 — contiguous NCHW input

Research toggle:

`force_contiguous_input=true`

The current preprocess path creates an NCHW transpose view. This candidate materializes it as contiguous before
`session.run`, moving any implicit input-layout copy into the measured preprocess stage without changing tensor
values.

Measured artifact:

`E:\SpectraTrack-data\runs\a4-nightowls-smoke400-contiguous.json`

Exact Smoke400 result:

| Metric | CONTROL | contiguous-input | Delta |
| --- | ---: | ---: | ---: |
| wall seconds | 152.009 | 148.056 | -3.953 s |
| throughput frames/s | 2.631 | 2.702 | +2.7% |
| speedup | 1.000x | **1.027x** | below 1.10x gate |
| ONNX calls | 1200 | 1200 | 0 |
| inference total | 116.834 s | 108.251 s | -8.583 s |
| preprocess total | 12.155 s | 18.220 s | +6.065 s |
| detector-policy total | 138.882 s | 135.752 s | -3.130 s |
| TP / FP / FN | 135 / 126 / 48 | 135 / 126 / 48 | 0 / 0 / 0 |
| precision / recall | 0.517241 / 0.737705 | 0.517241 / 0.737705 | 0 / 0 |
| output SHA-256 | `e09e9300...76be` | `e09e9300...76be` | **identical** |

The candidate clearly moves work out of synchronous inference into explicit preprocessing, but the net end-to-end
gain is only **2.7%**. The sequential host-load difference is therefore irrelevant to the decision: even the
observed favorable result is far below the 10% gate.

Candidate FULL5000 linear projection: **1850.5 s = 30.84 min**.

Decision: **STOP ON SMOKE / no FULL5000**.

### Smoke-first cost model

A5's 400/5000 slice is 8% of the parent frame count, so any candidate rejected on smoke avoids approximately 92%
of variable frame work versus naively running that candidate on FULL5000.

For the measured control + contiguous pair:

- naive projected FULL5000 pair: **3750.5 s = 62.51 min**
- actual Smoke400 pair: **300.1 s = 5.00 min**
- avoided projected work because no candidate survived: **3450.4 s = 57.51 min**
- avoided fraction: **92.0%**

If two research candidates are smoked and exactly one survives to a FULL5000 run, the frame-count model is
10,000 naive full frames versus 5,800 smoke-first frame-equivalents, approximately **42%** less variable work.
If none survives, the reduction is **92%**.

### Provider sanity check

An early three-frame run used the packaged `.yolo` Python environment and exposed only
`CPUExecutionProvider`. Its absolute latency was deliberately excluded from target-PC promotion evidence.
The authoritative Smoke400 pair above used the system environment with
`DmlExecutionProvider, CPUExecutionProvider`. The comparison gate now rejects cross-provider comparisons.

### Promotion recommendation

No new A4 optimization is promoted to FULL5000:

- person-only tile postprocess cannot reach the 1.10x gate even under an impossible all-postprocess-eliminated
  upper bound;
- contiguous-input is exact-output preserving but measured only **1.027x**, below the gate.

The remaining material performance lever is detector-call reduction. Existing bounded scheduler/call-budget
research remains a **future quality-gated hypothesis**, not a production change. NightOwls is
`tracking_supported=false` and Smoke400 is a sparse metadata-selected slice, so it must not be misused as
temporal tracking/scheduler ground truth. Any future cadence/ROI scheduler must be evaluated on a valid contiguous
tracking corpus with explicit recall/FN/discovery evidence before speed can justify promotion.

No production quality threshold, detector policy, enhancement policy, merge/release branch, or frozen corpus was
changed.



## A4 structural detector-call budget milestone — predeclared 2026-09-29

This section supersedes the earlier *future research* proposal, not the already rejected R1 micro-candidate.

Starting branch/head verified before new work:

`agent/vnext-performance @ 5c880267498a18b51b2e5c5f92f6c5360fafc212`

**Immutable R1-only lock was committed FIRST**, before any disjoint R2 scoring:

- pre-data lock commit: `0f910b6050c40c22e0607d548402389d528592e3`
- immutable lock Git blob SHA-1: `351bae3ce60f7365f79cd989a98551e962049c49`
- contract: `pc/benchmarks/vnext/performance/R2_STRUCTURAL_BUDGET_LOCK.v1.json`
- candidate: exactly one, `global_plus_least_supported_one_raw_tile_v1`

### Evidence and one structural hypothesis

Locked A5 R1 Smoke400 `nightowls-public-smoke400-r1`: 400 frames; control 1200 actual
ONNX invocations = 400 full-frame + 800 raw tiles; enhancement OFF. The two raw tiles
accounted for 60.9% of measured frame time. R1 control: TP=135, FP=126, FN=48,
GT=183, precision=0.517241, recall=0.737705, wall=152.009 s with
`DmlExecutionProvider,CPUExecutionProvider`.

Earlier contiguous-input micro-candidate achieved **1.027x**, below pre-existing
1.10x gate, and is **REJECTED**, not claimed as an improvement.

The one new structural hypothesis is to retain full-frame global discovery on EVERY image,
then infer just ONE of the two production raw tiles. After the full-frame inference, count
its predicted person centers in each tile. Execute the tile with fewer already-explained
person centers; equal counts use the first SHA-256 byte of UTF-8 `video#frame` modulo 2.
No ground truth, R2 result, tracking state or prior frame is used for scheduling.

The exact production `_tile_regions`, `_detect_once` decoding/thresholds and
`_merge_detections` IoU=0.55 are reused without changing production implementations.
Geometry other than exactly two production regions **BLOCKS**, rather than silently
falling back or searching alternate tile/overlap thresholds.

### Immutable compute/latency/quality gate

| Metric | Locked CONTROL | Locked candidate | Interpretation |
| --- | ---: | ---: | --- |
| Full-frame ONNX calls / selected image | 1 | 1 | global discovery retained every frame |
| Raw tile ONNX calls / selected image | 2 | 1 | structural reduction |
| Enhanced ONNX calls | 0 | 0 | A3 OFF decision unchanged |
| Total calls / selected image | 3 | **2 maximum and exact** | hard fail on mismatch |
| Total for identical 400 frames | 1200 | **800** | predicted structural calls, not a measured candidate result |
| Call reduction | — | **33.33%** | from the immutable schedule |
| Paired detector policy latency | control measured concurrently | must be at most control/1.10 | no hardware/throughput extrapolation |
| Recall delta and precision delta | A5 scorer | each must be nonnegative | both gates enforced |
| FN delta and FP delta | A5 scorer | each must be nonpositive | both gates enforced |
| GT denominator | same immutable GT | exactly equal | TP + FN = GT required |

`verify_pair` fail-closes if input byte digests, ordered frame keys, selected
manifest/GT/selection hash, model SHA, providers, configuration, per-frame/aggregate
invocation counters, denominator, precision or recall accounting differ. It uses
`qa_benchmark.evaluate_frames` with label `person` and IoU 0.5; ignore rules are
unchanged. A candidate can be `REJECT` on quality even if it saves exactly one
inference/frame. There is no threshold search and no after-R2 retuning.

### Reproducible research runner

- `pc/spectratrack/structural_call_budget.py`
- `pc/tests/test_structural_call_budget.py`

The paired runner opens each selected source image only once and runs control and
candidate on that same decoded array, alternating AB/BA execution order by selected
frame index. The report includes per-frame input SHA-256, exact source-key order,
detector wall/stage time, actual low-level ONNX invocation counts, selected tile,
both detection-output digests, model/provider provenance and canonical A5 TP/FP/FN
and precision/recall. R1 mode requires the three exact A5 R1 proof hashes.

Future disjoint R2 mode requires three explicit immutable R2 proof hashes AND the
exact locked R1 GT/selection proof. It rejects overlapping frame keys or source
paths before model inference. This implementation does **not** authorize a R2 run:
an independent owner must supply frozen disjoint R2 evidence and explicitly run it.

R1 reference bytes remain external to the repository; no model/video is checked in.

Example **R1-only** invocation on the target PC after attaching the exact external files:

```powershell
cd pc
python -m spectratrack.structural_call_budget --phase r1 \
  --ground-truth <EXACT_R1_GT.jsonl> \
  --selection-manifest <EXACT_R1_PROOF.json> \
  --source-root <EXACT_R1_FRAME_ROOT> \
  --model <EXACT_MODEL_SHA_MATCHING_yolo11x.onnx> \
  --source-commit <CURRENT_EXACT_RESEARCH_SHA> \
  --output <R1_PAIRED_REPORT.json>
```

There is intentionally no predeclared R2 invocation executed by A4 in this milestone.
Do not inspect or infer on holdout4200/FULL5000; NightOwls remains
`tracking_supported=false`, so no discovery-latency or identity/track-GT claim.

### Measurement and limitations

From the existing R1 report only:

- control frame wall: 152.009 s / 400 = 380.02 ms/frame (historical, includes full R1 pipeline);
- full+two-tile structural total: 1200 calls;
- locked one-tile structural total: 800 calls for 400 two-tile frames;
- actual new candidate DirectML latency: **UNMEASURED**, requires target model/frame bytes;
- actual new candidate TP/FP/FN and precision/recall: **UNMEASURED**;
- GPU usage / VRAM: not measured (no trustworthy per-process DirectML telemetry);
- source-seconds/processing-seconds ratio: unavailable for sparse selected still images.

**Risk:** the omitted tile can contain a small person missed by the full-frame pass.
Full-frame every image avoids permanently hiding the whole scene by a track-only ROI,
but does not guarantee small-person recall. The new GT gates are deliberately strict.

Decision pending R1 paired run and independent disjoint R2: **CONTINUE RESEARCH ONLY**.
No production policy changes, no other branch merges, no holdout/full inference,
no release/cutover.
