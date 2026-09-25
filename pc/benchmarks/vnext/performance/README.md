# SpectraTrack vNext A4 — Performance / Inference Budget Research

Research branch: `agent/vnext-performance`

Exact research base:

`vnext-base @ d03af3ae6425d3ea2e4d52e25389fecc09957394`

Immutable product baseline:

`integration @ 3ebc4d50213593cac62b97399447fccf6bbc1755`

This directory is research-only. Nothing here changes the production scheduler.

## Field blocker

A real run of `people-recall + adaptive` was observed at roughly:

`1 second source video ~= 2 minutes processing`

That is approximately 120 processing-seconds per source-second for that run.

The run does not currently have enough attached provenance in this branch to attribute the 120x cost to a specific resolution, source FPS, model hash, provider, or stage. Do not use it as a calibrated per-call latency.

## Current production compute model

Production geometry comes from `spectratrack.detector._tile_regions()`, not from an approximation.

Current adaptive people-recall does:

1. one full-frame detector call;
2. one raw low-confidence probe on every detector-owned tile;
3. a second enhanced detector call only for tiles where the quality router selects at least one operation.

Therefore, for a detector frame:

`total ONNX calls = 1 + tile_count + enhanced_tile_count`

where:

`0 <= enhanced_tile_count <= tile_count`.

At the current `tile_size=640`, `overlap=0.20`:

| Resolution | Exact tile count | Full-frame calls | Raw tile calls | Enhanced calls | Total calls/detector frame | Calls/source-second at 30 FPS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1280x720 | 6 | 1 | 6 | 0..6 | 7..13 | 210..390 |
| 1920x1080 | 8 | 1 | 8 | 0..8 | 9..17 | 270..510 |
| 2560x1440 | 15 | 1 | 15 | 0..15 | 16..31 | 480..930 |
| 3840x2160 | 32 | 1 | 32 | 0..32 | 33..65 | 990..1950 |

The 30 FPS column is a normalized reference only. For another source FPS, multiply calls/detector-frame by `source_fps / detect_every`.

### Exact 1080p regions

For 1920x1080, `tile=640`, `overlap=0.20`, production returns exactly:

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

So adaptive 1080p is exactly:

- 1 full-frame inference;
- 8 raw tile inferences;
- 0..8 enhanced tile inferences;
- 9..17 total ONNX calls on every detector frame.

The A4 tests invoke the production detector with a fake ONNX session and verify that production `last_inference_calls` reaches:

- 9 calls on a clean uniform 1080p frame where no adaptive operation is selected;
- 17 calls on a dark uniform 1080p frame where every tile is enhanced.

The same production-counter test verifies people-recall without adaptive enhancement at all four target resolutions.

## Stage observability

The current production performance report already exposes:

- capture/decode: `latency_ms.capture`;
- CMC: `latency_ms.camera_motion`;
- detector policy wall time: `latency_ms.detector`;
- detector preprocessing: `latency_ms.detector_preprocess`;
- ONNX inference: `latency_ms.detector_inference`;
- detector postprocess: `latency_ms.detector_postprocess`;
- appearance: `latency_ms.appearance`;
- tracker: `latency_ms.track`;
- rendering: `latency_ms.hud`;
- session logging: `latency_ms.session_write`;
- video writing: `latency_ms.record_write`;
- whole frame: `latency_ms.frame`;
- actual low-level calls: `onnx_inference_calls`;
- process CPU: `cpu_process`;
- provider list: `providers`.

The current production path does **not** separately expose adaptive quality-assessment time versus adaptive operation-preprocessing time inside the detector policy.

A3 now has research tooling on `agent/vnext-enhancement` that can attribute:

- quality assessment;
- raw probe detector/inference;
- operation preprocessing;
- enhanced detector/inference;
- enhanced inference-call count.

No measured A3 result artifact has been published yet, so A4 does not invent those costs.

GPU utilization and VRAM remain unknown unless a trustworthy telemetry source is supplied.

## Scheduler experiment

The isolated simulator compares four policies:

### CURRENT

Every detector frame:

- full-frame discovery;
- all raw tiles;
- every enhancement-eligible tile.

No call budget.

### COARSE_TO_FINE

Every detector frame:

- one global full-frame discovery call;
- only suspect ROIs receive raw expensive follow-up;
- one bounded enhanced follow-up slot may be reserved.

Discovery guarantee: global discovery every detector cadence.

### TRACK_GUIDED

- known-track ROIs plus suspect ROIs;
- periodic global rediscovery;
- scene-change and camera-motion triggers force immediate global rediscovery even on an otherwise cadence-skipped frame;
- no enhanced calls in the current TRACK_GUIDED research candidate.

Discovery guarantee: periodic global scan is never permanently removed.

### BUDGETED_ADAPTIVE

- known-track ROIs and suspect ROIs share a hard raw-call budget;
- when both exist, the simulator keeps at least one slot for each when budget permits;
- a bounded enhanced slot is reserved only when an eligible suspect exists;
- periodic global rediscovery;
- immediate global rescan on scene-change/camera-motion trigger;
- hard maximum ONNX calls/frame.

## Normalized 1080p scheduler comparison

The following table is a **compute simulation**, not a quality result.

Reference workload:

- 1920x1080;
- 30 FPS;
- 300 source frames;
- detector cadence 1;
- current tile count 8.

Candidate signal assumptions are explicit and intentionally small so they can later be replaced by A5/A1/A2 traces.

| Policy | Research config/signals | Avg ONNX calls/frame | Max calls/frame | Calls/source-second @30 FPS | Periodic global discovery bound |
| --- | --- | ---: | ---: | ---: | ---: |
| CURRENT min | 0 enhanced tiles | 9.000 | 9 | 270 | 1 frame |
| CURRENT max | 8 enhanced tiles | 17.000 | 17 | 510 | 1 frame |
| COARSE_TO_FINE | 4 suspects, 1 enhanced; max calls 6 | 6.000 | 6 | 180 | 1 frame |
| TRACK_GUIDED | 2 tracks, 1 suspect; max calls 4; global period 15 | 3.067 | 4 | 92 | 15 frames / 0.50 s |
| BUDGETED_ADAPTIVE | 2 tracks, 2 suspects, 1 enhanced; max calls 4; global period 10 | 4.000 | 4 | 120 | 10 frames / 0.33 s |

These numbers describe inference-call budgets only. They do not imply proportional wall-time speedup.

## Quality / cost frontier status

No frozen A5 real CCTV corpus revision is available yet.

Therefore:

| Policy | Recall delta | FN delta | Real processing s/source-s | New-person discovery |
| --- | --- | --- | --- | --- |
| CURRENT | pending A5 baseline | pending | field observation ~120x exists, provenance incomplete | global every detector frame |
| COARSE_TO_FINE | pending A5 | pending | pending target run | <= detector cadence |
| TRACK_GUIDED | pending A5 | pending | pending target run | <= periodic bound; earlier on trigger |
| BUDGETED_ADAPTIVE | pending A5 | pending | pending target run | <= periodic bound; earlier on trigger |

The scheduler should not advance to production on compute reduction alone.

## Discovery invariant

ROI scheduling must never make an unseen person permanently undiscoverable.

The research candidates enforce this by retaining global discovery:

- CURRENT: every detector frame;
- COARSE_TO_FINE: every detector frame;
- TRACK_GUIDED: periodic global scan plus immediate scene/camera trigger;
- BUDGETED_ADAPTIVE: periodic global scan plus immediate scene/camera trigger.

If `detect_every > 1`, scene-change/camera-motion triggers override the cadence in the experimental scheduler.

## Batching

A research-only `batch-probe` command is included.

It records:

- actual ONNX input shape;
- active provider(s);
- fixed/dynamic batch support;
- median latency for supported batch sizes;
- items/second.

It does **not** claim a batching win without a real model and a DirectML run.

No model binary is committed to the repository and no target RX 5700 XT DirectML model run is available in this branch. Therefore the DirectML batching result is currently:

`NOT MEASURED — no production model artifact + target GPU benchmark available`.

VRAM is intentionally left unknown rather than estimated.

## Commands

Current exact compute model:

```powershell
cd pc
python -m spectratrack.vnext_performance compute-model --source-fps 30
```

Scheduler simulation:

```powershell
python -m spectratrack.vnext_performance scheduler-sim \
  --policy BUDGETED_ADAPTIVE \
  --resolution 1080p \
  --frames 300 \
  --source-fps 30 \
  --track-rois 2 \
  --suspect-rois 2 \
  --enhancement-eligible-rois 1 \
  --global-period 10 \
  --max-calls 4 \
  --max-enhanced-calls 1
```

Convert an existing production `--perf-report` into source-time rates:

```powershell
python -m spectratrack.vnext_performance analyze-report \
  --report perf.json \
  --source-fps 30
```

DirectML batching probe on the exact target model:

```powershell
python -m spectratrack.vnext_performance batch-probe \
  --model path\\to\\exact-model.onnx \
  --batch-sizes 1,2,4 \
  --warmup 3 \
  --repeats 20
```

## Current recommendation

Do not micro-optimize Python/OpenCV first.

The dominant structural problem is that current people-recall can issue between 9 and 17 ONNX calls for a single 1080p detector frame and up to 65 calls for a 4K detector frame.

The research direction to carry forward is **BUDGETED_ADAPTIVE**, with TRACK_GUIDED as a useful lower-cost comparator, because it provides:

- a hard inference-call ceiling;
- bounded global rediscovery;
- immediate scene/camera rescan;
- non-starved track and suspect ROI work;
- a bounded enhanced follow-up slot.

This is a compute recommendation only. A5 recall/FN/discovery-latency evidence is still required before any production integration.
