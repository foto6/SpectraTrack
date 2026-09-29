# A3 vNext Enhancement Efficiency Research

Research-only tooling for measuring whether adaptive enhancement is worth its compute cost.

This directory does **not** change production runtime behavior.

## Ownership boundaries

A3 consumes ROI coordinates produced elsewhere. It does not generate a tile grid, schedule global work, change tracker behavior, or perform final NMS/NMM/fusion.

Expected upstream owners:

- A1: detector/tile/fusion policy;
- A4: runtime scheduling and compute budgets;
- A5: frozen corpus and human-confirmed ground truth.

## ROI manifest

Schema: `spectratrack-vnext-enhancement-roi-v1`.

First JSONL record:

```json
{"type":"metadata","schema":"spectratrack-vnext-enhancement-roi-v1","source":"a1-or-a4","corpus_revision":null}
```

ROI records:

```json
{"type":"roi","video":"clip.mp4","frame":12,"roi_id":"clip-12-r0","bbox":[100,80,740,720],"signals":{"weak_person":true,"known_track":false,"uncertainty":true}}
```

The ROI geometry is treated as external input. A3 does not derive or modify it.

Optional evidence flags:

- `weak_person`
- `known_track`
- `uncertainty`

These are gating inputs only. The profiler does not create a scheduler from them.

## Quality-router audit

Run without a model:

```powershell
python -m spectratrack.research.enhancement_efficiency quality-audit `
  --roi-manifest benchmarks/vnext/enhancement/regions.jsonl `
  --video-root benchmarks/videos `
  --experiment-id router-r1 `
  --source-commit <HEAD> `
  --output benchmarks/vnext/enhancement/router-r1.json
```

The audit records:

- current gate activation frequency;
- total current quality-assessment cost;
- component timing;
- reduced-representation candidates, default max side 320 and 160;
- mean absolute signal error against the current router;
- threshold/gate flip frequency.

Reduced candidates are experiments only. They do not change production thresholds.

## Operation-level profile

With a real model:

```powershell
python -m spectratrack.research.enhancement_efficiency profile `
  --roi-manifest benchmarks/vnext/enhancement/regions.jsonl `
  --video-root benchmarks/videos `
  --model path\to\model.onnx `
  --experiment-id ops-r1 `
  --source-commit <HEAD> `
  --selective-gate quality `
  --output benchmarks/vnext/enhancement/ops-r1.json
```

After A5 publishes a frozen corpus, add:

```text
--ground-truth <A5 ground_truth.jsonl>
--corpus-revision <immutable A5 revision>
```

Operations:

- `gamma`
- `clahe`
- `gamma_clahe`
- `bilateral`
- `sharpen`
- `current_adaptive` — production path as currently implemented, including its own quality reassessment;
- `current_adaptive_cached` — isolated candidate that reuses the already-computed quality map.

Selective gates:

- `quality`
- `dark`
- `blur`
- `compression`
- `weak-person`
- `known-track`
- `uncertainty`
- `quality-or-evidence`

## Cost accounting

The profiler distinguishes:

- quality-assessment milliseconds;
- operation preprocessing milliseconds;
- raw-probe detector wall time;
- raw-probe ONNX inference time;
- enhanced detector wall time;
- enhanced ONNX inference time;
- raw-probe inference calls;
- enhanced inference calls;
- total attributed calls for an isolated candidate.

The multi-candidate experiment reuses one raw-probe pass per ROI. Each candidate table then attributes that shared raw-probe cost back to the candidate as though it were run alone.

A3 does not execute the full-frame detector pass in this profiler. A4 must combine A3 ROI costs with A1/A4 full-frame and scheduler measurements before making end-to-end runtime claims.

## Quality accounting

When A5 ground truth is supplied, each operation reports:

- recovered GT persons;
- lost GT persons;
- pre-fusion FP observation delta;
- best-match bbox IoU delta;
- GT-relative bbox center jitter delta;
- GT-relative bbox size jitter delta.

These are ROI-level, pre-fusion measurements. Final frame-level FP/FN after A1 fusion must be evaluated by the canonical A5 comparison path.

## Raw corroboration

Raw corroboration is mandatory:

- raw probe below the acceptance threshold may support an enhanced candidate;
- an enhanced-only candidate without same-class spatial raw support is rejected;
- strong raw detections remain in the candidate result.

The tool never runs enhancement on the output of another enhancement candidate, preventing accidental double enhancement.

## Required rerun after A5 freeze

Rerun finalists on the exact A5 corpus revision, especially:

- night;
- compression;
- blur;
- tiny-person segments.

Do not use synthetic tests or GitHub Actions hardware as evidence of target RX 5700 XT quality-per-compute.


## Round 2 strict candidate contract

Broad adaptive enhancement is no longer a production candidate.

Round-2 finalist configuration is fixed before held-out evaluation:

- selective gate: `weak-person`;
- max enhanced ROIs per source frame: `1`;
- raw corroboration: required;
- baseline: enhancement OFF;
- candidates: `bilateral`, `current_adaptive_cached`, and `sharpen` only while it remains competitive.

Do not retune these gates after viewing NightOwls validation/slice candidate results.

The verified interrupted target-PC MOT17 artifact is summarized in:

`round2_mot17_weak1_verified.json`

It is **pre-fusion evidence only**. Its recovered-GT / FP values must not be used as the final enhancement decision.

## Post-fusion quality/cost summary

A3 does not implement or fork A1 final fusion. After A1/A5 produces canonical post-fusion OFF and strict-candidate result JSON files, summarize them together with the matching strict A3 profile:

```powershell
python -m spectratrack.research.strict_enhancement_summary \
  --baseline-result C:\path\off.postfusion.json \
  --candidate-result C:\path\bilateral.postfusion.json \
  --a3-profile C:\path\a3-strict-profile.json \
  --operation bilateral \
  --output C:\path\bilateral.strict-summary.json
```

Supported strict operations:

- `bilateral`;
- `current_adaptive_cached`;
- `sharpen`.

The summary refuses profiles unless provenance shows:

- `selective_gate=weak-person`;
- `max_enhanced_rois_per_frame=1`;
- raw corroboration enabled;
- matching GT hash/evaluation settings;
- matching model hash when available.

It reports:

- recovered/lost GT after fusion;
- post-fusion recall/precision and deltas;
- post-fusion FP delta;
- canonical A5 bbox-stability deltas;
- raw and extra enhancement ONNX calls;
- recovered GT per extra call;
- FP cost per recovered GT;
- wall-time / processing-seconds-per-source-second deltas;
- A5-provided `by_tag`, `by_attribute`, and `by_size` subgroup deltas.

The subgroup layer never invents NightOwls semantics. Low-light, occlusion, difficulty, pose, small/distant or other slices are reported only when the A5 canonical import exposes source-backed tags/attributes/size bins.

## NightOwls held-out gate

`nightowls-public-r1` is the primary Round-2 night enhancement benchmark once A5 freezes the official validation import/slice.

NightOwls validation/slice is held-out. Candidate results may decide accept/reject, but they must not be used to retune the fixed weak-person gate or enhancement budget in this cycle.

LLVIP may be used only as secondary visible/RGB evidence. IR/thermal frames are not SpectraTrack detector input.

If OFF is not beaten convincingly on post-fusion quality per compute and stability, the A3 handoff decision is:

`ENHANCEMENT OFF`
