# A3 — vNext Enhancement Efficiency Research

Branch:

`agent/vnext-enhancement`

Exact research base:

`vnext-base @ d03af3ae6425d3ea2e4d52e25389fecc09957394`

Immutable product baseline:

`integration @ 3ebc4d50213593cac62b97399447fccf6bbc1755`

Tested code HEAD before this handoff-only documentation update:

`46979da7f1e048060588d5ab0ba3c35004b3aa55`

The final branch HEAD is the documentation commit containing this handoff and is reported in the final agent response.

## 1. Objective and decision status

The field observation remains:

`people-recall + adaptive: about 1 second source video ~= 2 minutes processing`

That observation does **not** contain enough run provenance to attribute the approximately 120x cost to OpenCV preprocessing, ONNX inference, a particular resolution/model/provider, or a specific enhancement gate.

This branch therefore does not optimize production OpenCV and does not claim that "enhancement" itself is the bottleneck.

Current decision:

**continue research; not a production candidate yet.**

The branch adds research-only tooling that separates:

- quality-router cost;
- individual image-operation cost;
- raw weak-person probe detector cost;
- enhanced follow-up detector cost;
- actual low-level ONNX call counts;
- optional GT quality effects after an A5 frozen corpus exists.

No production behavior was changed.

## 2. Changed files

- `pc/spectratrack/research/__init__.py`
- `pc/spectratrack/research/enhancement_efficiency.py`
- `pc/tests/test_vnext_enhancement_efficiency.py`
- `pc/benchmarks/vnext/enhancement/README.md`
- `pc/benchmarks/vnext/enhancement/roi_manifest.example.jsonl`
- `docs/agents/vnext/enhancement.md` — this handoff only

Not changed:

- detector production policy;
- tile geometry;
- final NMS/NMM/fusion;
- app/runtime scheduler;
- tracker;
- integration/main/RC;
- other agent role files.

## 3. Current adaptive compute structure

Inspection of the production path shows that adaptive people-recall performs:

1. one A1-owned full-frame detector call;
2. one raw low-confidence probe on every detector-owned region;
3. one additional enhanced detector call for every region where the quality router activates at least one operation.

For A3's ROI-level experiment, therefore:

`raw probe calls = ROI count`

`enhanced follow-up calls = affected ROI count`

`A3 isolated candidate calls = ROI count + affected ROI count`

The A3 profiler intentionally does not execute or attribute the A1-owned full-frame pass or final fusion. A4 must add those costs for end-to-end scheduler estimates.

A4 independently documented the full production formula as:

`total ONNX calls/detector frame = 1 + tile_count + enhanced_tile_count`

and verified 1080p / tile 640 / overlap 0.20 as 8 detector-owned tiles, hence 9..17 ONNX calls per detector frame depending on enhancement activation. That is an A4 structural result, not an A3 target-hardware latency measurement.

## 4. Operation cost table

No real A5 corpus + exact target-model/provider experiment is available yet, so image-operation milliseconds below are deliberately **not fabricated**.

| Candidate | Current gate | Image preprocessing ms on frozen real corpus | Extra ONNX calls per affected ROI | Raw corroboration | Real GT quality result |
| --- | --- | ---: | ---: | --- | --- |
| gamma | darkness >= 0.22 | pending A5/target run | 1 | required | pending |
| CLAHE | darkness >= 0.22 | pending A5/target run | 1 | required | pending |
| gamma + CLAHE | darkness >= 0.22 | pending A5/target run | 1 | required | pending |
| bilateral denoise/deblock | noise >= 0.24 OR compression >= 0.20 | pending A5/target run | 1 | required | pending |
| mild sharpen | blur >= 0.18 plus structure/noise/compression guards | pending A5/target run | 1 | required | pending |
| current adaptive | union of current operation gates | pending A5/target run | 1 | required | pending |
| current adaptive with cached quality map | same operation sequence, cached quality candidate | pending A5/target run | 1 | required | pending |

The profiler records operation preprocessing ms separately from detector wall time and ONNX inference time.

Synthetic tooling validation verifies that one affected ROI with one raw probe and one enhanced follow-up is accounted as exactly:

- raw probe calls: 1;
- enhanced calls: 1;
- isolated total calls: 2.

That synthetic result validates accounting only. It is not CCTV quality/performance evidence.

## 5. Activation frequency

Real activation frequency is **pending** because A5 has not published a frozen real corpus revision.

The new `quality-audit` command records on a supplied ROI manifest:

- darkness gate frequency;
- blur gate frequency;
- compression gate frequency;
- noise gate frequency;
- actual current operation-combination frequency, including no-op regions.

This is required before deciding that a gate fires "too often".

No gate threshold was changed or calibrated by intuition.

## 6. Recovered / missed detections

Real values:

- recovered GT persons: **pending A5 frozen corpus**
- lost GT persons: **pending A5 frozen corpus**
- FN delta: **pending A5 frozen corpus**

The profiler can compute these against canonical `qa_benchmark.py` ground truth when `--ground-truth` and an immutable `--corpus-revision` are supplied.

The synthetic accounting test includes one artificial recovery only to verify that metric plumbing works. It must not be used as a product-quality result.

## 7. False-positive impact

Real new-FP impact is **pending A5 frozen corpus**.

A3 reports **pre-fusion FP observations** only, because A3 does not own final A1 fusion/NMS/NMM.

Final frame-level FP/precision must be evaluated by the canonical A5 comparison after A1 fusion.

Raw corroboration remains mandatory:

- enhanced-only person candidate without same-class spatial raw support -> reject;
- strong raw detections remain accepted;
- weak raw evidence may corroborate an enhanced candidate.

## 8. Quality-router audit

Current `assess_frame_quality()` was inspected rather than replaced.

Current behavior:

- grayscale is created at source ROI size;
- most quality signals are evaluated on a representation reduced to max side 640;
- darkness uses mean luma;
- blur uses Laplacian variance;
- noise uses a Gaussian-residual median;
- compression uses the 8-pixel block-boundary heuristic on **full-resolution** grayscale;
- low-resolution score uses source ROI dimensions.

Research tooling adds component timing and reduced-representation candidates at default max side 320 and 160.

For each reduced candidate it records:

- total assessment ms;
- per-component ms;
- mean absolute quality-signal difference vs current router;
- gate-flip frequency vs current thresholds.

This allows a cheaper quality representation to be judged by measured gate stability rather than intuition.

Quality-map reuse is also isolated:

- `current_adaptive` measures the production behavior, including its own internal quality reassessment;
- `current_adaptive_cached` reuses the already-computed quality map and measures that candidate separately.

No production quality threshold or algorithm was changed.

Temporal reuse of quality maps across frames remains unmeasured and is not claimed safe.

## 9. Selective candidate design

A3 does not create a tile grid or scheduler.

Research input is a detector/scheduler-owned ROI JSONL manifest:

schema:

`spectratrack-vnext-enhancement-roi-v1`

Each ROI contains:

- video;
- frame;
- stable experiment ROI id;
- externally supplied bbox;
- optional evidence signals.

Supported selective gates:

- quality issue;
- dark ROI;
- blur ROI;
- compression ROI;
- weak-person evidence;
- known-track ROI;
- uncertainty ROI;
- quality-or-evidence union.

The profiler applies each enhancement candidate directly to the **raw ROI**, not to another enhancement result, so candidate evaluation cannot accidentally double-enhance.

The only intentional combination candidate is the explicit current low-light sequence `gamma + CLAHE`, plus the complete current-adaptive sequence.

## 10. Quality-per-compute table

Real quality-per-compute values are pending the same frozen A5 corpus, exact model and provider.

The profiler will produce for every candidate:

| Metric | Recorded |
| --- | --- |
| recovered GT persons | yes, when A5 GT exists |
| lost GT persons / FN contribution | yes, when A5 GT exists |
| pre-fusion new FP observations | yes, when A5 GT exists |
| best-match bbox IoU delta | yes, when A5 GT exists |
| GT-relative bbox center jitter delta | yes, when A5 GT exists |
| GT-relative bbox size jitter delta | yes, when A5 GT exists |
| quality-assessment ms | yes |
| operation preprocessing ms | yes |
| raw-probe detector/inference ms | yes |
| enhanced detector/inference ms | yes |
| raw ONNX calls | yes |
| enhanced ONNX calls | yes |
| isolated candidate total ONNX calls | yes |
| processing seconds / sampled source second | yes, when source FPS is available |

A3 deliberately labels FP as pre-fusion and deliberately excludes A1 full-frame/fusion costs from its ROI-local accounting.

## 11. Recommended operations

No operation is promoted to production yet.

The evidence currently supports only these research recommendations:

- keep raw corroboration mandatory;
- compare gamma and CLAHE separately as well as the existing gamma+CLAHE combination;
- compare bilateral and sharpen independently rather than assuming the current combination helps;
- compare `current_adaptive` against `current_adaptive_cached` to quantify duplicate quality-assessment cost;
- run enhancement only on externally selected/suspect ROIs in selective candidates;
- preserve a no-enhancement path for already-good regions.

A keep/remove decision for gamma, CLAHE, bilateral, sharpen, or their combinations requires A5 real GT quality-per-compute results.

## 12. Operations / behavior to remove or disable

No individual image operation is justified for permanent removal yet.

What is **not** supported as a production strategy is blind "enhance every region" behavior without measured quality benefit and compute budget.

A3 recommends disabling any future production proposal that:

- accepts enhanced-only candidates without raw evidence;
- double-enhances an ROI;
- runs an enhanced follow-up on every ROI regardless of measured gate/evidence;
- reports preprocessing cost without separately counting the extra ONNX call.

## 13. Dependency / handoff to A4 scheduler

A3 outputs the signals A4 needs:

- ROI count;
- operation gate frequency;
- selective gate frequency;
- affected ROI count;
- quality-assessment ms;
- operation preprocessing ms;
- raw detector/inference ms;
- enhanced detector/inference ms;
- raw probe inference calls;
- enhanced inference calls;
- processing seconds per sampled source second;
- optional GT quality deltas.

A4 remains owner of:

- full-frame call accounting;
- detector cadence;
- scheduler/budget policy;
- global/track/scene-change scheduling;
- hard inference-call budgets.

A4 should combine:

`A1/A4 full-frame work + A3 raw ROI work + A3 affected enhanced ROI work`

rather than treat A3's ROI-local total as end-to-end runtime.

No A4 branch content was merged or modified.

## 14. A5 corpus revision

At handoff time, the observed `agent/vnext-qa` branch contains evaluator/tooling changes but no published frozen real CCTV corpus revision.

A5 corpus revision used by A3:

**none / not yet available**

Therefore no real:

- operation ms table on target evidence;
- activation frequency;
- recovered/missed count;
- FP delta;
- bbox localization/jitter delta;
- end-to-end quality-per-compute ranking

is claimed in this handoff.

Required rerun after A5 freeze:

- night;
- compression;
- blur;
- tiny-person subsets;
- negatives where available.

All finalists must use the exact same frozen A5 revision and exact model/provider provenance.

## 15. Tests actually run

Exact tested code HEAD:

`46979da7f1e048060588d5ab0ba3c35004b3aa55`

GitHub Actions PC CI:

- workflow run: `36165997758`
- run number: `133`
- conclusion: **success**

Actual results:

- `ruff check spectratrack tests`: success, `All checks passed!`
- `python -m compileall -q spectratrack tests` + `pytest -q`: **153 passed in 1.31s**
- synthetic tracker benchmark: `500 frames / 24 targets / 11970 observations`
- synthetic tracker benchmark elapsed: `0.376 s`
- synthetic tracker benchmark throughput: `1328.1 tracker_fps`
- diagnostics: success
- self-check: success
- PyInstaller standalone Windows build: success
- standalone `SpectraTrack-PC.exe --help`: success
- standalone `SpectraTrack-PC.exe batch --help`: success
- packaging/artifact upload: success

The CI tracker benchmark and synthetic A3 unit fixtures are tooling/regression checks only. They are not real CCTV enhancement measurements and are not RX 5700 XT performance evidence.

## 16. Commits in this research branch before handoff docs

- `043ee0b2902c349058dc6fcccd4326f3e8b53fb7` — research(a3): add enhancement efficiency profiler
- `6b98b8f66167326651183fddadee2be376270ee1` — research(a3): mark research package
- `02eff23f29def2a62457425c604ec61874cba258` — research(a3): reuse quality map for adaptive combination
- `7f813ce04bca7ac1a7b76fa9c6144139224def53` — research(a3): separate current and cached adaptive costs
- `e57d13cc7639fb18f231a107d7100f08afe33c69` — test(a3): cover enhancement research profiler
- `8b9b0d7e10023ac69dc1e8b6d480c2818b829837` — docs(a3): document enhancement profiling workflow
- `5a1214f5f110358a8d28248eddc3825f001ef163` — docs(a3): add ROI manifest example
- `27c8c4871c90b23dd97e8b9299d626bde27fcbb1` — research(a3): fix adaptive cost attribution
- `0df214e605c8f8636183787ae6be6ded1df2eee3` — test(a3): cover adaptive operation combinations
- `728b990cd7c9953ca7aaa8d9f2656a1eae823c3d` — test(a3): verify enhancement call accounting
- `46979da7f1e048060588d5ab0ba3c35004b3aa55` — test(a3): use canonical GT object type

## 17. Cross-agent overlap

File-level overlap with the observed A4 performance branch at code handoff: **none**.

File-level overlap with the observed A5 QA branch at code handoff: **none**.

A3 reads canonical A5 GT via existing `qa_benchmark.py` but does not modify the evaluator.

A3 consumes externally supplied ROI/evidence signals and does not implement A1 tile geometry or A4 scheduling.

## 18. Readiness

Research tooling readiness: **yes**.

Production enhancement readiness: **no**.

Evidence status:

**continue research**

Next evidence gate:

1. A5 publishes an immutable human-confirmed corpus revision;
2. exact detector model/hash/provider is fixed;
3. run `quality-audit`;
4. run operation finalists and selective gates on the same ROI manifest/corpus;
5. pass measured A3 costs/signals to A4;
6. compare final frame-level quality through A5/A1-owned evaluation/fusion;
7. only then decide which operations or gates deserve integrator review.


## Round 2 supplement — explicit per-frame enhancement budget

Validated research code HEAD before this documentation update:

`898caa9c6ecff4d15b0d5cbabaae9390b8e86fd8`

Round-2 tooling now supports an explicit research budget:

`--max-enhanced-rois-per-frame N`

Semantics:

- `0` preserves the previous unlimited research-profiler behavior;
- positive values cap actual enhanced follow-up inference independently for each operation and source frame;
- operation/selective gate counts are still recorded even when the budget blocks the expensive follow-up;
- blocked follow-ups are reported as `budget_skipped_rois`;
- raw probe calls remain unchanged and separately counted;
- production enhancement/runtime behavior is unchanged.

This makes the strict weak-evidence experiment reproducible in the profiler itself instead of relying only on a specially pre-filtered ROI manifest.

The intended first Round-2 setting remains:

- selective gate: `weak-person`;
- max enhanced ROIs per source frame: `1`;
- raw corroboration mandatory;
- compact operation set: bilateral / current_adaptive_cached / sharpen;
- no-enhancement remains the baseline outside this profiler.

### CI

GitHub Actions run:

`36241047867`

Result: **SUCCESS**

Observed:

- ruff: passed;
- compile + pytest: **154 passed in 2.66 s**;
- tracker smoke: 500 frames / 24 targets / 11970 observations;
- tracker smoke throughput: 1798.9 tracker FPS;
- crossing/reappearance smoke ID switches: 0;
- diagnostics/self-check: passed;
- Windows standalone build/smoke/package/upload: passed.

A deterministic test verifies that two eligible ROIs in one source frame produce:

- 2 raw probe calls;
- only 1 enhanced inference call when the cap is 1;
- 1 affected ROI;
- 1 budget-skipped ROI.

### Decision status

**TOOLING READY; QUALITY DECISION STILL PENDING.**

The target-PC strict weak-person result must still be verified or rerun after the Remote Desktop Commander control channel is reliable. This code change does not by itself prove that any enhancement operation should be enabled in production.


## Round 2 public-first enhancement assignment — 2026-09-26

Status: **IN PROGRESS / TARGET-PC STRICT PROFILE NEEDS VERIFICATION**

Broad adaptive enhancement remains rejected as the default candidate.

Current strict candidate stays:

- selective gate: weak-person;
- max enhanced ROIs/source-frame: 1;
- raw corroboration mandatory.

Public evidence order:

1. finish/verify MOT17 strict weak-person evidence when target-PC control recovers;
2. after A5 freezes `nightowls-public-r1`, use NightOwls as the primary night/low-light/blur/occlusion quality-cost benchmark;
3. optional secondary evidence: LLVIP **visible/RGB only** if additional low-light coverage is useful;
4. never use IR/thermal as SpectraTrack detector input.

Compare:

- enhancement OFF;
- bilateral;
- current_adaptive_cached;
- sharpen only if competitive under the strict gate.

If no candidate clears post-fusion quality/cost and stability gates, the required Round-2 outcome is:

`ENHANCEMENT OFF`

Do not tune low-light thresholds on a held-out NightOwls validation split after seeing its results.


## Round 2 continuation — target-PC verification before restart

State at continuation start:

- branch: `agent/vnext-enhancement`;
- exact starting HEAD verified: `4b13f8eaec4e9339ea3247fc759f85400e1a13b4`;
- strict research configuration remains:
  - selective gate = `weak-person`;
  - max enhanced ROIs/source-frame = `1`;
  - raw corroboration mandatory;
  - comparison set = enhancement OFF / bilateral / current_adaptive_cached / sharpen only if competitive.

Existing interrupted target-PC artifact expected from the durable coordinator worklog:

`C:\Users\foto6\SpectraTrack-data\runs\a3-round2-weak1.json`

Existing strict weak-person ROI manifest:

`C:\Users\foto6\SpectraTrack-data\derived\MOT17-04-first300-roi-weak1.jsonl`

### BLOCKED — target-PC artifact verification

A direct Remote Desktop Commander read of the expected A3 result artifact was attempted before any restart.

Result:

- target device relay returned `No devices available`;
- artifact completion/corruption state remains **UNVERIFIED**;
- no A3 benchmark process was restarted;
- no local artifact was overwritten;
- no completion/failure is inferred from the relay failure.

This preserves the Round-2 rule: verify existing process/artifact state before restarting interrupted work.

### A5 public-night gate

Current observed A5 head at this checkpoint:

`agent/vnext-qa @ f4f406182df54335e2c26a34091915408204789c`

The branch contains the NightOwls/DanceTrack import assignment and constrained semantics, but no frozen `nightowls-public-r1` artifact/revision is yet published.

Therefore:

- NightOwls candidate evaluation is **NOT STARTED**;
- no held-out NightOwls result has been observed;
- no gates/thresholds are retuned from held-out evidence;
- LLVIP IR/thermal is not used.

### Safe next work while target PC is disconnected

Only Git-side research/tooling review is allowed:

1. verify the strict profiler can express OFF/bilateral/current_adaptive_cached and the one-ROI budget;
2. close any measurement gap for required post-fusion quality metrics without duplicating A1 fusion policy;
3. wait for reliable target-PC artifact inspection before resuming a missing local run;
4. wait for A5 `nightowls-public-r1` freeze before primary night evaluation.

Production enhancement remains non-candidate unless strict evidence clears the Round-2 quality/cost gates.


## Round 2 checkpoint — interrupted target-PC A3 artifact recovered

### DONE — existing artifact verified before any restart

The previously interrupted target-PC result exists and is complete:

`C:\Users\foto6\SpectraTrack-data\runs\a3-round2-weak1.json`

Verified file metadata:

- SHA-256: `cad760e97e32380ac3da87d66f56bfc5d2a11097839b45e1806b3d9469e291a6`;
- size: `8933` bytes;
- last-write UTC: `2026-09-26T09:56:06Z`;
- artifact source commit: `86b1460b6f9879038de92eb1e1341a96221e6c03`.

Matching ROI manifest:

`C:\Users\foto6\SpectraTrack-data\derived\MOT17-04-first300-roi-weak1.jsonl`

SHA-256:

`55ac3fa574fd7e3653b68538974a728c4f68e789cdbf7b73fbea059e4d44430f`

Process inspection found no surviving A3 benchmark Python process. The only Python processes were VS Code Black formatter processes plus the control supervisor.

Therefore the interrupted A3 benchmark is **DONE**, not restarted, and no existing artifact was overwritten.

A compact Git-persistent copy of its provenance/summary is committed as:

`pc/benchmarks/vnext/enhancement/round2_mot17_weak1_verified.json`

### Exact experiment provenance

- corpus revision: `mot17-public-r1`;
- GT SHA-256: `b32f401268055f585b2e2b2ed90fb62749c924f3e3ff2986b15d390c323c35f4`;
- source video SHA-256: `13a4760ab8c9b54b9ce998a46179e8483609019f41fd521f4e199eb156d32642`;
- model: YOLO11x ONNX;
- model SHA-256: `e84cbad768b218d74ecc85e3e52d84631123719a6951b3ddf6eddc850d5b3f73`;
- model input: 960x960;
- providers: DirectML + CPU fallback;
- sampled source: 10.0 seconds;
- shared three-candidate experiment wall time: 437.343 s;
- actual low-level ONNX calls across the shared experiment: 2726;
- raw probe calls attributed to each isolated candidate: 2400.

Strict gate provenance:

- weak evidence threshold source: A1 tile evidence `0.12 <= score < 0.35`;
- one weak-person selected ROI per source frame in the frozen manifest;
- raw corroboration required;
- no GT was used to choose the ROI.

Important provenance nuance: this artifact predates the later explicit
`max_enhanced_rois_per_frame` profiler field. The one-ROI budget was enforced by
the preselected ROI manifest, not by the later profiler setting. It is valid as the
recorded Round-2 pre-fusion experiment, but a final NightOwls run must use the
explicit profiler cap `1`.

### Verified operation evidence — PRE-FUSION ONLY

| Candidate | Affected ROIs | Extra ONNX calls | Recovered GT | Lost GT | New FP observations | Recovered / extra call | FP / recovered | Center jitter delta | Size jitter delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| bilateral | 30 | 30 | 15 | 0 | +372 | 0.5000 | 24.8000 | +0.0000357 | -0.0001287 |
| current_adaptive_cached | 158 | 158 | 101 | 0 | +2170 | 0.6392 | 21.4851 | +0.0009605 | +0.0010520 |
| sharpen | 138 | 138 | 87 | 0 | +1890 | 0.6304 | 21.7241 | +0.0011781 | +0.0014535 |

Additional measured costs:

| Candidate | Image preprocessing ms | Enhanced detector ms | Enhanced ONNX inference ms |
| --- | ---: | ---: | ---: |
| bilateral | 70.837 | 3880.186 | 3163.282 |
| current_adaptive_cached | 870.734 | 21498.367 | 17034.452 |
| sharpen | 226.529 | 18754.684 | 14856.208 |

Interpretation is deliberately limited:

- bilateral has the lowest FP cost in absolute count and near-neutral measured pre-fusion jitter, but recovers only 15 GT observations;
- cached adaptive recovers more evidence per extra call than bilateral but adds 2170 pre-fusion FP observations and worsens both recorded jitter deltas;
- sharpen is not clearly competitive with cached adaptive on recovered/extra-call or FP/recovered and has the largest center/size jitter regression of the three;
- none of these pre-fusion results can pass the Round-2 promotion gate because post-fusion recall, precision, FP, and canonical A5 stability are still missing.

This evidence does **not** justify retaining enhancement.

## Round 2 checkpoint — canonical post-fusion summary tooling

### DONE — post-fusion metric gap closed without duplicating A1 fusion

Added research-only:

`pc/spectratrack/research/strict_enhancement_summary.py`

and tests:

`pc/tests/test_vnext_enhancement_postfusion.py`

A3 still does not implement final fusion.

The summary tool consumes:

1. canonical already-post-fusion enhancement-OFF result;
2. canonical already-post-fusion strict enhancement candidate result;
3. the matching strict A3 operation-cost profile.

It rejects incomparable evidence when:

- GT SHA differs;
- label / match-IoU differs;
- model SHA differs when present;
- A3 selective gate is not `weak-person`;
- A3 explicit max enhanced ROI cap is not `1`;
- raw corroboration is not enabled;
- profile GT/model provenance disagrees with the post-fusion run.

Metrics produced:

- recovered GT;
- lost GT;
- post-fusion recall + delta;
- post-fusion precision + delta;
- post-fusion FP delta;
- canonical detection bbox-stability/jitter deltas;
- raw ONNX calls;
- extra enhancement ONNX calls;
- recovered GT / extra enhancement call;
- FP cost / recovered GT;
- wall-time delta;
- post-fusion ONNX-call delta;
- processing seconds/source second when present;
- A5 canonical subgroup deltas from `by_tag`, `by_attribute`, and `by_size`.

A3 does not invent NightOwls tags/attributes. Only A5 source-backed subgroup
metadata is surfaced.

Research code HEAD containing the post-fusion comparator and subgroup tests:

`4c18ab061700caeb52c34472035096a2dd618ec2`

GitHub Actions run:

`36254809619` / PC CI #266 — **SUCCESS**

Actual validation:

- Ruff: `All checks passed!`;
- compile + pytest: **159 passed in 1.90 s**;
- synthetic tracker smoke: 500 frames / 24 targets / 11970 observations;
- smoke elapsed: 0.363 s / 1376.4 tracker FPS;
- diagnostics: passed;
- self-check: passed;
- Windows standalone build: passed;
- standalone `--help` and `batch --help`: passed;
- package/artifact upload: passed.

These CI numbers validate tooling/regression behavior only, not enhancement quality.

## NightOwls primary gate — current state

A5 advanced during this checkpoint and now has an official NightOwls validation
importer in its own branch. Latest observed A5 code includes:

- official `nightowls_validation.json` only;
- official PNG/JSON/SDK semantics;
- deterministic pre-result slice support;
- official pedestrian scoring and ignore handling;
- source-backed occlusion/difficulty/pose metadata;
- source-backed size strata;
- no invented blur/contrast/occlusion semantics.

However, target-PC data inspection currently shows no NightOwls import/freeze files
under the existing SpectraTrack-data public/import/run locations.

Therefore:

`nightowls-public-r1` freeze used by A3 = **NOT YET AVAILABLE**

and:

- no NightOwls candidate run has been started by A3;
- no held-out result has been observed by A3;
- no gate/threshold has been retuned;
- LLVIP IR/thermal has not been used.

### NEXT / blocking dependencies

Final A3 decision remains blocked on:

1. A5 freezing the official NightOwls validation/slice as `nightowls-public-r1`;
2. a fixed post-fusion A1 policy/result contract for OFF and each strict candidate;
3. exact same NightOwls bytes/GT/model/provider for OFF and candidate;
4. strict explicit A3 config `weak-person + cap=1 + raw corroboration`;
5. post-fusion quality/cost/stability summary.

Candidate order remains fixed:

1. OFF baseline;
2. bilateral;
3. current_adaptive_cached;
4. sharpen only if it remains competitive.

If no strict candidate gives a convincing post-fusion quality-per-compute and
stability gain on the frozen held-out evidence, A3 must hand off:

`ENHANCEMENT OFF`


## Round 2 checkpoint — NightOwls candidate set locked before held-out

To prevent held-out tuning, A3 froze the NightOwls candidate set **before any
NightOwls candidate result was observed**.

Committed lock:

`pc/benchmarks/vnext/enhancement/round2_nightowls_candidate_lock.json`

Final held-out comparison set:

1. `OFF` — baseline;
2. `bilateral` — low-work selective candidate;
3. `current_adaptive_cached` — higher-recovery selective candidate.

Fixed configuration:

- `selective_gate=weak-person`;
- max enhanced ROIs/source frame = `1`;
- raw corroboration required;
- person acceptance = `0.12`;
- raw probe = `0.08`;
- corroboration IoU = `0.10`.

### Sharpen decision — excluded before NightOwls held-out

`sharpen` is no longer a NightOwls held-out finalist in this Round-2 cycle.

This decision uses only the already-verified MOT17 development evidence and was
made before seeing any NightOwls candidate output.

Relative to `current_adaptive_cached`, sharpen was:

- worse on recovered GT / extra call: `0.6304` vs `0.6392`;
- worse on FP / recovered GT: `21.7241` vs `21.4851`;
- worse on center jitter delta: `+0.0011781` vs `+0.0009605`;
- worse on size jitter delta: `+0.0014535` vs `+0.0010520`.

Sharpen's measured image preprocessing was cheaper, and its total incremental
preprocess+enhanced-detector milliseconds per recovered observation was roughly
comparable to cached adaptive, but this single-run timing advantage was small and
did not offset the worse quality ratios/stability.

It therefore does not meet the user's condition "sharpen only if actually
competitive" strongly enough to spend held-out NightOwls inference budget.

No candidate/gate/threshold change is permitted after the frozen NightOwls
candidate results are observed in this cycle.


## Round 2 continuation — target-PC revalidation and CI fixture repair

Continuation request expected:

`agent/vnext-enhancement @ 4b13f8eaec4e9339ea3247fc759f85400e1a13b4`

After `git fetch origin --prune`, the remote branch had already advanced through
the same A3 Round-2 research line. The requested commit remains an ancestor; it
was not reset or rewritten. Before this state update the branch advanced to the
strict same-source alternate handoff work and then to:

`5b1a2b139edf39c56d24cdb80959a977f116d95e`

### DONE — exact target-PC branch/worktree verification

Target worktree:

`C:\Users\foto6\SpectraTrack-worktrees\a3`

Verified after fetch:

- branch: `agent/vnext-enhancement`;
- working tree: clean before the fixture repair;
- no A3 benchmark process was active;
- another active Python workload belonged to A1 held-out detection and was not
  touched.

No A3 interrupted benchmark was restarted.

### DONE — existing interrupted A3 artifact reverified before restart

Artifact:

`C:\Users\foto6\SpectraTrack-data\runs\a3-round2-weak1.json`

Reverified on the target PC:

- SHA-256: `cad760e97e32380ac3da87d66f56bfc5d2a11097839b45e1806b3d9469e291a6`;
- size: 8,933 bytes;
- last-write UTC: `2026-09-26T09:56:06.4866128Z`;
- schema: `spectratrack-vnext-enhancement-profile-v1`;
- artifact source commit: `86b1460b6f9879038de92eb1e1341a96221e6c03`;
- corpus revision: `mot17-public-r1`;
- actual low-level ONNX calls across the shared experiment: 2,726;
- wall time: 437.342953 s.

This matches the previously persisted verified MOT17 strict-profile evidence.
The artifact was not overwritten or rerun.

### BLOCKED — NightOwls held-out cannot start yet

Target-PC searches under:

- `C:\Users\foto6\SpectraTrack-data`;
- `E:\SpectraTrack`

found no NightOwls dataset/import/freeze/result files.

Latest observed A5 branch:

`agent/vnext-qa @ bf63820f81cd13fdfae8680a200e25f030d314fc`

A5 importer/tooling is CI-validated, but its current state still records:

- real NightOwls validation import: not run;
- frozen `nightowls-public-r1`: not frozen;
- frozen NightOwls hash: unavailable.

Therefore A3 has **not** started NightOwls candidate inference, has not observed
held-out candidate results, and has not retuned any gate/threshold.

The held-out candidate lock remains unchanged:

1. enhancement OFF;
2. bilateral;
3. current_adaptive_cached.

Sharpen remains excluded from the NightOwls held-out candidate set based only on
the pre-held-out MOT17 development evidence already documented above.

Strict semantics remain fixed:

- selective gate = `weak-person`;
- max enhanced ROIs/source-frame = `1`;
- raw corroboration mandatory.

LLVIP remains unused; if later activated it is visible/RGB only. IR/thermal is
not SpectraTrack input.

### FAILED -> FIXED — strict handoff CI fixture provenance

GitHub Actions run `36255981087` / PC CI #305 failed:

- Ruff: PASS;
- pytest: **1 failed, 163 passed**;
- failure:
  `test_apply_alternates_replaces_same_source_measurement_without_new_evidence_source`.

Root cause:

the test prefusion fixture still used the placeholder model hash
`model-sha`, while the strict same-source handoff correctly requires the
enhancement evidence model SHA-256 to equal the prefusion model SHA-256.

The production/research provenance invariant was kept. Only the test fixture was
corrected to the actual SHA-256 of its synthetic `fake-model` bytes:

`5496d6743cdde4fe98674e290e44b3220f5a42bf0c442eb207e88fd041e34d48`

Fix commit:

`5b1a2b139edf39c56d24cdb80959a977f116d95e`

Target-PC validation after fast-forward:

- focused strict-handoff test: **5 passed in 0.56 s**;
- all A3 efficiency/post-fusion/strict-handoff tests: **21 passed in 0.60 s**.

GitHub Actions run `36256240336` / PC CI #313 reached successful Ruff,
compile+pytest, synthetic benchmark, diagnostics and self-check before this
documentation update; packaging/build completion should be checked on the
final branch head before claiming full CI success.

### Current decision

No new enhancement quality result was produced in this continuation because the
primary held-out NightOwls corpus is not frozen.

The A3 production decision remains unresolved between:

- strict bilateral;
- strict current_adaptive_cached;
- **ENHANCEMENT OFF**.

If the frozen NightOwls held-out comparison does not show a convincing
post-fusion quality-per-compute/stability gain, the required A3 recommendation
remains:

`ENHANCEMENT OFF`
