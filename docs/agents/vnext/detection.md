# A1 — vNext Detection / Fusion Research Handoff

Branch:

`agent/vnext-detection`

Exact research base:

`vnext-base @ d03af3ae6425d3ea2e4d52e25389fecc09957394`

Immutable product baseline:

`integration @ 3ebc4d50213593cac62b97399447fccf6bbc1755`

Validated code/research HEAD before this handoff-file-only commit:

`95fe077346a0b955726ea8e89422734e638a4452`

This branch was created from the exact research base and is 13 research commits ahead of it at the validated code HEAD. It is not rebased onto later coordination state and no specialist branch was merged into it.

The final handoff commit changes only this role file. The exact final branch HEAD is therefore reported by the branch/PR and in the architect handoff response rather than pretending this file can embed the SHA of the commit that contains itself.

## 1. Scope and result

A1 owns detector/fusion research only.

Completed research tooling:

- isolated current cross-pass hard NMS from decoder-local NMS;
- captured full-frame + tile detections after decoder-local NMS and before final cross-pass fusion;
- implemented research-only conservative GreedyNMM-style grouping;
- implemented research-only score/evidence-weighted coordinate fusion;
- built deterministic stress cases for bbox winner-flip, tile boundaries, nearby people, partial/full-body geometry, tiny people, edge people, and crowded high-overlap distinct people;
- measured recall/precision/FP/FN, duplicate count, fusion mistakes, bbox localization IoU, center/width/height/area jitter and temporal IoU on the synthetic fixture;
- emitted canonical `spectratrack-detection-replay-v1` output for A2 without rerunning detector inference;
- built research-only ONNX backend adapters/harnesses for current YOLO, RF-DETR and RT-DETRv2;
- added exact model SHA/provider/contract/timing/inference-call provenance fields;
- researched person-specialist feasibility and training/data/license gates;
- made no production detector/runtime behavior change.

Not completed because the required evidence does not exist yet:

- real CCTV NMS/NMM/weighted quality comparison;
- real current-YOLO vs RF-DETR vs RT-DETRv2 quality comparison;
- real target-RX-5700-XT backend timing/VRAM comparison;
- person-specialist fine-tuning;
- production detector/fusion recommendation.

Those remain blocked by the missing frozen A5 human-confirmed corpus and, for backend experiments, exact locally available model exports/weights.

## 2. Production isolation

Final diff relative to `vnext-base` modifies/adds only research documentation/tooling/tests:

- `pc/benchmarks/vnext/detection/BACKEND_SPECIALIST_RESEARCH.md`
- `pc/benchmarks/vnext/detection/README.md`
- `pc/benchmarks/vnext/detection/synthetic_fusion_report.json`
- `pc/spectratrack/research/__init__.py`
- `pc/spectratrack/research/vnext_detection_fusion.py`
- `pc/spectratrack/research/vnext_detector_backend_lab.py`
- `pc/tests/test_vnext_detection_fusion.py`
- `pc/tests/test_vnext_detector_backend_lab.py`
- `docs/agents/vnext/detection.md` (handoff state only)

No production `detector.py`, `app.py`, tracker, enhancement, scheduler, config, shared type, integration/main/RC file is changed by A1.

Decoder-local NMS remains baseline behavior. All A1 fusion candidates operate only on detections that have already passed the existing per-inference decoder-local NMS.

## 3. Owned research commits

Validated 13-commit research history, oldest to newest:

1. `cf816f4007fc60116a120f7baa44e2dcc7f8b9e0` — research(a1): add cross-pass fusion experiment harness
2. `f75094e29bcbafaa972e66904353ff6f1e6e031f` — research(a1): mark detection research package
3. `94ec73bfa300d2580362a197d1aa5a056ffbd8d7` — test(a1): cover fusion stress and replay semantics
4. `1a0fd91dbce06dc57dce7a78efc30186a77dc244` — research(a1): record synthetic fusion stress result
5. `c0a649fdd180d1f160a869d45676c50480142e24` — docs(a1): document fusion experiment workflow
6. `d4e8fc084247a36ec0c21074e51ed957b3c79c7f` — research(a1): add reproducible detector backend lab
7. `a6d06643864a43d6935e919aec247c809102e217` — fix(a1): serialize backend specs from slots dataclass
8. `7343dd3dfe6fe1bf964bc896daedd232b7a8c207` — test(a1): cover backend adapter contracts
9. `80d3f5972ecb40e1d66fba0499e4bf3d8a01fda3` — docs(a1): record backend and specialist feasibility gates
10. `21796dbccea7162a6b9675e2fec5cc6bff6cf868` — research(a1): add backend execution smoke command
11. `367e22e48ad1b599af6d0bb3d1062bf902d455b6` — fix(a1): avoid annotations import shadowing
12. `d86bdca4fb75273bc4e79db13fb8492d1f27a741` — fix(a1): use RT-DETRv2 width-height target size order
13. `95fe077346a0b955726ea8e89422734e638a4452` — test(a1): assert RT-DETRv2 width-height target sizes

History was not rebased or force-pushed.

## 4. Stage 1 experiment design

Canonical detector baseline under investigation:

`full-frame inference + overlapping tile inference + final cross-pass hard class-aware NMS`

The research harness separates two levels explicitly:

### Decoder-local NMS

The current detector's existing per-inference NMS remains active and unchanged.

### Final cross-pass fusion

Only the merge across already-decoded full-frame/tile detections changes between candidates:

- **A / hard-nms** — current winner-takes-box class-aware hard NMS semantics;
- **B / conservative-nmm** — direct-to-seed GreedyNMM-style grouping with safety gates, then envelope geometry;
- **C / weighted** — the same conservative grouping, then score/evidence-weighted bbox coordinates.

The conservative grouping deliberately avoids transitive chain merging. A detection must match the highest-score seed directly.

Default synthetic fusion config:

- fusion IoU: `0.55`
- evaluation IoU: `0.50`
- center-distance gate: `0.20 * min(box dimensions)`
- max width/height ratio: `1.80`
- score power: `1.0`
- full-frame evidence weight: `1.0`
- tile evidence weight: `1.0`

## 5. Synthetic / stress evidence

Artifact:

`pc/benchmarks/vnext/detection/synthetic_fusion_report.json`

Schema:

`spectratrack-vnext-fusion-synthetic-v1`

Fixture:

- 14 frames;
- 16 scored GT people;
- 15 duplicate detections before final fusion;
- 7 consecutive matched jitter pairs;
- 0 ONNX inference calls because detector outputs are deterministic synthetic pre-fusion evidence;
- committed measured harness wall time: `0.002002216999983375 s` on the environment that produced the fixture.

Coverage includes:

- alternating full-frame/tile score winner;
- tile-boundary duplicate;
- full-frame + tile duplicate;
- two nearby people;
- partial-person vs full-body bbox;
- tiny person;
- frame-edge person;
- deliberately high-overlap distinct people.

These numbers validate failure modes and metric plumbing only. They are not CCTV quality evidence.

## 6. NMS vs NMM vs weighted fusion table

| Metric | Current hard NMS | Conservative NMM | Weighted coordinates |
| --- | ---: | ---: | ---: |
| TP | 15 | 16 | 16 |
| FP | 1 | 1 | 1 |
| FN | 1 | 0 | 0 |
| Recall | 0.9375 | 1.0000 | 1.0000 |
| Precision | 0.9375 | 0.9412 | 0.9412 |
| Duplicate count before fusion | 15 | 15 | 15 |
| Fusion mistakes | 1 | 0 | 0 |
| Mean bbox localization IoU | 0.8916 | 0.8679 | 0.9583 |
| Center jitter, px | 1.4142 | 0.0000 | 0.1201 |
| Width jitter, px | 10.0000 | 0.0000 | 0.8493 |
| Height jitter, px | 8.0000 | 0.0000 | 0.6794 |
| Area jitter ratio | 0.3221 | 0.0000 | 0.02735 |
| Temporal IoU | 0.7369 | 1.0000 | 0.9745 |

Interpretation limited to this constructed fixture:

- current hard NMS reproduces the expected winner-flip jitter failure and deliberately loses one of two highly overlapping distinct people;
- conservative NMM avoids that merge in the included safety case and is perfectly stable on the repeated synthetic geometry, but its envelope box reduces mean localization IoU;
- weighted coordinate fusion retains the safety behavior while giving the best synthetic mean localization IoU and much lower jitter than hard NMS;
- **weighted fusion is the candidate to continue**, not a production winner;
- none of the three may be ranked for real CCTV until A5 freezes a common corpus.

## 7. Crowded / two-close-person safety

The synthetic fixture contains a high-overlap two-person case specifically to prevent a "smooth box at any cost" fusion.

Observed on that fixture:

- hard NMS: one fusion mistake and one FN;
- conservative NMM: zero fusion mistakes;
- weighted: zero fusion mistakes.

The safety comes from same-class direct-seed matching plus center-distance and size-ratio gates, not from lowering the fusion IoU alone.

This is not proof against all crowded scenes. Dense real CCTV and crossings remain mandatory post-A5 tests.

## 8. Bbox stability measurements

A1 research measures:

- center jitter;
- width jitter;
- height jitter;
- area jitter;
- temporal IoU;
- matched bbox localization IoU.

For synthetic moving-GT sequences, temporal IoU translates the previous predicted box by GT-center motion before comparison so ordinary object translation is not counted as detector jitter.

A5 independently implemented canonical GT-relative normalized bbox-stability metrics for the frozen-corpus workflow. Once the corpus exists, A1 must use/stamp the A5 common metrics for shared leaderboard evidence rather than treating the synthetic A1 units as the canonical final metric.

## 9. Pre-fusion dump and canonical replay

A1 can capture current YOLO evidence after decoder-local NMS but before final cross-pass merge.

Pre-fusion schema:

`spectratrack-detection-prefusion-v1`

Recorded provenance includes:

- source commit;
- stable video ID + video SHA-256;
- detector/model identifier + model SHA-256;
- provider;
- requested/actual detector input dimensions;
- confidence + decoder-local NMS IoU;
- person threshold;
- tile size/overlap;
- frame dimensions/timestamps;
- candidate bbox/score/class/label;
- evidence source (`full` or tile ID/region);
- policy-run count;
- actual low-level ONNX inference-call count;
- accumulated detector stage timings;
- wall time.

Any frozen pre-fusion artifact can be converted without detector inference into:

`spectratrack-detection-replay-v1`

with `hard-nms`, `conservative-nmm`, or `weighted` fusion.

Canonical replay detections contain:

- bbox;
- score;
- class_id;
- label;
- `appearance: null`.

Current A5 state explicitly records that A1 `95fe0773...` replay output with `appearance: null` is directly compatible with the A5 replay validator. Non-null appearance data requires explicit versioning, but A1 does not emit it.

This allows A2 to compare tracker candidates on identical frozen detector evidence without rerunning detector inference.

## 10. Detector backend lab

Research CLI:

`python -m spectratrack.research.vnext_detector_backend_lab`

Commands:

- `specs` — emit frozen adapter assumptions;
- `probe` — open exact ONNX, hash it and record real input/output/provider contract;
- `smoke` — execute one real image through the exact export and record timings/detections;
- `benchmark` — after A5 freeze, evaluate one backend on the exact annotated frames.

Benchmark provenance/metrics include:

- exact source commit;
- A5 corpus revision + GT SHA-256;
- backend ID;
- model SHA-256;
- provider list/priority;
- actual ONNX input/output contract;
- thresholds/preprocessing/postprocessing settings;
- preprocess/inference/postprocess/detector-wall/benchmark-wall time;
- actual inference-call count;
- person recall/precision/FP/FN;
- A5 size/tag breakdown inherited from canonical QA evaluator;
- matched bbox localization IoU;
- VRAM only if a real measurement source is supplied.

Provider list/priority is not treated as proof that every node ran on DirectML.

## 11. Detector compatibility status

| Backend | License/provenance gate | ONNX / postprocess state | DirectML state | Real A5 CCTV result |
| --- | --- | --- | --- | --- |
| Current SpectraTrack YOLO-compatible ONNX | SpectraTrack adapter is MIT; exact upstream model/weights license must be recorded separately. For Ultralytics YOLO11, research notes record AGPL-3.0 / Enterprise licensing. | Existing production-compatible raw/end-to-end decoder; letterbox + decoder-local class-aware NMS. | Existing Windows baseline prefers DirectML with CPU fallback; exact frozen model run still needs result provenance. | NOT RUN — A5 corpus absent. |
| RF-DETR Nano/Small/Medium research candidate | Research notes record Apache-2.0 for core Nano-Large code/models; exact checkpoint hash/provenance still required. Larger plus-license variants intentionally excluded from first pass. | Research adapter implements raw normalized cxcywh + logits semantics, per-class sigmoid and global query/class top-k; preprocessing follows documented RGB/resize/ImageNet normalization contract. | **UNVERIFIED** on exact SpectraTrack export/target system. ORT support is not DirectML proof. | NOT RUN — A5 corpus absent. |
| RT-DETRv2 R18/R34 research candidate | Repository recorded as Apache-2.0; exact checkpoint provenance still required. | Research adapter follows official `images + orig_target_sizes -> labels + boxes + scores` export contract. A1 found and fixed the required `orig_target_sizes=[width,height]` ordering and added a regression test. | **UNVERIFIED** on exact SpectraTrack export/target system. | NOT RUN — A5 corpus absent. |

No published COCO mAP is used as a SpectraTrack result.

## 12. Person-specialist feasibility

### YOLO-family pedestrian specialist

Technically feasible, but exact upstream checkpoint/export licensing must be selected before training/redistribution. No training started.

### RF-DETR Nano/Small pedestrian specialist

Technically feasible for research and a reasonable first pilot candidate after data/legal gates. No assumption is made that a larger model improves tiny-person CCTV performance.

### CrowdHuman

Research notes record the official dataset terms as non-commercial research/education only and prohibiting image redistribution.

A proposed conversion keeps visible-body `vbox` as the primary person target for an experiment aligned with visible-person CCTV evaluation, preserves full-body boxes only as an explicitly separate experiment, and maps ignored/masked entries to ignore rather than positives.

No dataset files are committed.

### WiderPerson

Useful candidate content-wise, but official licensing was not reliably confirmed during this pass.

Status:

**BLOCKED pending official license confirmation.**

### Training leakage rule

The A5 GOLDEN validation corpus must never be used for training.

Any future user-confirmed CCTV training split requires separate provenance/hashes and explicit overlap checking against A5 validation source/video/frame identities.

### Training-cost rule

No full training is justified before:

1. dataset/license clearance;
2. format conversion + annotation sanity checks;
3. frozen A5 validation;
4. a small overfit sanity test;
5. a 3–5 epoch pilot measuring actual throughput, peak VRAM and wall time.

No made-up training dollar/time estimate is recorded.

## 13. Tests actually run

GitHub Actions PC CI run:

`36167290573`

Validated subject HEAD:

`95fe077346a0b955726ea8e89422734e638a4452`

Result:

**SUCCESS**

Observed CI results:

- `ruff check spectratrack tests` — success, `All checks passed!`;
- compile + full pytest — **155 passed in 1.84s**;
- `python -m spectratrack.benchmark --frames 500 --targets 24` — success;
- diagnostics — success;
- ONNX Runtime providers observed: `DmlExecutionProvider,CPUExecutionProvider`;
- diagnostics reports `directml=yes`;
- self-check — `SELF_CHECK=PASS`;
- PyInstaller standalone Windows build — success;
- standalone `SpectraTrack-PC.exe --help` — success;
- standalone `SpectraTrack-PC.exe batch --help` — success;
- source/Windows package creation and artifact upload — success.

Non-blocking packaging warning:

- PyInstaller could not collect optional `onnxruntime.quantization` because Python package `onnx` was not installed. Standalone build and smoke tests still succeeded.

CI validates code/tests/build behavior. It does **not** provide real CCTV quality evidence or exact RF-DETR/RT-DETRv2 model inference evidence because those weights/corpus were not present.

## 14. Current A5 dependency status

Current observed A5 branch during this handoff:

`agent/vnext-qa @ c52abcdf3f156bd01cbe782cc16aeb0f0deffe5f`

A5 reports:

- corpus revision: **NOT ASSIGNED**;
- corpus SHA-256: **NOT AVAILABLE**;
- real CCTV baseline quality: **NOT MEASURED**;
- common A1-A4 comparison: **NOT READY**.

The expected frozen artifacts are not present:

- `pc/benchmarks/vnext/qa/corpus.manifest.json` — absent;
- `pc/benchmarks/vnext/qa/golden.jsonl` — absent.

Therefore A1 correctly has no real:

- recall/precision/FP/FN comparison for fusion;
- night/blur/compression/occlusion/high-angle result;
- backend quality ranking;
- target-hardware performance ranking;
- person-specialist validation result.

## 15. Required rerun after A5 freeze

Once A5 publishes an immutable revision/hash:

1. record A5 revision, corpus SHA-256, GOLDEN GT SHA-256 and exact input-video hashes;
2. freeze exact current-YOLO model file/hash and exact detector/tile config;
3. collect one pre-fusion dump per frozen video using the same decoder-local outputs;
4. generate hard-NMS, conservative-NMM and weighted canonical replays from those identical pre-fusion artifacts;
5. validate every replay with A5 `validate-replay`;
6. evaluate/stamp A1 fusion evidence using A5 canonical GT-relative bbox-stability/scoring settings;
7. check crowded/crossing fusion mistakes manually as well as quantitatively;
8. obtain exact licensed RF-DETR and RT-DETRv2 exports, record SHA-256/provenance, run `probe` + `smoke`;
9. benchmark current YOLO, RF-DETR and RT-DETRv2 on the exact same annotated frames/configured evaluation IoU;
10. record actual provider, preprocess/inference/postprocess/wall time and inference calls; add VRAM only from real telemetry;
11. compare generic vs any future person-specialist model on the same frozen GOLDEN only after training on a separate non-overlapping TRAIN corpus;
12. return stamped artifacts to A5 leaderboard/architect review.

## 16. Known failures / limitations

- synthetic fusion fixture is deliberately constructed and too small for a production decision;
- conservative NMM envelope geometry can make localization worse even when temporal jitter is low;
- weighted fusion safety gates are not proven on real dense crowds/crossings;
- decoder-local NMS can already discard candidate boxes before A1 cross-pass fusion sees them; A1 intentionally did not remove it without separate evidence;
- the pre-fusion collector currently targets the existing non-adaptive current-YOLO full-frame/tile path; enhancement semantics remain A3-owned;
- no exact RF-DETR or RT-DETRv2 weight file was committed or benchmarked;
- ORT/ONNX compatibility from upstream documentation does not prove DirectML execution on the target export;
- provider priority cannot prove per-node execution/fallback;
- backend person-class mapping must be verified against each exact export;
- real RX 5700 XT timing/VRAM has not been measured;
- CrowdHuman terms constrain use/redistribution;
- WiderPerson licensing remains unresolved;
- no person-specialist training has been run;
- A1 synthetic jitter units are not the A5 final normalized cross-role units.

## 17. Recommendation

Evidence supports:

**CONTINUE RESEARCH — weighted coordinate fusion is the first fusion candidate to take to the frozen real corpus.**

Reason limited to current evidence:

- it removes the constructed hard-NMS winner-flip instability;
- it preserves both people in the included high-overlap distinct-person safety case;
- unlike envelope NMM, it improves rather than worsens synthetic mean bbox localization in the fixture;
- it remains isolated and reproducible.

This is not a recommendation to change production behavior.

Backend priority after A5 freeze:

- keep current YOLO as the exact baseline;
- probe RF-DETR Nano/Small first;
- probe RT-DETRv2 R18/R34 under the same corpus/provenance rules;
- do not rank them before exact model runs.

Person-specialist work should remain feasibility-only until generic-detector failure modes are measured on the frozen GOLDEN and a separate legal TRAIN source is available.

## 18. What must NOT be integrated now

Do **not** integrate into production yet:

- weighted fusion;
- conservative NMM;
- any change to decoder-local NMS;
- RF-DETR backend replacement;
- RT-DETRv2 backend replacement;
- person-specialist weights/training path;
- synthetic thresholds as production defaults;
- any quality claim derived from `synthetic_fusion_report.json`.

The research tooling itself may be reviewed as experimental infrastructure, but production behavior stays unchanged until architect review after A5 evidence.

## 19. Architect-review readiness

**READY FOR ARCHITECT RESEARCH REVIEW.**

A1 Phase-0 deliverables are complete and reproducible:

- isolated fusion candidates;
- stress/safety fixtures;
- bbox-stability measurements;
- pre-fusion capture;
- canonical replay export;
- backend compatibility/benchmark harness;
- person-specialist feasibility gates;
- green full CI;
- no production changes.

**NOT READY FOR PRODUCTION INTEGRATION DECISION.**

The missing dependency is the frozen, human-confirmed A5 corpus plus exact backend model artifacts for real execution.


## Round 2 supplement — evidence-aware weak-person fusion

A new research-only `evidence-aware` fusion policy keeps weighted geometry but changes weak-detection acceptance. It accepts: (a) any strong detection, (b) medium-confidence detections when full-frame evidence exists, or (c) weak detections only when at least two independent inference sources corroborate the same conservative fusion group. Single weak tile-only detections are rejected. Tracker state is not used.

On the existing MOT17-04 first-600-frame development subset at person floor 0.12, the default round-2 policy measured:

- recall: `0.7499813`;
- precision: `0.6095377`;
- bbox localization IoU: `0.8231071`;
- center jitter: `1.6979 px`;
- fusion mistakes: `60`.

This is a substantial precision/stability improvement over the prior weighted/NMM candidates, but it does **not** yet preserve their recall gain. It therefore remains a development candidate only. Held-out MOT17 sequence validation and threshold-policy refinement are required before any integrator recommendation.


## Round 2 supplement — resumable public-corpus fusion runner

A1 now includes:

`python -m spectratrack.research.vnext_detection_corpus`

Purpose:

- collect full-frame + tile pre-fusion evidence from canonical A5 public JSONL records;
- resolve both MOT17 image sequences and CrowdHuman direct images using the existing A1 source-resolution helpers;
- evaluate `hard-nms`, `conservative-nmm`, `weighted`, and `evidence-aware` on identical detector evidence;
- optionally report per-video metrics;
- optionally write one prefusion artifact per logical video;
- resume only when source commit, corpus revision, GT hash, model hash, detector settings, and frame-limit settings match exactly;
- separate newly executed inference calls from inference calls represented by reused prefusion artifacts.

Research-code HEAD:

`fd9cd0f776d1abb86b7063731af9a77343fc37c8`

GitHub Actions PC CI:

- run: `36239750252`;
- conclusion: **success**;
- ruff: success;
- compile + pytest: **163 passed in 1.68s**;
- synthetic tracker benchmark: **1369.3 tracker FPS**, crossing/reappearance ID switches 0;
- diagnostics: `DmlExecutionProvider,CPUExecutionProvider`, `directml=yes`;
- self-check: `SELF_CHECK=PASS`;
- standalone Windows build, CLI smoke, packaging and artifact upload: success.

This tooling does not itself constitute held-out quality evidence. The required next run remains the fixed MOT17 held-out sequence evaluation plus CrowdHuman dense-safety evaluation on the target evidence.


## Round 2 supplement — fixed MOT17 validation split

Split artifact:

`pc/benchmarks/vnext/detection/round2_mot17_split.json`

Development:

- `MOT17-04` only.

Held out:

- `MOT17-02`
- `MOT17-05`
- `MOT17-09`
- `MOT17-10`
- `MOT17-11`
- `MOT17-13`

Rationale: the Round-2 evidence-aware policy was already tuned using MOT17-04 first-600-frame development evidence. The whole logical sequence is therefore excluded from held-out claims.

CrowdHuman validation remains a separate dense-detection safety check.

Held-out results are evaluation-only for this cycle: do not use them to silently retune the thresholds. A failed held-out gate must remain recorded as a failed candidate or trigger a separately versioned next research cycle.


## Round 2 coordinator verification — 2026-09-26 19:30 +07

### DONE — canonical replay-export fix validated

Current branch HEAD before this documentation update:

`704c9a645d276ceb1d2acd26f0dc44ad296196ce`

Replacement GitHub Actions run:

`36241884895` — **SUCCESS**

The previous lint failure (`F821 Undefined name json`) remains part of the recorded history. Commit `704c9a6...` fixes only the missing test import; the held-out policy/config is unchanged.

### NOT DONE — held-out target-PC evidence

Still required, without retuning on held-out data:

- MOT17 held-out sequences 02/05/09/10/11/13;
- hard-nms vs conservative-nmm vs weighted vs evidence-aware on identical pre-fusion evidence;
- per-sequence and aggregate precision/recall/F1/FP/FN/localization/stability/fusion/cost metrics;
- CrowdHuman validation dense-safety comparison;
- canonical replay artifacts for surviving policies.

The target-PC run must not be restarted or duplicated until local process/artifact state is positively checked.


## Round 2 public-first detection assignment — 2026-09-26

Status: **IN PROGRESS / TARGET-PC HELD-OUT STILL NOT DONE**

The fixed MOT17 development/held-out split remains unchanged. Held-out results must not be used to retune thresholds.

Immediate evidence order:

1. MOT17 held-out: hard-nms / conservative-nmm / weighted / evidence-aware on identical prefusion evidence;
2. CrowdHuman validation: dense/close-person/fusion safety, no tracking metrics;
3. after A5 freezes `nightowls-public-r1`, run surviving A1 policies on NightOwls validation without threshold retuning on the validation set.

NightOwls is a public night detector/fusion gate, not a new tuning corpus unless A5 explicitly publishes a separate train/development split.

For surviving temporal policies, emit canonical `spectratrack-detection-replay-v1` for A2.

Do not change detector families or production detector semantics in response to held-out results during this cycle.


## Round 2 execution recovery checkpoint — 2026-09-26

Starting branch/head verified before work:

`agent/vnext-detection @ bccf087df01b74bc463dcc3525e5665110ff8225`

Canonical Round-2 coordination docs were read from `coord/vnext-round2` because those coordination-only files are not copied into the specialist branch.

### Target-PC restart guard

Status: **NEEDS VERIFICATION / BLOCKED BY CONTROL CHANNEL**

Required held-out command remains unchanged and no held-out threshold/fusion retuning has occurred.

Remote target-PC device observed:

- device: `DESKTOP-64LCMQ8`;
- initial status during recovery: briefly online, then offline before process/file interrogation could complete;
- last observed remote timestamp: `2026-09-26T15:54:48.004Z`.

Because the device went offline before substantive queries completed, the following local state cannot yet be positively determined:

- active A1 held-out process;
- partial/completed `a1-round2-mot17-heldout.json`;
- `a1-round2-prefusion-mot17` contents;
- completion marker;
- run log tail;
- artifact timestamp/size/hash.

GitHub repository/issue/PR search found no uploaded held-out A1 result or prefusion artifact for:

- `a1-round2-mot17-heldout`;
- `a1-round2-prefusion-mot17`;
- evidence-aware MOT17 held-out;
- CrowdHuman Round-2 evidence-aware fusion.

Therefore **no restart is authorized yet**. When the target PC is reachable again, inspect process + artifact + marker + log + timestamp/size first and resume/reuse existing prefusion evidence when provenance matches. Only start fresh if existing state is positively absent or invalid.

### Current evidence state

- MOT17 DEV remains `MOT17-04`.
- MOT17 HELD OUT remains `02/05/09/10/11/13`.
- held-out retuning remains prohibited.
- CrowdHuman validation remains NOT DONE for the Round-2 candidate.
- `nightowls-public-r1` is not yet available to A1; NightOwls run remains pending A5 freeze.
- canonical replay export tooling is CI-validated and ready for surviving policies once held-out evidence exists.

Decision at this checkpoint:

**NEEDS MORE EVIDENCE** — no A1 policy can be accepted/rejected from held-out until target-PC artifact state is recovered and the frozen runs complete.


## Round 2 metric-completeness fix before held-out execution

Research-code HEAD after this step:

`d8253ad449f0af39bd3d13e048255fcdd28d54b3`

Reason:

The frozen held-out contract requires F1 and per-sequence compute provenance. The resumable corpus runner already produced TP/FP/FN/precision/recall and aggregate compute, but F1 was not explicit and represented detector wall/call provenance was not broken out per sequence. A target-PC inference run must not be repeated merely because the report schema omitted required fields.

Changes are research-only:

- fusion evaluation now reports explicit `f1`;
- per-sequence detector provenance now records policy runs, represented inference calls, new vs reused inference calls, represented detector wall time, new vs reused detector wall time, stage timings, source modes, and whether evidence came from fresh inference or reused prefusion;
- top-level performance sums represented/new/reused detector wall time in addition to current benchmark wall time;
- tests cover F1 and fresh-vs-reused per-sequence cost provenance.

No production detector/runtime file changed. No fusion thresholds/policy values changed. Held-out remains evaluation-only with no retuning.

Status:

**IN PROGRESS / CI VERIFICATION REQUIRED**

Target-PC execution remains **NEEDS VERIFICATION** because the device is still offline; no restart has been issued.


## Round 2 resumable-provenance self-review fix

Research-code HEAD after this step:

`530353f2ebe1010f47510426056611edabaaa22b`

Self-review found a subtle reporting error in the newly added per-sequence cost accounting: under `--resume`, a newly collected prefusion file exists by the time reporting runs, so inferring reuse from a post-write `is_file()` check could mislabel fresh inference as reused.

Fix:

- reuse state is now decided before collection and carried explicitly through the loop;
- first resumable run records `new_inference`;
- a second run over the same validated prefusion records `reused_prefusion`;
- regression test executes both passes and verifies new/reused ONNX call and represented detector-wall accounting.

Fusion policy/config/thresholds are unchanged. No production file changed.

Status:

**IN PROGRESS / CI VERIFICATION REQUIRED**

Target-PC held-out execution remains **NEEDS VERIFICATION** and has not been restarted.


## Round 2 artifact-recovery hardening before target run

Research-code HEAD after this step:

`af1220cefaa6e1ebfa3eb9c05229aa0382361efa`

Reason:

The target-PC restart contract requires process/artifact/marker/log/timestamp-size verification before any expensive rerun. The corpus runner previously wrote the final report and replay paths but did not persist hashes/size/mtime or a completion marker.

Research-only additions:

- every prefusion artifact recorded in the final report now includes path, SHA-256, size bytes, and mtime_ns;
- every canonical replay artifact records the same provenance;
- final corpus output writes a sidecar completion marker:
  `<output>.complete.json`;
- completion marker binds source commit, corpus revision, GT SHA-256, model SHA-256, final output path/hash/size/mtime;
- CLI completion output prints method results plus final output and marker provenance;
- regression tests verify prefusion/replay hashes and completion-marker binding.

This does not change fusion thresholds, detector semantics, production code, or held-out split.

Status:

**IN PROGRESS / CI VERIFICATION REQUIRED**

Target PC remains offline; held-out inference has not been restarted.


## Round 2 target-PC state positively verified before held-out start

Verification status:

**VERIFIED ABSENT — fresh held-out start is authorized.**

Target PC:

- device: `DESKTOP-64LCMQ8`;
- A1 worktree: `C:\Users\foto6\SpectraTrack-worktrees\a1`;
- worktree branch: `agent/vnext-detection`;
- worktree was clean at verification;
- worktree HEAD before this start-checkpoint commit: `bae608cc58cc90bd2d709a6c0ac1d0d1063dd6cc`.

Process/session verification:

- Remote Desktop Commander reported no active terminal sessions;
- process list contained no `python.exe` / detector benchmark process;
- PID 3108 and 20668 `python3.13.exe` were verified as VS Code Black Formatter language-server processes;
- PID 26224 `pythonw.exe` was verified as `C:\Users\foto6\SpectraTrack-control\supervisor.py run`;
- therefore no existing A1 `vnext_detection_corpus` inference process was found.

Two attempted broad PowerShell command-line filters failed due shell quoting/parsing. Those probes are recorded as **FAILED VERIFICATION PROBES**, but they were superseded by direct process enumeration plus exact PID command-line inspection and do not leave benchmark state ambiguous.

Expected old A1 Round-2 artifacts were checked directly and are absent:

- `C:\Users\foto6\SpectraTrack-data\runs\a1-round2-mot17-heldout.json`;
- `C:\Users\foto6\SpectraTrack-data\runs\a1-round2-mot17-heldout.json.complete.json`;
- `C:\Users\foto6\SpectraTrack-data\runs\a1-round2-mot17-heldout.json.console.log`;
- `C:\Users\foto6\SpectraTrack-data\runs\a1-round2-prefusion-mot17`.

The runs directory contains only earlier Round-1/dev A1 evidence and other-role artifacts; no uploaded/local Round-2 held-out result was found.

Verified target inputs:

- model: `E:\SpectraTrack\yolo11x.onnx`;
- model size: `228267957` bytes;
- model SHA-256: `e84cbad768b218d74ecc85e3e52d84631123719a6951b3ddf6eddc850d5b3f73`;
- MOT17 canonical GT: `C:\Users\foto6\SpectraTrack-data\imports\mot17-public.jsonl`;
- MOT17 GT SHA-256: `28dcb9d197e0a098a1efb097f1589177350192a8f5f1be3e2ab5cd18d8f205c7`;
- MOT17 media root: `C:\Users\foto6\SpectraTrack-data\public\MOT17\MOT17`;
- CrowdHuman canonical GT: `C:\Users\foto6\SpectraTrack-data\imports\crowdhuman-val-fbox.jsonl`;
- CrowdHuman GT SHA-256: `2576c6a1db502cef1ffd103337b7728e628d6dfca8bedb3df9d606ce2f23dd0f`;
- CrowdHuman media root: `C:\Users\foto6\SpectraTrack-data\public\CrowdHuman`.

Frozen MOT17 split remains unchanged:

- DEV: `MOT17-04`;
- HELD OUT: `MOT17-02/05/09/10/11/13`;
- held-out metrics are evaluation-only and MUST NOT retune fusion/evidence thresholds.

Held-out launch policy is unchanged:

- input size 960;
- detector confidence 0.35;
- decoder-local NMS IoU 0.45;
- person floor 0.12;
- tile 640;
- overlap 0.20;
- match IoU 0.50;
- fusion IoU 0.55;
- center ratio 0.20;
- size ratio 1.80;
- evidence weak/solo/strong = 0.12/0.20/0.35;
- evidence min sources = 2;
- methods = hard-nms / conservative-nmm / weighted / evidence-aware;
- provider preference = DirectML with baseline CPU fallback.

Planned local artifact paths:

- result: `C:\Users\foto6\SpectraTrack-data\runs\a1-round2-mot17-heldout.json`;
- completion marker: result path + `.complete.json`;
- console log: `C:\Users\foto6\SpectraTrack-data\runs\a1-round2-mot17-heldout.json.console.log`;
- prefusion: `C:\Users\foto6\SpectraTrack-data\runs\a1-round2-prefusion-mot17`;
- canonical replays: `C:\Users\foto6\SpectraTrack-data\runs\a1-round2-replays-mot17`.

Decision:

**IN PROGRESS — START HELD-OUT.** Fresh start is permitted because prior process/artifact state was positively verified absent.


## Round 2 MOT17 held-out launch

Status:

**IN PROGRESS**

Target-PC launch was authorized only after the prior process/artifact state was positively verified absent.

Exact experiment source commit:

`c5f913f7c6b007dbcedc84aa03e4235f589ee062`

Target environment:

- device: `DESKTOP-64LCMQ8`;
- worktree: `C:\Users\foto6\SpectraTrack-worktrees\a1`;
- Python: `C:\Users\foto6\SpectraTrack-env\Scripts\python.exe`;
- Python version: 3.12.10;
- ONNX Runtime: 1.24.4;
- providers available: `DmlExecutionProvider,CPUExecutionProvider`;
- OpenCV: 4.14.0;
- NumPy: 2.5.3.

Launch process:

- Remote Desktop Commander process PID: `16456`;
- corpus revision: `mot17-public-r1`;
- GT SHA-256: `28dcb9d197e0a098a1efb097f1589177350192a8f5f1be3e2ab5cd18d8f205c7`;
- model: `E:\SpectraTrack\yolo11x.onnx`;
- model SHA-256: `e84cbad768b218d74ecc85e3e52d84631123719a6951b3ddf6eddc850d5b3f73`;
- provider preference: DirectML, baseline CPU fallback;
- DEV excluded: `MOT17-04`;
- HELD OUT: `MOT17-02/05/09/10/11/13`;
- methods: `hard-nms`, `conservative-nmm`, `weighted`, `evidence-aware`;
- person floor: 0.12;
- input size: 960;
- decoder NMS IoU: 0.45;
- tile size/overlap: 640 / 0.20;
- match/fusion IoU: 0.50 / 0.55;
- evidence weak/solo/strong: 0.12 / 0.20 / 0.35;
- evidence min independent sources: 2;
- held-out retuning: FORBIDDEN.

Artifacts:

- result: `C:\Users\foto6\SpectraTrack-data\runs\a1-round2-mot17-heldout.json`;
- completion marker: `C:\Users\foto6\SpectraTrack-data\runs\a1-round2-mot17-heldout.json.complete.json`;
- log: `C:\Users\foto6\SpectraTrack-data\runs\a1-round2-mot17-heldout.json.console.log`;
- prefusion directory: `C:\Users\foto6\SpectraTrack-data\runs\a1-round2-prefusion-mot17`;
- replay directory: `C:\Users\foto6\SpectraTrack-data\runs\a1-round2-replays-mot17`.

No quality/cost claim is made until the process exits successfully and output + completion marker hashes/sizes are verified.


## Round 2 MOT17 held-out first launch failure

Status:

**FAILED LAUNCH — NO BENCHMARK INFERENCE EXECUTED**

Attempted experiment source commit:

`c5f913f7c6b007dbcedc84aa03e4235f589ee062`

Observed process:

- target process PID: `16456`;
- exit code: `1`;
- runtime: about `0.03 s`;
- console log path: `C:\Users\foto6\SpectraTrack-data\runs\a1-round2-mot17-heldout.json.console.log`;
- log size: `66` bytes;
- log created/modified: `2026-09-26T16:31:03.231Z`.

Cause:

The launcher command was assembled without the required `&&` separator between the `cd /d ...\pc` command and the Python invocation. Windows `cmd` therefore rejected the shell syntax before Python / ONNX Runtime / dataset code executed.

Post-failure artifact verification:

- result JSON: ABSENT;
- completion marker: ABSENT;
- prefusion directory: ABSENT;
- replay directory: ABSENT.

Therefore this failed launch contains **zero benchmark evidence** and cannot be interpreted as a detector/corpus failure.

Recovery rule:

- fix only the shell command separator;
- do not alter held-out split, thresholds, model, fusion config, or corpus;
- repeat only after this failure is Git-persisted.

Decision:

**RETRY AUTHORIZED — launcher-only failure, no inference/artifact state to preserve.**


## Round 2 MOT17 held-out retry launch

Status:

**IN PROGRESS — REAL BENCHMARK PROCESS RUNNING**

Exact experiment source commit:

`e2dad933119e0639413200466ffbda0baeba6b2e`

The only change from the failed first launch is the shell-command separator. Fusion policy, thresholds, corpus, split, model and provider configuration are unchanged.

Target process:

- Remote Desktop Commander PID: `21908`;
- launcher status immediately after start: running.

The failed first-launch log was preserved separately at:

`C:\Users\foto6\SpectraTrack-data\runs\a1-round2-mot17-heldout.failed-launch-1.console.log`

Current real-run log:

`C:\Users\foto6\SpectraTrack-data\runs\a1-round2-mot17-heldout.json.console.log`

All result/prefusion/replay/marker paths remain those recorded in the launch checkpoint.

Decision:

**IN PROGRESS — DO NOT START A DUPLICATE RUN.**


## Round 2 live held-out verification — existing retry preserved

Status:

**IN PROGRESS — EXISTING RUN VERIFIED, DUPLICATE RESTART FORBIDDEN**

Remote branch state observed before this checkpoint:

`agent/vnext-detection @ 2a2176e5b21d9a0d7e472da0ec5cf76c550f4f9a`

The coordinator-requested `bccf087df01b74bc463dcc3525e5665110ff8225` is a verified ancestor of the current branch; the branch is 16 commits ahead of that checkpoint. No reset/rebase/history rewrite was performed.

Target PC:

- device: `DESKTOP-64LCMQ8`;
- retry terminal PID: `21908`;
- benchmark launcher child: PID `2100`;
- benchmark Python worker: PID `21104`;
- worker state: alive/responding;
- observed working set: about `658907136` bytes;
- observed accumulated CPU at verification: about `241.9 s`;
- process creation: `2026-09-26 23:34:38 +07`.

The exact active command was re-read from the target process table and still binds:

- source commit: `e2dad933119e0639413200466ffbda0baeba6b2e`;
- corpus: `mot17-public-r1`;
- GT: canonical MOT17 public JSONL;
- model: `E:\SpectraTrack\yolo11x.onnx`;
- held out only: MOT17-02/05/09/10/11/13;
- `--resume --per-video`;
- frozen policy/config from the previous launch checkpoint.

Artifact state at verification:

- final result JSON: ABSENT;
- completion marker: ABSENT;
- current console log: present, size `0` bytes, created/modified `2026-09-26T16:34:38.691Z`;
- prefusion directory: present, currently empty, created `2026-09-26T16:34:42.807Z`;
- replay directory: ABSENT.

Interpretation:

The run is active and has not yet reached the first persisted per-sequence prefusion artifact. Empty stdout/log at this point is not treated as a failure because the Python worker is alive and accumulating CPU. No duplicate run is authorized.

Decision:

**IN PROGRESS / NEEDS COMPLETION — preserve PID 21908 run and continue artifact/process monitoring.**


## Round 2 NightOwls dependency verification

Observed A5 branch:

`agent/vnext-qa @ bf63820f81cd13fdfae8680a200e25f030d314fc`

A5 currently records:

- NightOwls importer/tooling: READY;
- deterministic slice tooling: READY;
- official validation import: NOT RUN;
- `nightowls-public-r1`: NOT FROZEN;
- real canonical JSONL/import manifest hashes: NOT AVAILABLE.

Repository checks also found no frozen `nightowls-public-r1` manifest/JSONL artifact.

Decision:

**BLOCKED ON A5 FREEZE — do not run or tune on NightOwls yet.** When A5 publishes a frozen full validation or explicitly named frozen slice, A1 must evaluate surviving policies without threshold retuning.


## Round 2 MOT17 held-out checkpoint — MOT17-02 prefusion complete

Status:

**DONE — FIRST HELD-OUT SEQUENCE PREFUSION VERIFIED**

The existing retry process remains active and has completed/persisted the first held-out sequence without any restart.

Sequence:

- corpus: `mot17-public-r1`;
- split: HELD OUT;
- logical video: `golden/public/mot17/MOT17-02`;
- source commit: `e2dad933119e0639413200466ffbda0baeba6b2e`;
- GT SHA-256: `28dcb9d197e0a098a1efb097f1589177350192a8f5f1be3e2ab5cd18d8f205c7`;
- model SHA-256: `e84cbad768b218d74ecc85e3e52d84631123719a6951b3ddf6eddc850d5b3f73`;
- provider: `DmlExecutionProvider,CPUExecutionProvider`;
- frame size: `1920x1080`;
- detector input: `960x960`;
- detector conf / decoder IoU: `0.35 / 0.45`;
- person floor: `0.12`;
- tile / overlap: `640 / 0.20`;
- decoder-local NMS: enabled;
- final cross-pass fusion in prefusion artifact: not applied.

Artifact:

`C:\Users\foto6\SpectraTrack-data\runs\a1-round2-prefusion-mot17\golden_public_mot17_MOT17-02.jsonl`

Verified artifact provenance:

- size: `13668394` bytes;
- modified UTC: `2026-09-26T16:45:11.987Z`;
- SHA-256: `31b0d5397c0ffffd6d89384648272442ca6aa1fbb7f4f7b5764a6c3a2c24523d`.

Verified prefusion summary:

- policy runs: `600`;
- actual ONNX calls: `5400`;
- source mode: `image_sequence`;
- detector wall time: `628.8552067000419 s`;
- preprocess: `42044.41860737279 ms`;
- inference: `533934.2153006 ms`;
- postprocess: `37245.49529911019 ms`.

The final held-out result/marker are not expected until all six held-out sequences complete. The live process continues into the remaining sequences; no duplicate run has been started.

Decision:

**VALID PARTIAL EVIDENCE / HELD-OUT RUN CONTINUES.**


## Round 2 MOT17 held-out checkpoint — MOT17-02 fusion metrics

Status:

**DONE — PER-SEQUENCE METRICS COMPUTED FROM VERIFIED PREFUSION, ZERO NEW ONNX CALLS**

The four frozen policies were evaluated from the verified MOT17-02 prefusion artifact only. No detector inference was repeated and no held-out thresholds were changed.

| metric | hard-nms | conservative-nmm | weighted | evidence-aware |
| --- | ---: | ---: | ---: | ---: |
| TP | 10027 | 10537 | 10453 | 9943 |
| FP | 10717 | 13822 | 13938 | 9658 |
| FN | 8554 | 8044 | 8128 | 8638 |
| precision | 0.483369 | 0.432571 | 0.428560 | 0.507270 |
| recall | 0.539637 | 0.567085 | 0.562564 | 0.535117 |
| F1 | 0.509955 | 0.490778 | 0.486503 | 0.520821 |
| bbox localization IoU | 0.770657 | 0.780299 | 0.776506 | 0.781918 |
| center jitter px | 4.083129 | 3.252418 | 3.100992 | 2.974692 |
| width jitter px | 3.916002 | 4.221506 | 3.706967 | 3.630977 |
| height jitter px | 5.417835 | 3.510400 | 3.887722 | 3.662282 |
| area jitter ratio | 0.080419 | 0.075273 | 0.067962 | 0.062632 |
| temporal IoU | 0.893901 | 0.901516 | 0.906383 | 0.912444 |
| duplicates before fusion | 20715 | 20715 | 20715 | 20715 |
| fusion mistakes | 557 | 322 | 322 | 322 |

Sequence interpretation only:

- evidence-aware has the strongest F1, precision, localization IoU, center/area jitter and temporal IoU on MOT17-02;
- evidence-aware recall is slightly below hard NMS and below NMM/weighted on this sequence;
- conservative NMM has the highest recall but materially worse precision/F1;
- no held-out decision is made from one sequence and no policy retuning is allowed.

Compute provenance is the verified prefusion checkpoint already recorded above:

- 600 policy runs;
- 5400 actual ONNX calls represented;
- detector wall time 628.8552067000419 s;
- zero new ONNX calls for this fusion-only evaluation.

Decision:

**NEEDS REMAINING HELD-OUT SEQUENCES — no ACCEPT/REJECT yet.**


## Round 2 MOT17 held-out checkpoint — MOT17-05 complete

Status:

**DONE — PREFUSION VERIFIED + PER-SEQUENCE METRICS COMPUTED, ZERO NEW ONNX CALLS FOR FUSION EVAL**

Verified prefusion artifact:

`C:\Users\foto6\SpectraTrack-data\runs\a1-round2-prefusion-mot17\golden_public_mot17_MOT17-05.jsonl`

Artifact provenance:

- size: `4593753` bytes;
- modified UTC: `2026-09-26T16:48:28.990Z`;
- SHA-256: `183ec28487671db5ec1440572632f55e87c4c45dab58ab806172b1dec9a6ea26`.

Verified compute summary:

- frame size: `640x480`;
- policy runs: `837`;
- actual ONNX calls represented: `1674`;
- detector wall time: `196.88813610002398 s`;
- preprocess: `13592.796397046186 ms`;
- inference: `166581.87309885398 ms`;
- postprocess: `11295.304001774639 ms`.

All four frozen fusion policies are metric-identical on this sequence:

| metric | all four policies |
| --- | ---: |
| TP | 4917 |
| FP | 2955 |
| FN | 2000 |
| precision | 0.624619 |
| recall | 0.710857 |
| F1 | 0.664954 |
| bbox localization IoU | 0.752755 |
| center jitter px | 4.623796 |
| width jitter px | 4.694502 |
| height jitter px | 4.467823 |
| area jitter ratio | 0.078469 |
| temporal IoU | 0.869318 |
| duplicates before fusion | 4955 |
| fusion mistakes | 0 |

Interpretation:

MOT17-05 is `640x480` while the frozen tile size is `640`; the tile region covers the full frame, so full-frame and tile evidence have identical geometry and all cross-pass policies collapse to the same result. This is a useful held-out no-regression sanity case but carries no policy discrimination.

No threshold/config retuning occurred.

Decision:

**VALID HELD-OUT EVIDENCE / RUN CONTINUES.**


## Round 2 MOT17 held-out checkpoint — MOT17-09 complete

Status:

**DONE — PREFUSION VERIFIED + PER-SEQUENCE METRICS COMPUTED, ZERO NEW ONNX CALLS FOR FUSION EVAL**

Verified prefusion artifact:

`C:\Users\foto6\SpectraTrack-data\runs\a1-round2-prefusion-mot17\golden_public_mot17_MOT17-09.jsonl`

Artifact provenance:

- size: `6766422` bytes;
- modified UTC: `2026-09-26T16:57:35.117Z`;
- SHA-256: `ef65ba4a59e8cdfa0fd26272a2a122e752a9899cbf1294d9859f16d60e793791`.

Verified compute summary:

- frame size: `1920x1080`;
- policy runs: `525`;
- actual ONNX calls represented: `4725`;
- detector wall time: `545.9547427999787 s`;
- preprocess: `39472.8775001131 ms`;
- inference: `467177.791796159 ms`;
- postprocess: `26591.237901127897 ms`.

Per-sequence held-out metrics:

| metric | hard-nms | conservative-nmm | weighted | evidence-aware |
| --- | ---: | ---: | ---: | ---: |
| TP | 4003 | 4446 | 4429 | 4400 |
| FP | 7214 | 10945 | 10980 | 9018 |
| FN | 1322 | 879 | 896 | 925 |
| precision | 0.356869 | 0.288870 | 0.287429 | 0.327918 |
| recall | 0.751737 | 0.834930 | 0.831737 | 0.826291 |
| F1 | 0.483980 | 0.429233 | 0.427221 | 0.469509 |
| bbox localization IoU | 0.734688 | 0.795355 | 0.779949 | 0.780892 |
| center jitter px | 11.541275 | 5.819352 | 6.272497 | 5.991973 |
| width jitter px | 7.222500 | 7.054980 | 6.905799 | 6.791823 |
| height jitter px | 12.614094 | 5.020469 | 5.910009 | 5.474048 |
| area jitter ratio | 0.080342 | 0.060357 | 0.059934 | 0.058484 |
| temporal IoU | 0.880651 | 0.910020 | 0.907705 | 0.909688 |
| duplicates before fusion | 8807 | 8807 | 8807 | 8807 |
| fusion mistakes | 150 | 47 | 47 | 47 |

Held-out interpretation only:

- evidence-aware materially improves recall, localization stability, temporal IoU and fusion-mistake count relative to hard NMS on MOT17-09;
- hard NMS retains higher precision and F1 on this sequence;
- conservative NMM has the highest recall/localization IoU but pays the largest FP/precision cost;
- weighted is close to conservative NMM and does not recover the precision loss;
- therefore evidence-aware is not a universal metric winner and must be judged on the full held-out aggregate without retuning.

No threshold/config retuning occurred.

Decision:

**VALID HELD-OUT EVIDENCE / NEEDS REMAINING SEQUENCES AND AGGREGATE.**


## Round 2 MOT17 held-out checkpoint — MOT17-10 complete

Status:

**DONE — PREFUSION VERIFIED + PER-SEQUENCE METRICS COMPUTED, ZERO NEW ONNX CALLS FOR FUSION EVAL**

Verified prefusion artifact:

`C:\Users\foto6\SpectraTrack-data\runs\a1-round2-prefusion-mot17\golden_public_mot17_MOT17-10.jsonl`

Artifact provenance:

- size: `9238134` bytes;
- modified UTC: `2026-09-26T17:09:04.571Z`;
- SHA-256: `360c2068084e066cdf5b21464f36ae1768b7ca8cb3befc6319fbc3ff3c2baebf`.

Verified compute summary:

- frame size: `1920x1080`;
- policy runs: `654`;
- actual ONNX calls represented: `5886`;
- detector wall time: `689.2355315999594 s`;
- preprocess: `51066.86669762712 ms`;
- inference: `585967.2606033273 ms`;
- postprocess: `36147.17189595103 ms`.

Per-sequence held-out metrics:

| metric | hard-nms | conservative-nmm | weighted | evidence-aware |
| --- | ---: | ---: | ---: | ---: |
| TP | 9311 | 9565 | 9494 | 9212 |
| FP | 10353 | 12052 | 12126 | 8246 |
| FN | 3525 | 3271 | 3342 | 3624 |
| precision | 0.473505 | 0.442476 | 0.439130 | 0.527666 |
| recall | 0.725382 | 0.745170 | 0.739639 | 0.717669 |
| F1 | 0.572985 | 0.555249 | 0.551080 | 0.608173 |
| bbox localization IoU | 0.743739 | 0.756773 | 0.750872 | 0.753961 |
| center jitter px | 4.628579 | 3.638982 | 3.644249 | 3.592393 |
| width jitter px | 3.295475 | 3.054630 | 2.881973 | 2.831969 |
| height jitter px | 5.797963 | 3.145798 | 3.600189 | 3.460468 |
| area jitter ratio | 0.087744 | 0.075028 | 0.073263 | 0.068003 |
| temporal IoU | 0.845826 | 0.858795 | 0.858279 | 0.863390 |
| duplicates before fusion | 12134 | 12134 | 12134 | 12134 |
| fusion mistakes | 92 | 64 | 64 | 64 |

Held-out interpretation only:

- evidence-aware improves precision, F1, center/width/area stability, temporal IoU and fusion mistakes relative to hard NMS;
- evidence-aware recall is lower than hard NMS on this sequence;
- conservative NMM has the highest recall but lower precision/F1;
- weighted does not recover precision versus NMM;
- no policy/config retuning occurred.

Decision:

**VALID HELD-OUT EVIDENCE / NEEDS MOT17-11, MOT17-13 AND AGGREGATE.**

## Round 2 MOT17 held-out checkpoint — MOT17-11 complete

Status:

**DONE — PREFUSION VERIFIED + PER-SEQUENCE METRICS COMPUTED, ZERO NEW ONNX CALLS FOR FUSION EVAL**

Verified prefusion artifact: `C:\Users\foto6\SpectraTrack-data\runs\a1-round2-prefusion-mot17\golden_public_mot17_MOT17-11.jsonl`; size `11882740` bytes; modified UTC `2026-09-26T17:24:42.2995317Z`; SHA-256 `52dda22d4f5d69a7e6d2d63dd0195d9586b8020038a04be54f8a6c8ba7e74f02`.

Frozen provenance is unchanged: corpus `mot17-public-r1`, HELD OUT, source commit `e2dad933119e0639413200466ffbda0baeba6b2e`, GT SHA-256 `28dcb9d197e0a098a1efb097f1589177350192a8f5f1be3e2ab5cd18d8f205c7`, model SHA-256 `e84cbad768b218d74ecc85e3e52d84631123719a6951b3ddf6eddc850d5b3f73`, provider `DmlExecutionProvider,CPUExecutionProvider`, input `960`, detector conf/decoder IoU `0.35/0.45`, person floor `0.12`, tile/overlap `640/0.20`, match/fusion IoU `0.50/0.55`, evidence weak/solo/strong `0.12/0.20/0.35`, min sources `2`.

Verified compute summary: frame size `1920x1080`; policy runs `900`; represented actual ONNX calls `8100`; detector wall time `937.4122037000488 s`; preprocess `66826.04820246343 ms`; inference `803115.2779022232 ms`; postprocess `45510.823701974005 ms`. Fusion-only evaluation used the saved prefusion artifact and added zero ONNX calls.

| metric | hard-nms | conservative-nmm | weighted | evidence-aware |
| --- | ---: | ---: | ---: | ---: |
| TP / FP / FN | 6766 / 22345 / 2670 | 6952 / 27760 / 2484 | 6941 / 27784 / 2495 | 6808 / 21290 / 2628 |
| precision / recall / F1 | 0.232421 / 0.717041 / 0.351052 | 0.200277 / 0.736753 / 0.314941 | 0.199885 / 0.735587 / 0.314350 | 0.242295 / 0.721492 / 0.362764 |
| bbox localization IoU | 0.757762 | 0.828310 | 0.817573 | 0.820990 |
| center / width / height jitter px | 8.341876 / 4.669243 / 10.503713 | 3.888914 / 4.044220 / 5.171380 | 4.125586 / 3.968566 / 6.066937 | 3.983145 / 3.908266 / 5.832358 |
| area jitter ratio / temporal IoU | 0.055734 / 0.915243 | 0.038159 / 0.938267 | 0.039005 / 0.936519 | 0.037044 / 0.939217 |
| duplicates / fusion mistakes | 10054 / 34 | 10054 / 11 | 10054 / 11 | 10054 / 11 |

Held-out interpretation only: evidence-aware has the best F1/precision/area stability/temporal IoU among the four on MOT17-11 and materially improves stability/fusion mistakes vs hard NMS; conservative NMM has the highest recall/localization IoU. No threshold/config retuning occurred.

Decision: **VALID HELD-OUT EVIDENCE / NEEDS MOT17-13 AND FULL AGGREGATE.**

## Round 2 MOT17 held-out checkpoint — MOT17-13 complete

Status:

**DONE — PREFUSION VERIFIED + PER-SEQUENCE METRICS COMPUTED, ZERO NEW ONNX CALLS FOR FUSION EVAL**

Verified prefusion artifact: `C:\Users\foto6\SpectraTrack-data\runs\a1-round2-prefusion-mot17\golden_public_mot17_MOT17-13.jsonl`; size `11703141` bytes; modified UTC `2026-09-26T17:37:48.6360118Z`; SHA-256 `cbfb254a6c13dd138a92c9c1d54dbb22e8cab3e639dc2d46f4f05ad343055741`.

Frozen provenance is unchanged: corpus `mot17-public-r1`, HELD OUT, source commit `e2dad933119e0639413200466ffbda0baeba6b2e`, GT SHA-256 `28dcb9d197e0a098a1efb097f1589177350192a8f5f1be3e2ab5cd18d8f205c7`, model SHA-256 `e84cbad768b218d74ecc85e3e52d84631123719a6951b3ddf6eddc850d5b3f73`, provider `DmlExecutionProvider,CPUExecutionProvider`, input `960`, detector conf/decoder IoU `0.35/0.45`, person floor `0.12`, tile/overlap `640/0.20`, match/fusion IoU `0.50/0.55`, evidence weak/solo/strong `0.12/0.20/0.35`, min sources `2`.

Verified compute summary: frame size `1920x1080`; policy runs `750`; represented actual ONNX calls `6750`; detector wall time `786.0482912000734 s`; preprocess `55815.77189848758 ms`; inference `667988.0271981237 ms`; postprocess `43810.955403605476 ms`. Fusion-only evaluation used saved prefusion and added zero ONNX calls.

| metric | hard-nms | conservative-nmm | weighted | evidence-aware |
| --- | ---: | ---: | ---: | ---: |
| TP / FP / FN | 8709 / 7284 / 2933 | 8992 / 9393 / 2650 | 8930 / 9461 / 2712 | 8585 / 6560 / 3057 |
| precision / recall / F1 | 0.544551 / 0.748067 / 0.630288 | 0.489094 / 0.772376 / 0.598928 | 0.485564 / 0.767050 / 0.594679 | 0.566854 / 0.737416 / 0.640983 |
| bbox localization IoU | 0.759105 | 0.772458 | 0.766620 | 0.769180 |
| center / width / height jitter px | 3.086414 / 2.538411 / 3.511091 | 2.832500 / 2.457267 / 3.075117 | 2.715979 / 2.237067 / 2.844085 | 2.668574 / 2.206511 / 2.729856 |
| area jitter ratio / temporal IoU | 0.092658 / 0.852829 | 0.086524 / 0.861262 | 0.079202 / 0.864744 | 0.076024 / 0.868155 |
| duplicates / fusion mistakes | 18916 / 183 | 18916 / 96 | 18916 / 96 | 18916 / 96 |

Held-out interpretation only: evidence-aware has the best F1/precision and strongest stability/temporal IoU on MOT17-13 while reducing fusion mistakes vs hard NMS; its recall is lower than hard NMS. Conservative NMM has the highest recall but larger FP cost. No threshold/config retuning occurred.

Decision: **VALID HELD-OUT EVIDENCE / ALL SIX PREFUSION SEQUENCES COMPLETE; NEEDS VERIFIED FINAL AGGREGATE + MARKER/REPLAYS.**

## Round 2 MOT17 held-out final handoff

Status: **DONE — HELD-OUT DECISION FROZEN; NO HELD-OUT RETUNING**

Final result: `C:\Users\foto6\SpectraTrack-data\runs\a1-round2-mot17-heldout.json`; size `46022` bytes; modified UTC `2026-09-26T17:42:06.2680236Z`; SHA-256 `94a208a44ff2e29f69575ed5c58c4f5a1720605aedaf5e713f7798234d065fa4`. Completion marker SHA-256: `fc26bc4c4431da262c35a3663af47edfb5ba62c2d37579aad816f0350f9be776`.

Frozen provenance: corpus `mot17-public-r1`; DEV `MOT17-04`; HELD OUT `MOT17-02/05/09/10/11/13`; source commit `e2dad933119e0639413200466ffbda0baeba6b2e`; GT SHA-256 `28dcb9d197e0a098a1efb097f1589177350192a8f5f1be3e2ab5cd18d8f205c7`; model SHA-256 `e84cbad768b218d74ecc85e3e52d84631123719a6951b3ddf6eddc850d5b3f73`; providers `DmlExecutionProvider,CPUExecutionProvider`; 4266 frames; 32535 actual ONNX calls; represented detector wall `3784.394112100126 s`; end-to-end benchmark wall `4043.225418200018 s`.

| policy | decision | precision | recall | F1 | FP | TP | bbox IoU | center jitter px | temporal IoU | fusion mistakes |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| hard-nms | **RETAIN CONTROL** | 0.418094 | 0.675549 | 0.516517 | 60868 | 43733 | 0.755325 | 5.436632 | 0.875081 | 1016 |
| conservative-nmm | **REJECT / NOT RETAINED** | 0.371183 | 0.701438 | 0.485468 | 76927 | 45409 | 0.779633 | 3.756440 | 0.887958 | 540 |
| weighted | **REJECT / NOT RETAINED** | 0.368963 | 0.697654 | 0.482663 | 77244 | 45164 | 0.773226 | 3.785148 | 0.889162 | 540 |
| evidence-aware | **PASS / RETAIN CHALLENGER** | 0.431776 | 0.677588 | 0.527449 | 57727 | 43865 | 0.776246 | 3.700660 | 0.892984 | 540 |

Evidence-aware vs hard-NMS: F1 +0.010931, precision +0.013683, recall +0.002039, TP +132, FP -3141, bbox IoU +0.020921, center jitter -1.735972 px, temporal IoU +0.017902, fusion mistakes -476. It clears the MOT17 held-out gate as the sole challenger.

Conservative NMM and weighted improve recall/stability but fail the aggregate trade-off: each loses >0.031 F1 and >0.046 precision versus hard-NMS while adding >16000 FP. They are not retained for Round-2 downstream gates.

Thresholds/config remain exactly frozen at person floor 0.12, fusion IoU 0.55, center ratio 0.20, size ratio 1.80, evidence weak/solo/strong 0.12/0.20/0.35 and min sources 2. Held-out results were not used to retune any value. Production detector/runtime files remain unchanged.

### Surviving canonical A2 replay artifacts

The retained MOT17 downstream pair is hard-NMS control + evidence-aware challenger. Canonical schema remains `spectratrack-detection-replay-v1`. All hashes are bound in the final result artifact.

- MOT17-02 hard/evidence: `5c16dc13a0454205078aab5e3c1e73ff011ea3b094cf40a0bd3a2fc16b7599a8` / `b9b5c674ad063e1ec589dade4c631ad0620db5e217d3bf4a90fbc0d899edd5c8`
- MOT17-05 hard/evidence: `15bf1be018c763a68faa4e7431082876350c1f112b2393bb2d80576924ebafe2` / `194f6222351073d56b5e3608d4764271854da7a81e26851aff9a24d977ce377a`
- MOT17-09 hard/evidence: `16f650bc90200ad94f91e3acc260ae36b999772502014c3ab428e3c617522e70` / `f6df4b2b43ffa5105d2b27c251e746f7b3634f45d3069549a5e75fa8ded6a987`
- MOT17-10 hard/evidence: `1bb37e91ee650c8fa4f6e92be86a7f7610e65fee1ba0e68154c3980dd96a9abe` / `313cef621be976bfa5ede97d66e57c0ab97818eed704435be02dcb36133164b4`
- MOT17-11 hard/evidence: `285486ce370e33783ad45d7a3665f46ca8ad8d0f75f2517a0e48633130a450aa` / `f2f7254b90df66d9fc250b5d09c4e9bc3221db0200364507a7d0029eb4988b57`
- MOT17-13 hard/evidence: `e8fa5697abf508f64629fc003c502b3fdc1d1b77e5e80c2b3fea13c6ef86a428` / `1ead6450d34df74e6cd0744e6e297916d1e48200692cbfb3a0cacf1e5bcd99fa`

### Remaining A1 gates / blockers

- **NOT DONE — CrowdHuman dense-safety:** mandatory next gate on frozen `crowdhuman-val-fbox-r1`; compare hard-NMS control vs evidence-aware and preserve full four-policy aggregate only as supporting evidence. Required: precision/recall/F1, TP/FP/FN, fusion mistakes, bbox localization/behavior. Tracking metrics are forbidden.
- **BLOCKED — NightOwls:** do not run A1 until A5 publishes frozen `nightowls-public-r1` or an explicitly frozen/hash-bound validation slice. No NightOwls held-out threshold tuning is allowed.
- **NOT DONE — final A2 handoff after dense-safety:** MOT17 replay bytes already exist; evidence-aware remains provisional for downstream use until CrowdHuman confirms close-person/dense-crowd safety.
- **DEFERRED — private CCTV sanity gate:** remains downstream of public evidence and A5 human-confirmed freeze; no private review is requested from A1 now.
- **LOCKED — production integration:** no changes to production detector, `integration`, `main`, RC, or release are authorized from this handoff.

Next step: run CrowdHuman dense-safety only after process/artifact/marker/log state is positively verified. No threshold retuning.

## Round 2 CrowdHuman dense-safety restart guard

Status: **VERIFIED ABSENT — FRESH START AUTHORIZED**

Before launch, target-PC state was checked explicitly. No active `python.exe` running `spectratrack.research.vnext_detection_corpus` was present. Expected CrowdHuman Round-2 result, completion marker, console log and prefusion directory were all absent, so there is no prior A1 CrowdHuman job/artifact state to preserve.

Verified clean worktree HEAD before this checkpoint: `68fd5f0459b5c0db3cbd128254aa78ab4e98da74`.

Frozen inputs:

- corpus revision: `crowdhuman-val-fbox-r1`;
- GT: `C:\Users\foto6\SpectraTrack-data\imports\crowdhuman-val-fbox.jsonl`;
- GT size: `42766074` bytes;
- GT SHA-256: `2576c6a1db502cef1ffd103337b7728e628d6dfca8bedb3df9d606ce2f23dd0f`;
- media root: `C:\Users\foto6\SpectraTrack-data\public\CrowdHuman`;
- validation images: `4370`;
- scored person boxes: `99481`;
- model: `E:\SpectraTrack\yolo11x.onnx`;
- model size: `228267957` bytes;
- model SHA-256: `e84cbad768b218d74ecc85e3e52d84631123719a6951b3ddf6eddc850d5b3f73`.

Run semantics remain frozen: input 960, detector conf 0.35, decoder-local NMS IoU 0.45, person floor 0.12, tile 640, overlap 0.20, match/fusion IoU 0.50/0.55, center ratio 0.20, size ratio 1.80, evidence weak/solo/strong 0.12/0.20/0.35, evidence min sources 2, methods hard-nms/conservative-nmm/weighted/evidence-aware, DirectML preference with CPU fallback.

CrowdHuman is detection-only dense/close-person safety evidence. No tracking/ID metrics and no replay handoff are produced from this corpus. No threshold retuning is permitted.

Decision: **START CROWDHUMAN DENSE-SAFETY AFTER THIS CHECKPOINT IS COMMITTED.**
