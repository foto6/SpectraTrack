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
