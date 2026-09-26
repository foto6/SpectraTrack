# A5 — vNext CCTV QA / Experiment Control Handoff

Branch:

`agent/vnext-qa`

Exact research base:

`vnext-base @ d03af3ae6425d3ea2e4d52e25389fecc09957394`

Immutable product baseline:

`integration @ 3ebc4d50213593cac62b97399447fccf6bbc1755`

Validated code/docs HEAD before this state-only handoff commit:

`162e4af576189b31a5da34a7fb30fe6f387ee8ac`

No A1-A4 branch was merged or cherry-picked into A5. `integration`, `main`, RC and other agent branches were not modified.

## 1. Scope and result

A5 remains an independent evaluator / experiment-control role.

Implemented:

- canonical human-confirmed CCTV corpus workflow built around the existing `qa_benchmark.py` JSONL ground truth;
- physical/logical TRAIN vs GOLDEN separation;
- source-video/hash/dimension validation;
- deterministic frozen corpus manifest + corpus SHA-256;
- local frame extraction and human box annotation helper;
- additive QA metrics for bbox stability, track continuity and compute cost without changing result schema v1;
- strict validation for `spectratrack-detection-replay-v1`;
- stamped end-to-end QA results;
- stamped specialist A1-A4 evidence tied to its real source artifact hash;
- reproducible common leaderboard that rejects corpus/scoring/provenance mismatches and deliberately does not rank candidates.

Not implemented:

- detector/fusion candidate;
- tracker candidate;
- enhancement candidate;
- scheduler candidate;
- automatic ground-truth generation.

## 2. Corpus format

Ground truth remains the existing canonical `qa_benchmark.py` JSONL format.

Recommended local layout:

```text
pc/benchmarks/vnext/qa/
  videos/
    train/
    golden/
  frames/
  train.jsonl
  golden.jsonl
  corpus.manifest.json
  results/
  replays/
  leaderboard.json
  leaderboard.md
```

Large user videos, extracted PNGs, replays and generated results are gitignored.

Required GOLDEN coverage:

- `tiny_person`
- `distant_person`
- `normal_person`
- `partial_occlusion`
- `heavy_occlusion`
- `night_dark`
- `motion_blur`
- `compression`
- `high_angle_cctv`
- `crossing_people`
- `camera_motion`
- `negative`

Coverage semantics are evidence-backed:

- person-specific categories require at least one scored person;
- `crossing_people` requires at least two scored people;
- bbox height <24 px also supplies `tiny_person`;
- `negative` means no person objects at all, not “all persons ignored”;
- empty tagged frames cannot fake person-specific coverage.

Exact SHA duplication across TRAIN/GOLDEN is rejected. Re-encoded/cropped semantic overlap cannot be detected reliably by SHA and remains prohibited by workflow.

## 3. Annotation workflow

Canonical workflow:

```text
user CCTV
  -> lossless review-frame extraction
  -> optional pre-annotation draft
  -> human visual confirmation/correction
  -> canonical qa_benchmark JSONL
  -> corpus validation
  -> human-confirmed freeze
  -> immutable corpus revision/hash
```

Frame extraction:

```powershell
cd pc
python -m spectratrack.vnext_qa extract-frames `
  --video benchmarks/vnext/qa/videos/golden/night_gate_01.mp4 `
  --video-id golden/night_gate_01.mp4 `
  --output-dir benchmarks/vnext/qa/frames/night_gate_01 `
  --every-seconds 0.5
```

Human annotation:

```powershell
python -m spectratrack.vnext_qa annotate `
  --batch benchmarks/vnext/qa/frames/night_gate_01/frames.json `
  --output benchmarks/vnext/qa/golden.jsonl `
  --tags "night_dark,high_angle_cctv,tiny_person,compression"
```

Annotator controls:

- drag: person bbox;
- `i`: stable person ID;
- `a`: per-person attributes;
- `g`: toggle `ignore=true`;
- `u`: undo;
- `c`: clear;
- Enter/`n`: save+next;
- `b`: previous;
- `q`: save+quit.

Per-frame attributes/ignore state is isolated so it cannot silently leak from one frame/person to the next.

Auto-label output is never ground truth by itself. Existing pre-annotation boxes may be loaded for correction, but freeze requires explicit human-review confirmation.

## 4. Corpus validation and frozen revision

Validate:

```powershell
python -m spectratrack.vnext_qa validate-corpus `
  --video-root benchmarks/vnext/qa/videos `
  --train-ground-truth benchmarks/vnext/qa/train.jsonl `
  --golden-ground-truth benchmarks/vnext/qa/golden.jsonl `
  --output benchmarks/vnext/qa/corpus.validation.json
```

Checks include:

- duplicate video+frame rows;
- non-integer/negative frame IDs;
- malformed/non-positive bboxes;
- duplicate object IDs;
- invalid labels;
- missing videos;
- bbox vs real source dimensions;
- annotated frame index vs source frame count when available;
- source SHA-256/dimensions/FPS/frame count;
- required GOLDEN coverage;
- exact TRAIN/GOLDEN SHA leakage.

Freeze:

```powershell
python -m spectratrack.vnext_qa freeze-corpus `
  --video-root benchmarks/vnext/qa/videos `
  --train-ground-truth benchmarks/vnext/qa/train.jsonl `
  --golden-ground-truth benchmarks/vnext/qa/golden.jsonl `
  --revision cctv-golden-r1 `
  --reviewer "<human reviewer>" `
  --confirm-human-reviewed `
  --output benchmarks/vnext/qa/corpus.manifest.json
```

Manifest hash is deterministic and revalidated on load.

### Current frozen corpus

Corpus revision: **NOT ASSIGNED**

Corpus SHA-256: **NOT AVAILABLE**

Reason: no real user CCTV set has been supplied, human-confirmed and frozen during this cycle. A5 did not substitute synthetic fixtures or random public footage for the requested human-confirmed GOLDEN set.

Therefore the common comparative run is not yet unlocked.

## 5. Detection replay validation

Canonical schema:

`spectratrack-detection-replay-v1`

Command:

```powershell
python -m spectratrack.vnext_qa validate-replay `
  --replay benchmarks/vnext/qa/replays/a1-baseline.jsonl `
  --output benchmarks/vnext/qa/replays/a1-baseline.validation.json
```

A5 validates:

- full 40-hex source commit;
- video identity;
- optional video SHA-256;
- detector/model identifier;
- optional model SHA-256;
- provider;
- complete config object;
- metadata dimensions;
- unique strictly increasing frame indices;
- frame dimensions equal replay metadata;
- finite timestamps;
- bbox inside dimensions;
- finite score/class/label;
- versioned non-null appearance metadata;
- replay file SHA-256 + config SHA-256.

Observed cross-agent compatibility:

- A1 `agent/vnext-detection @ 95fe077346a0b955726ea8e89422734e638a4452` already emits the canonical replay schema with `appearance: null`; that path is directly compatible with A5.
- A2 `agent/vnext-tracking @ ce6d3d1c07e702abd047eef4e523af151e0db9cd` parses the same canonical replay and may serialize numeric appearance vectors. A5 accepts a numeric vector only when replay config declares explicit `appearance_schema`; unversioned appearance vectors are rejected per vNext research rules.
- A2's parser is intentionally more permissive about short source strings/hashes than A5. Common comparative evidence must pass the stricter A5 provenance gate.

A2 may add research-only frame extensions such as `detector_ran`, `camera_motion` and `camera_transform`; A5 does not reject those extensions.

## 6. Metrics

Existing detection metrics remain:

- TP / FP / FN;
- precision / recall;
- recall by bbox height: <24, 24-47, 48-95, >=96;
- frame-tag breakdown;
- object-attribute recall;
- NEW FALSE NEGATIVE comparison gate.

Additive vNext metrics under unchanged result schema v1:

### Bbox stability

For matched objects with stable GT IDs, predicted boxes are normalized relative to each frame's GT center/width/height before temporal differencing.

Reported for detections and tracks:

- normalized center jitter mean;
- width log jitter mean;
- height log jitter mean;
- area log jitter mean;
- temporal IoU mean;
- pair count.

This reduces genuine object/camera translation and scale from the jitter signal. Sparse annotations still limit temporal interpretation.

### Tracking

- track recall;
- ID switches;
- fragmentation;
- mean uninterrupted track length in annotated frames;
- uninterrupted segment count;
- mean/max recovery latency in source-frame index distance;
- recovery event count.

### Performance / provenance

- input video SHA-256 and dimensions;
- actual ONNX inference calls;
- ONNX calls/frame;
- wall time;
- processed source seconds when FPS is known;
- processing seconds/source second;
- existing detector policy timing/FPS;
- externally measured VRAM only when supplied; no estimate is fabricated.

## 7. Benchmark commands

Canonical baseline run remains `qa_benchmark`:

```powershell
python -m spectratrack.qa_benchmark run `
  --ground-truth benchmarks/vnext/qa/golden.jsonl `
  --video-root benchmarks/vnext/qa/videos `
  --model ..\models\yolo11n.onnx `
  --run-name BASELINE_CURRENT `
  --revision <exact-40hex-subject-commit> `
  --output benchmarks/vnext/qa/results/baseline-current.json
```

Bind a canonical QA result to the frozen corpus:

```powershell
python -m spectratrack.vnext_qa stamp-result `
  --manifest benchmarks/vnext/qa/corpus.manifest.json `
  --result benchmarks/vnext/qa/results/baseline-current.json `
  --role Baseline `
  --experiment current `
  --output benchmarks/vnext/qa/results/baseline-current.stamped.json
```

A raw result is accepted only when GT hash, exact GOLDEN video set/SHA/dimensions, model hash/provider, exact source commit, settings and evaluation provenance are present.

## 8. Specialist A1-A4 evidence

Research-specific outputs do not have to pretend to be end-to-end QA results.

A5 supports:

`spectratrack-vnext-evidence-v1`

Required normalized evidence includes:

- role / experiment / scope;
- full source commit;
- frozen corpus revision/hash;
- GOLDEN GT hash;
- exact original source-artifact SHA-256;
- full candidate config;
- common evaluation settings;
- actually measured quality fields;
- actually measured compute fields;
- exact full GOLDEN input-video provenance;
- replay SHA where relevant;
- model ID/hash/provider where inference was used.

Stamp command:

```powershell
python -m spectratrack.vnext_qa stamp-evidence `
  --manifest benchmarks/vnext/qa/corpus.manifest.json `
  --evidence benchmarks/vnext/qa/results/a2-tracking.evidence.json `
  --source-artifact benchmarks/vnext/qa/results/a2-tracking.raw.json `
  --output benchmarks/vnext/qa/results/a2-tracking.stamped.json
```

A5 hashes the supplied source artifact itself and rejects a mismatch.

Specialist evidence that covers only a favorable subset of GOLDEN videos is rejected from the common leaderboard.

## 9. Leaderboard format

Command after common reruns:

```powershell
python -m spectratrack.vnext_qa leaderboard `
  --manifest benchmarks/vnext/qa/corpus.manifest.json `
  --run benchmarks/vnext/qa/results/baseline-current.stamped.json `
  --run benchmarks/vnext/qa/results/a1-fusion.stamped.json `
  --run benchmarks/vnext/qa/results/a2-tracking.stamped.json `
  --run benchmarks/vnext/qa/results/a3-enhancement.stamped.json `
  --run benchmarks/vnext/qa/results/a4-scheduler.stamped.json `
  --output-json benchmarks/vnext/qa/leaderboard.json `
  --output-md benchmarks/vnext/qa/leaderboard.md
```

Rows contain:

- role / experiment / explicit measurement scope;
- source commit;
- model hash/provider where applicable;
- settings hash plus full settings in JSON;
- recall / precision / FN / FP;
- four person-height recall bins;
- tracking recall / ID switches / fragmentation;
- uninterrupted track length / recovery latency;
- bbox stability;
- ONNX calls / calls per frame;
- wall seconds / processing seconds per source second;
- VRAM only when trustworthy.

Leaderboard rejects:

- different corpus revision/hash;
- different frozen GOLDEN video bytes/dimensions;
- incompatible scoring/evaluation settings;
- modified stamped payload/settings/evaluation;
- duplicate role+experiment rows.

Configuration differences are allowed only when explicit and hashed; they are not hidden.

The table intentionally does not compute a synthetic total score, sort candidates by “best”, or declare a winner.

## 10. Current baseline numbers

Real CCTV quality baseline: **NOT MEASURED**

Real CCTV processing seconds/source second for this frozen corpus: **NOT MEASURED**

Real corpus ONNX calls/frame: **NOT MEASURED**

Real ID-switch / fragmentation / jitter values: **NOT MEASURED**

Target-GPU VRAM: **NOT MEASURED**

The known field observation `people-recall + adaptive ~= 120 processing seconds/source second` remains a blocker observation with incomplete provenance; it is not promoted here to a calibrated A5 baseline.

Synthetic CI tracker throughput is only regression/smoke evidence and is not CCTV quality/performance evidence.

## 11. Missing evidence / limitations

Blocking evidence:

1. real user CCTV covering all required categories;
2. human confirmation of GOLDEN annotations;
3. first frozen corpus manifest/revision/hash;
4. same-corpus baseline run;
5. A1-A4 reruns on the exact frozen evidence;
6. target-PC performance evidence for any real-time/DirectML conclusion.

Known tooling limitations:

- exact SHA detects byte-identical TRAIN/GOLDEN leakage, not re-encoded/cropped overlap from the same recording;
- ID/stability metrics need stable GT IDs and are strongest on densely annotated temporal segments;
- recovery latency is source-frame-index distance between last match and recovery;
- OpenCV GUI annotator was code/CI validated but not manually exercised on real user CCTV in this cycle;
- A2's current real replay runner outputs track frames/replay SHA but does not itself provide a full common GT-scored result + explicit full tracker config. For common leaderboard use it must return full config/provenance and then be normalized/stamped by A5;
- A3/A4 synthetic/structural reports are not eligible common quality rows until they rerun against the entire frozen GOLDEN set.

## 12. Instructions for A1-A4 after freeze

Observed branch heads at final synchronization snapshot:

- A1 detection: `95fe077346a0b955726ea8e89422734e638a4452`
- A2 tracking: `ce6d3d1c07e702abd047eef4e523af151e0db9cd`
- A3 enhancement: `86b1460b6f9879038de92eb1e1341a96221e6c03`
- A4 performance: `1cd9f38a09d35f3b125c4a214d8514200ee0f76d`

### A1

- run baseline/fusion candidates on the exact frozen GOLDEN;
- use exact model SHA/provider/config;
- emit canonical replay for every GOLDEN video;
- use full source commit SHA;
- pass A5 replay validation;
- report final quality + actual ONNX calls/wall time;
- return raw research artifact plus stamped evidence/result.

### A2

- consume exactly the same validated A1 replay bytes for all tracker candidates;
- no detector inference in tracker-only comparison;
- report exact replay SHA, candidate commit, full tracker config;
- report track recall, switches, fragmentation, uninterrupted length, recovery latency, stability, wall/CPU cost;
- do not modify replay detections;
- version any non-null appearance vector via `config.appearance_schema`;
- return raw tracker result + normalized/stamped A5 evidence.

### A3

- use the same frozen GOLDEN and exact A1/model provenance;
- report operation/gate config, activation frequency, recovered/lost GT, FP effect, bbox effect and actual raw/enhanced ONNX calls;
- clearly label ROI/pre-fusion-only metrics;
- full common leaderboard evidence must cover the whole GOLDEN set rather than only selected favorable ROIs;
- return raw profiler artifact + normalized/stamped A5 evidence.

### A4

- run scheduler candidates against the same full GOLDEN evidence and exact detector/enhancement configuration;
- report actual ONNX calls, calls/frame, wall time, processing seconds/source second and quality metrics;
- structural simulation alone stays outside the common quality leaderboard;
- no cheaper candidate gets a GO if recall/FN/discovery evidence regresses silently;
- return raw scheduler/perf artifact + normalized/stamped A5 evidence.

## 13. Instructions for the user

The user should not edit manifests/leaderboard internals manually.

Minimum workflow:

1. put real CCTV files under `pc/benchmarks/vnext/qa/videos/golden/` (and separate tuning footage under `videos/train/`);
2. use `extract-frames` to select representative frames;
3. run `annotate` and draw person boxes, keeping the same ID for the same person across nearby frames;
4. include true negative frames and all required conditions;
5. run `validate-corpus`;
6. visually re-check every scored GOLDEN frame;
7. run `freeze-corpus --confirm-human-reviewed`;
8. provide agents the immutable `corpus.manifest.json`, `golden.jsonl`, corpus revision/hash and access to the exact referenced local video bytes;
9. return generated result/replay/evidence JSON files from candidate reruns to A5.

Full copy/paste commands and controls are in:

`pc/benchmarks/vnext/qa/README.md`

## Validation actually run

Validated code/docs HEAD:

`162e4af576189b31a5da34a7fb30fe6f387ee8ac`

GitHub Actions PC CI:

- workflow run: `36168095145`
- job: `108180668997`
- conclusion: **success**
- Ruff: **PASS**, `All checks passed!`
- compileall + pytest: **158 passed in 1.98s**
- synthetic tracker benchmark: **PASS**
  - 500 frames
  - 24 targets
  - 11970 observations
  - elapsed 0.360 s
  - 1387.8 tracker_fps
  - 24 active tracks
- diagnostics: **PASS**
- self-check: **PASS**, `SELF_CHECK=PASS`
- PyInstaller standalone build: **PASS**
- `SpectraTrack-PC.exe --help`: **PASS**
- `SpectraTrack-PC.exe batch --help`: **PASS**
- source artifact upload: **PASS**
- Windows artifact upload: **PASS**

The synthetic tracker FPS above is CI smoke/performance only, not a vNext CCTV benchmark number.

## Files changed by A5

- `DECISIONS.md`
- `pc/spectratrack/qa_benchmark.py`
- `pc/spectratrack/vnext_qa.py`
- `pc/tests/test_qa_benchmark.py`
- `pc/tests/test_vnext_qa.py`
- `pc/benchmarks/vnext/qa/README.md`
- local-artifact `.gitignore` files under `pc/benchmarks/vnext/qa/{videos,frames,replays,results}/`
- this role-state file

No production detector/tracker/enhancement/scheduler candidate implementation was added.

## Readiness

A5 corpus/evaluator/experiment-control tooling: **READY FOR CORPUS INTAKE**

Frozen human-confirmed corpus: **NOT READY / NO USER DATA YET**

Common A1-A4 comparative run: **NOT READY**

The exact unlock condition is:

1. human-confirmed GOLDEN passes validation;
2. frozen manifest/revision/hash is published;
3. baseline + A1-A4 rerun on those exact bytes/settings;
4. every row passes A5 provenance/comparability stamping.

Only after that can A5 assemble the evidence table for architect review. A5 does not declare the winner.

# Public dataset import supplement — MOT17 / CrowdHuman

Validated supplement code/docs HEAD before this state-only commit:

`a0bd8d6acb947853fff1a5487be20326bf5615b9`

GitHub Actions run:

`36221141511` — **SUCCESS**

Observed validation:

- Ruff: PASS — `All checks passed!`
- compileall + pytest: **163 passed in 1.62s**
- synthetic tracker benchmark: PASS — 500 frames / 24 targets / 11970 observations, 1446.3 tracker_fps
- diagnostics: PASS
- self-check: PASS
- PyInstaller standalone build: PASS
- standalone app help + batch help: PASS
- source/Windows artifact packaging and upload: PASS

The tracker throughput above is synthetic CI evidence only, not a MOT17/CrowdHuman performance result.

## Supplement objective

Reduce manual GT work without changing the A5 QA architecture.

Added isolated public-dataset import tooling that converts official annotations into the existing canonical `qa_benchmark.py` JSONL. There is still one evaluator.

Commands:

- `python -m spectratrack.vnext_qa import-mot17 ...`
- `python -m spectratrack.vnext_qa import-crowdhuman ...`

Public dataset binaries and generated import artifacts remain local/gitignored.

## Terms / policy checked before implementation

MOT17 / MOTChallenge:

- dataset terms: CC BY-NC-SA 3.0;
- non-commercial use;
- attribution/share-alike requirements.

CrowdHuman:

- non-commercial research/education only;
- images may not be redistributed;
- importer accepts validation annotations only for A5 detector evaluation.

Import requires explicit `--acknowledge-terms`.

## MOT17 conversion

Default source directories use the FRCNN copy of the seven unique train sequences so identical physical sequences are not imported three times.

Selected base sequences by default:

`02,04,05,09,10,11,13`

Preserved provenance:

- exact source sequence, for example `MOT17-10-FRCNN`;
- original 1-based frame number as `source_frame`;
- source FPS;
- every source image SHA-256;
- `seqinfo.ini` SHA-256;
- `gt.txt` SHA-256;
- importer exact commit;
- conversion settings.

Scoring conversion:

- class 1 pedestrian with mark > 0 -> scored `person`, stable ID `MOT17-XX:<id>`;
- classes 2/7/8/12 -> canonical ignored person regions even when the MOT consider/ignore flag is 0;
- zero-marked pedestrians -> omitted;
- unrelated classes -> omitted;
- 1-based MOT xywh -> 0-based xyxy without clipping;
- full-body boxes may extend outside image; target-like boxes with no image intersection are omitted because canonical QA cannot observe or match them.

Visibility is preserved as numeric source metadata + numeric visibility-bin attributes. A5 does not invent partial/heavy-occlusion semantic labels from visibility.

Semantic sequence tags are added only from official sequence descriptions:

- MOT17-04: night + elevated/high-angle;
- MOT17-05: camera motion;
- MOT17-10: night + camera motion;
- MOT17-11: camera motion;
- MOT17-09: low angle;
- no inferred condition tags for 02/13.

MOT17 stable IDs enable A5 tracking/identity/temporal metrics.

## CrowdHuman conversion

A5 accepts only `annotation_val.odgt`.

Default evaluation bbox:

`fbox` / full body.

Visible-body `vbox` is an explicit opt-in and must be frozen under a separate corpus revision.

Conversion:

- `tag=person` + not ignored -> scored person;
- `tag=mask` OR `extra.ignore=1` -> canonical ignored person region;
- fbox/vbox xywh -> xyxy without clipping;
- fully non-intersecting target-like boxes are omitted because canonical QA cannot observe or match them;
- original fbox/vbox/box_id/occ remain in source provenance;
- integer `extra.occ` is exposed literally as `crowdhuman_occ_<value>`;
- visible/full area ratio is exposed only as explicit numeric visibility-bin attributes;
- no scene/CCTV tags are inferred;
- no stable canonical person IDs are created.

Because CrowdHuman validation consists of independent images, A5 tracking metrics now score only GT objects with stable IDs; CrowdHuman therefore reports no track recall / switches / fragmentation.

## Source abstraction

Canonical QA JSONL gained only optional provenance fields:

- `source`
- `source_frame`
- `source_sequence`
- `source_fps`
- `allow_out_of_bounds`

Old JSONL remains valid.

The canonical runner now accepts:

- ordinary video;
- image-sequence directory;
- single still image.

This permits MOT17/CrowdHuman to stay in original local dataset layouts instead of being converted to synthetic videos.

## Import manifest / freeze provenance

Public importer provenance schema:

`spectratrack-public-dataset-import-v1`

It is not a second GT format. It only binds conversion provenance.

Each import manifest records:

- dataset name/version/split;
- terms/license metadata + acknowledgement;
- selected sequences/split;
- conversion settings;
- importer exact source commit;
- source file path/role/SHA-256/byte length;
- aggregate source provenance hash;
- converted canonical JSONL SHA-256;
- deterministic import-manifest SHA-256.

`validate-corpus --dataset-import ...` re-hashes every referenced source file before accepting the public import.

`freeze-corpus` embeds the verified import provenance in the existing frozen corpus manifest.

Public official GT can be frozen with:

`--confirm-public-dataset-terms`

Private user GT still requires:

`--confirm-human-reviewed`

## Canonical scoring caveat

Imported annotations are evaluated by SpectraTrack's common A5 matcher/ignore semantics.

They must not be described as bit-for-bit official MOTChallenge or CrowdHuman benchmark results unless those official evaluators are run separately.

## Recommended hybrid evidence suite

Keep separate immutable corpus revisions rather than hiding different semantics in one score:

1. `mot17-public-r1`
   - tracking, IDs, temporal boxes, pedestrian detection;
2. `crowdhuman-val-fbox-r1`
   - dense person/crowd/full-body detection and occlusion attributes;
3. `cctv-golden-r1`
   - small private human-confirmed user CCTV holdout.

Public data reduces manual annotation substantially but is **not** a replacement for private CCTV domain validation.

A2 should use MOT17 + private CCTV for identity conclusions; CrowdHuman is detection-only.

## Files added/changed in this supplement

Added:

- `pc/spectratrack/public_dataset_import.py`
- `pc/tests/test_public_dataset_import.py`
- `pc/benchmarks/vnext/qa/PUBLIC_DATASETS.md`
- `pc/benchmarks/vnext/qa/public/.gitignore`
- `pc/benchmarks/vnext/qa/imports/.gitignore`

Extended without creating a second evaluator:

- `pc/spectratrack/qa_benchmark.py`
- `pc/spectratrack/vnext_qa.py`
- `pc/tests/test_vnext_qa.py`
- `pc/benchmarks/vnext/qa/README.md`
- `DECISIONS.md`

## Real public dataset measurement status

Actual MOT17 download/import/run in this agent environment: **NOT RUN**

Actual CrowdHuman download/import/run in this agent environment: **NOT RUN**

Reason: datasets are large, externally licensed/terms-controlled, and the user explicitly requested that they not be downloaded/committed automatically.

Importer semantics were tested with small synthetic on-disk fixtures matching the public annotation/layout formats. CI passed all 163 tests.

Therefore:

- no real MOT17 corpus revision/hash is claimed yet;
- no real CrowdHuman corpus revision/hash is claimed yet;
- no public-dataset quality numbers are claimed yet.

Exact user commands are documented in:

`pc/benchmarks/vnext/qa/PUBLIC_DATASETS.md`

## Supplement readiness

Public import tooling: **READY FOR LOCAL DATASET INTAKE**

Private CCTV workflow: unchanged.

Common product GO: still requires the private human-confirmed CCTV holdout in addition to public benchmark evidence.



## Round 2 supplement — private CCTV review ergonomics

Round-2 coordination keeps public evidence and private human-reviewed CCTV evidence separate.

The annotation command now accepts an optional `--draft <jsonl>` input. Draft rows seed frames that have not yet been saved by the human annotator, while any already-reviewed output row always takes precedence. The draft is never treated as human-confirmed ground truth by itself and the frozen-corpus confirmation gate remains unchanged.

A local review pack may therefore use current-detector boxes to reduce drawing work without silently promoting machine labels into GOLDEN. Large videos, extracted frames, draft labels, and the reviewed private corpus remain local/generated artifacts rather than repository content.


## Round 2 coordinator verification — 2026-09-26 19:30 +07

### DONE — private review progress accounting

Validated code/test HEAD before this documentation update:

`afc6baf1cc0216ec21fa71f298be88e76519cdfa`

GitHub Actions run:

`36241763043` — **SUCCESS**

New research-only review-progress tooling distinguishes human-saved rows from draft preannotations. Draft rows never count as reviewed. The report records reviewed/pending frame counts, pending frames with/without draft boxes, reviewed person boxes, reviewed negative frames, completion fraction, and pending frame keys. Duplicate review-batch frame keys are rejected.

### NOT DONE — human confirmation

The existing 46-frame private review pack remains non-GOLDEN until a human reviews/corrects it. `cctv-golden-r1` must not be frozen from AI-only draft annotations.
