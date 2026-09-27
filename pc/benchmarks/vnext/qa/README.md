# SpectraTrack vNext CCTV QA / Golden Set

This directory is the A5 research workspace for a **small, human-confirmed, reproducible CCTV corpus**.

To reduce manual annotation, A5 also supports isolated import of official **MOT17 train GT** and
**CrowdHuman validation GT** into this same canonical QA format. See
`benchmarks/vnext/qa/PUBLIC_DATASETS.md`. Public benchmarks supplement the private CCTV holdout;
they do not replace it.

It extends the existing `spectratrack.qa_benchmark` workflow. It does **not** replace the canonical JSONL ground-truth format and does not treat model output as ground truth.

## Non-negotiable split

Keep two physically separate source groups:

```text
pc/benchmarks/vnext/qa/
  videos/
    train/
      ...
    golden/
      ...
  frames/
    ...
  train.jsonl
  golden.jsonl
  corpus.manifest.json
  results/
  replays/
  leaderboard.json
  leaderboard.md
```

- `TRAIN`: may be used for threshold/model/scheduler tuning.
- `GOLDEN`: validation only. Do not train/tune against it.
- A source video must never appear in both splits.
- The validator rejects exact SHA-256 duplicates across TRAIN/GOLDEN.
- Do not create overlapping train/golden clips from the same source recording; SHA checks cannot detect semantic overlap after re-encoding/cropping.

Large videos, extracted frames, generated results, and replays stay local/ignored.

## Required GOLDEN coverage

The first usable frozen revision must cover all of:

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
- negative frames with no people

A frame may have multiple tags.

Person-specific coverage is evidence-backed: `tiny_person`, `distant_person`, `normal_person`, `partial_occlusion`, and `heavy_occlusion` count only when the frame has at least one scored person box. `crossing_people` requires at least two scored people. A tag on an empty frame cannot satisfy those categories.

The validator also treats a scored person with bbox height < 24 px as `tiny_person`, and an annotated frame with no person objects at all as `negative`. A frame containing only `ignore=true` people is not a true negative.

## 1. Put user videos in the corpus folders

Example:

```text
pc/benchmarks/vnext/qa/videos/golden/night_gate_01.mp4
pc/benchmarks/vnext/qa/videos/train/night_gate_tuning_01.mp4
```

Keep original source files unchanged after a corpus is frozen.

## 2. Extract review frames

From `pc/`:

```powershell
python -m spectratrack.vnext_qa extract-frames `
  --video benchmarks/vnext/qa/videos/golden/night_gate_01.mp4 `
  --video-id golden/night_gate_01.mp4 `
  --output-dir benchmarks/vnext/qa/frames/night_gate_01 `
  --every-seconds 0.5
```

Or choose exact source-frame indices:

```powershell
python -m spectratrack.vnext_qa extract-frames `
  --video benchmarks/vnext/qa/videos/golden/night_gate_01.mp4 `
  --video-id golden/night_gate_01.mp4 `
  --output-dir benchmarks/vnext/qa/frames/night_gate_01 `
  --frames 120,121,122,180,240
```

The helper writes lossless PNG review frames plus `frames.json` with source SHA-256, dimensions, FPS, and source frame indices.

Extraction is not annotation.

## 3. Human annotation

Use the built-in local annotator:

```powershell
python -m spectratrack.vnext_qa annotate `
  --batch benchmarks/vnext/qa/frames/night_gate_01/frames.json `
  --output benchmarks/vnext/qa/golden.jsonl `
  --tags "night_dark,high_angle_cctv,tiny_person,compression"
```

Controls:

- left-drag: draw/update the active person's bbox;
- `i`: set the stable person ID, for example `p1`;
- `a`: set/clear comma-separated per-person attributes such as `partial_occlusion`;
- `g`: toggle `ignore=true` for the active person when a box is too ambiguous to score;
- `u`: undo the last box;
- `c`: clear boxes on the current frame;
- `n` or Enter: save + next frame;
- `b`: save + previous frame;
- `q`: save + quit.

Use the **same object ID for the same visible person across annotated frames**. Tracking metrics depend on this.

If a reviewed frame has no people, leave it with no boxes; the helper records it as `negative`.

The output remains the existing canonical `qa_benchmark` JSONL format.

### Pre-annotation

External auto-labeling may be used to speed up review, but it is only a draft. If pre-annotations are converted into the canonical JSONL file first, the annotator will display those existing boxes for manual correction.

Auto-label output is **not** golden ground truth and is never frozen merely because a model produced it.

Before freeze, every scored GOLDEN frame/box must be visually confirmed by a human. The freeze command requires an explicit human-review confirmation flag and reviewer name.

## 4. Validate annotations before freeze

```powershell
python -m spectratrack.vnext_qa validate-corpus `
  --video-root benchmarks/vnext/qa/videos `
  --train-ground-truth benchmarks/vnext/qa/train.jsonl `
  --golden-ground-truth benchmarks/vnext/qa/golden.jsonl `
  --output benchmarks/vnext/qa/corpus.validation.json
```

Validation checks:

- canonical JSONL parsing;
- duplicate `video + frame` entries;
- invalid/negative frame indices;
- invalid/non-positive bboxes;
- duplicate object IDs within a frame;
- allowed labels;
- missing source files;
- bbox dimensions against the source video;
- annotated frame index against source frame count when available;
- required GOLDEN coverage;
- exact SHA-256 TRAIN/GOLDEN leakage;
- source video SHA-256/dimensions/FPS/frame count.

Do not freeze while `valid=false`.

## 5. Freeze the first usable human-confirmed corpus revision

Example revision naming:

`cctv-golden-r1`

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

The manifest records:

- corpus revision;
- TRAIN/GOLDEN GT SHA-256;
- every source-video SHA-256;
- video dimensions/FPS/frame count;
- GOLDEN coverage counts;
- human reviewer confirmation;
- deterministic `corpus_sha256`.

Once A1-A4 start a comparison round, do not mutate that manifest or its referenced source/GT files. Create a new corpus revision instead.

## 6. Benchmark the frozen GOLDEN set

The existing evaluator remains canonical:

```powershell
python -m spectratrack.qa_benchmark run `
  --ground-truth benchmarks/vnext/qa/golden.jsonl `
  --video-root benchmarks/vnext/qa/videos `
  --model ..\models\yolo11n.onnx `
  --run-name BASELINE_CURRENT `
  --revision <exact-subject-commit-sha> `
  --output benchmarks/vnext/qa/results/baseline-current.json
```

For vNext research the additive result fields include:

- source video SHA-256/dimensions;
- actual ONNX inference-call count;
- ONNX calls/frame;
- processing wall time;
- source-video seconds processed;
- processing seconds/source second;
- GT-relative detection bbox stability;
- GT-relative tracking bbox stability;
- mean uninterrupted track length on annotated frames;
- recovery latency in source-frame index distance.

Existing schema version and existing comparator fields remain unchanged.

### Bbox stability definition

For matched objects with stable GT IDs, predicted boxes are first expressed relative to each frame's GT center/width/height.

This reduces genuine person/camera translation and scale from the stability signal.

Reported means:

- normalized center jitter;
- width log-ratio jitter;
- height log-ratio jitter;
- area log-ratio jitter;
- temporal IoU of consecutive GT-relative boxes.

Sparse annotations still limit temporal interpretation; dense annotated segments are preferred for crossings/occlusions.

## 7. Bind a result to the frozen corpus

Raw QA results do not themselves prove which frozen video bytes were used. A5 therefore stamps each result after verifying GT hash + exact GOLDEN video SHA/dimensions:

```powershell
python -m spectratrack.vnext_qa stamp-result `
  --manifest benchmarks/vnext/qa/corpus.manifest.json `
  --result benchmarks/vnext/qa/results/a1-fusion.json `
  --role A1 `
  --experiment fusion-candidate-01 `
  --output benchmarks/vnext/qa/results/a1-fusion.stamped.json
```

The stamped envelope records:

- corpus revision/hash;
- result file SHA-256;
- canonical result payload SHA-256;
- subject/source commit;
- full result/settings;
- settings hash;
- scoring/evaluation hash.

A leaderboard row must come from a stamped run.

### Specialist research output

A2/A3/A4 may legitimately produce a research-specific JSON rather than a full `qa_benchmark` result. Do **not** disguise those partial measurements as an end-to-end detector run.

For those cases, A5 accepts a normalized evidence envelope:

`spectratrack-vnext-evidence-v1`

Required envelope fields:

- `role`, `experiment`, `scope`;
- exact 40-hex `source_commit`;
- frozen `corpus_revision` + `corpus_sha256`;
- GOLDEN `ground_truth_sha256`;
- SHA-256 of the original specialist research artifact;
- exact model/provider when the experiment used inference;
- complete candidate `config`;
- common `evaluation` settings;
- normalized `quality` fields actually measured;
- normalized `compute` fields actually measured;
- provenance including **all** frozen GOLDEN input-video SHA/dimensions;
- replay SHA when the experiment is tracker-replay based.

A5 verifies the supplied source artifact bytes against the declared SHA before stamping:

```powershell
python -m spectratrack.vnext_qa stamp-evidence `
  --manifest benchmarks/vnext/qa/corpus.manifest.json `
  --evidence benchmarks/vnext/qa/results/a2-tracking.evidence.json `
  --source-artifact benchmarks/vnext/qa/results/a2-tracking.raw.json `
  --output benchmarks/vnext/qa/results/a2-tracking.stamped.json
```

Partial specialist rows remain visibly labeled by `scope`, for example `tracker-replay` or `enhancement-roi`. Missing quality fields stay null/blank; A5 never fills them from intuition.

A specialist artifact that used only a subset of GOLDEN videos is **not** eligible for the common leaderboard. This prevents a favorable subset from being compared against a full-corpus row.

## 8. Canonical detection replay validation

A1/A2 use exactly:

`spectratrack-detection-replay-v1`

Validate a dump:

```powershell
python -m spectratrack.vnext_qa validate-replay `
  --replay benchmarks/vnext/qa/replays/a1-baseline.jsonl `
  --output benchmarks/vnext/qa/replays/a1-baseline.validation.json
```

Validation requires/protects:

- full source commit SHA;
- stable video identity;
- optional source-video SHA-256;
- detector/model identifier;
- model SHA-256 when present;
- provider;
- complete config object;
- source dimensions;
- unique strictly increasing frame indices;
- per-frame dimensions;
- finite bbox/score/class/label data;
- versioned non-null appearance metadata.

Replay detections are evidence. A2 must not silently mutate an A1 replay. Derived/transformed replay data needs a new artifact with provenance.

A1 currently emits `appearance: null`, which is directly compatible. If A2 or a later producer serializes a numeric appearance vector, the replay `config` must also declare a non-empty `appearance_schema` (for example a versioned SpectraTrack descriptor identifier). Unversioned appearance vectors fail A5 validation because the canonical research plan requires non-null appearance metadata to be explicitly versioned.

## 9. Build the shared leaderboard

After A1-A4 return stamped runs:

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

The leaderboard rejects:

- different corpus revision/hash;
- different GOLDEN GT/video bytes at stamp time;
- incompatible scoring/evaluation settings;
- stamped payload/settings/evaluation hash tampering;
- duplicate role+experiment rows.

Candidate configuration differences are allowed only because the full settings object and hash are carried in the evidence. They are therefore visible, not hidden.

The table reports quality **and** compute cost. It deliberately does not sort candidates by a synthetic score and does not declare a winner.

## A1-A4 handoff contract

### A1 detection / fusion

Return:

- exact source commit;
- exact model ID + SHA-256;
- full detector/fusion config;
- provider;
- canonical detection replay(s);
- QA result on frozen GOLDEN;
- actual ONNX calls;
- stamped run.

A1 replay must pass `validate-replay` before A2 consumes it.

### A2 tracking

Use the **same validated A1 replay sequence** across tracker candidates.

Return:

- replay SHA/provenance;
- tracker config;
- ID switches;
- fragmentation;
- track recall;
- uninterrupted track length;
- recovery latency;
- bbox stability;
- stamped comparable run/result.

Do not rerun detector inference when claiming a tracker-only comparison.

### A3 enhancement

Return:

- exact raw source/model/config;
- enhancement operation/mode;
- quality delta on the same GOLDEN revision;
- FP/FN impact;
- actual extra ONNX calls;
- processing seconds/source second;
- stamped run.

Enhanced-only detections still require raw corroboration under project rules.

### A4 performance / scheduler

Return:

- exact scheduler config/budgets;
- actual ONNX calls;
- calls/frame;
- processing wall time;
- processing seconds/source second;
- quality metrics on the same GOLDEN revision;
- stamped run.

A cheaper run with hidden recall loss is not a performance win.

## Current corpus status

This repository intentionally does **not** contain a fabricated GOLDEN corpus or fake baseline result.

Until real user CCTV has been extracted, human-confirmed, validated, and frozen:

- corpus revision: **not assigned**;
- corpus SHA-256: **not available**;
- real baseline quality numbers: **not available**;
- common A1-A4 comparative run: **not ready**.

Synthetic/unit fixtures validate the tooling only.

## NightOwls SMOKE400 promotion gate

The frozen FULL5000 contract remains unchanged:

- revision: \`nightowls-public-slice5000-r1\`
- corpus SHA-256: \`1ba30ef5adad0f6bedba4319d1c3b5f246d4c576c92b26f8b98b68b4b94a0ae8\`
- frames: 5000
- tracking_supported: false

The triage-only slice is \`nightowls-public-smoke400-r1\`, seed
\`spectratrack-round2-nightowls-smoke400-v1\`.

Selection is metadata-only and candidate-independent. It ranks the locked parent identity fields
\`video, frame, source, source_frame, source_sequence\` with SHA-256. First, the lowest-ranked
frame from every parent logical video is selected; remaining slots are filled by the global
metadata hash rank. GT objects/tags and candidate detections/scores/outcomes are never ranking
inputs. After selection only, canonical \`frame\` ordinals are reindexed contiguously within each
logical video so the existing \`qa_benchmark.py\` JSONL contract remains valid. The selection proof
records the exact parent line/frame and row hash for every smoke row.

Two-stage policy:

- SMOKE400 is triage only. Invalid/incomplete/provenance-mismatched runs fail immediately.
- An obviously bad candidate may be rejected when FN or FP is at least 2x the locked reference
  and also at least 10 errors worse. Both conditions are required for that error class.
- Any non-rejected candidate is only a survivor. Ambiguous results and all promotion candidates
  must run FULL5000.
- A SMOKE400 pass is never a final full-corpus quality claim.

Variable per-frame work is approximately 8% of FULL5000: 400/5000 frames, about 92% fewer frame
evaluations and a 12.5x frame-count reduction. Fixed model-load/startup/I/O overhead is excluded.

Commands are exposed as \`build-smoke-slice\` and \`validate-smoke-slice\` under
\`python -m spectratrack.vnext_qa\`.

## Blind follow-up partition: smoke400-r2 + holdout4200-r1

After smoke400-r1 became observed by research agents, A5 froze a disjoint blind follow-up from the
same immutable `nightowls-public-slice5000-r1` universe. FULL5000 and smoke400-r1 bytes/hashes
remain unchanged.

`nightowls-public-smoke400-r2` uses seed
`spectratrack-round2-nightowls-smoke400-r2-v1`. Selection first excludes the exact r1 parent
membership from its frozen selection proof. It then picks one metadata-hash-ranked remaining frame
per logical sequence where a second frame exists, followed by global metadata-hash fill from the
remaining unseen frames. GT labels, objects, tags, candidate detections/scores and candidate
outcomes are forbidden selection inputs.

The parent has 143 logical sequences. Only the five numbered recording sequences have another
eligible frame after r1; 138 one-frame `unassigned-image-*` sequences cannot supply a second pick.
Those 138 deterministic fallbacks are listed in the r2 proof. R2 therefore contains 5
second-per-sequence picks plus 395 disjoint hash-fill picks.

`nightowls-public-holdout4200-r1` is not sampled. It is the exact set complement after frozen r1
and r2 parent membership, leaving exactly 4200 frames and no further selection freedom.

Partition invariant: r1/r2/holdout pairwise intersections are zero and their union is exactly the
original 5000 parent frames. Both new canonical JSONLs deterministically reindex only local
per-sequence frame ordinals while their proofs bind every output row to the exact parent line,
source frame and row hash. `tracking_supported=false` remains mandatory.

Stage semantics are fixed: r1 is discovery/triage and is no longer blind for r1-inspired work; r2
is blind validation for r1-inspired hypotheses; holdout4200 is the final disjoint NightOwls
heldout after decisions are locked. FULL5000 may be used afterward for characterization/aggregate
reporting, but it is not an independent holdout once any constituent subsets have been observed.

Build/validation CLI:
`python -m spectratrack.blind_smoke_gate build ...`
`python -m spectratrack.blind_smoke_gate validate ...`
