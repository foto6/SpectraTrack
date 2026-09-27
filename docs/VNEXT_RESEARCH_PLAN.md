# SpectraTrack vNext Research Plan

Status: coordination-only research baseline.

Immutable product baseline:

- `integration @ 3ebc4d50213593cac62b97399447fccf6bbc1755`
- `main @ 2012eaae2f4ffe820a66d12e40346d911616cd03`

The current integrated RC remains the immutable comparison baseline. Do not merge `integration` into `main` as part of this cycle.

## Why release is paused

Real-video validation exposed issues that are not acceptable for release:

- visible bbox jitter;
- bbox geometry that can fit the person poorly;
- ID loss / fragmentation;
- missed people;
- unacceptable cost for `people-recall + adaptive`, with observed processing around 1 second of source video per roughly 2 minutes of compute in at least one real run.

These observations are evidence of problems, not yet proof of root cause.

## Research principle

Every candidate follows:

`baseline -> isolated candidate -> same data -> metrics -> cost -> decision`

A candidate is not accepted because it is modern, has strong published COCO metrics, or looks good on one clip.

Published benchmark results are prior evidence only. SpectraTrack decisions require measurements on the same frozen SpectraTrack corpus revision.

## Branch contract

Coordination branch:

- `vnext-base`
- starts exactly from `integration @ 3ebc4d50213593cac62b97399447fccf6bbc1755`
- coordination/research docs only

Specialist branches must start simultaneously from the exact `vnext-base` HEAD reported by the architect handoff:

- `agent/vnext-detection`
- `agent/vnext-tracking`
- `agent/vnext-enhancement`
- `agent/vnext-performance`
- `agent/vnext-qa`

Rules:

- no specialist branch merges another specialist branch;
- do not change `integration`;
- do not change `main`;
- do not release;
- do not commit large video/model binaries;
- keep research tooling isolated from production behavior;
- production integration happens later only in `agent/vnext-integrator`.

Preferred experiment-output area:

`pc/benchmarks/vnext/<role>/`

Large/generated artifacts should remain ignored or external.

## Required experiment provenance

Every experiment must record at least:

- source commit SHA;
- role / experiment identifier;
- detector/model identifier;
- model SHA-256 when applicable;
- input video/frame identity;
- full relevant config;
- image width/height;
- execution provider: CPU or DirectML;
- processing wall time;
- detector policy-run count when relevant;
- actual low-level ONNX inference-call count;
- quality metrics when ground truth exists.

Do not report FPS or latency from unrelated hardware as representative of the target RX 5700 XT system.

## Canonical detection replay format

Detector and tracker research must be separable. A2 must be able to replay a frozen detection dump without running detector inference again.

Canonical format: JSONL, schema `spectratrack-detection-replay-v1`.

First record:

```json
{
  "type": "metadata",
  "schema": "spectratrack-detection-replay-v1",
  "source_commit": "<sha>",
  "video": "<stable-video-id-or-relative-path>",
  "video_sha256": "<optional-but-preferred>",
  "detector": "<backend/model identifier>",
  "model_sha256": "<sha256-or-null>",
  "provider": "DmlExecutionProvider",
  "config": {},
  "width": 1920,
  "height": 1080
}
```

Frame records:

```json
{
  "type": "frame",
  "video": "<same-video-id>",
  "frame": 123,
  "timestamp_s": 4.92,
  "width": 1920,
  "height": 1080,
  "detections": [
    {
      "bbox": [100.0, 120.0, 180.0, 310.0],
      "score": 0.82,
      "class_id": 0,
      "label": "person",
      "appearance": null
    }
  ]
}
```

Minimum required detection fields are:

- `video`
- `frame`
- `bbox`
- `score`
- `class_id`
- `label`

Optional appearance metadata must be explicitly versioned and must not be interpreted as biometric identity.

Replay consumers must not silently mutate detections. If a tracker candidate requires transformed measurements, write a derived replay artifact with provenance rather than overwriting the source dump.

## Phase 0: parallel work before real GT is ready

No agent waits for A5.

### A1 Detection / Fusion

Immediate scope:

- measure instability from full-frame + overlapping tiles;
- compare current class-aware hard NMS;
- compare conservative NMM / GreedyNMM-style merge;
- compare score-weighted coordinate fusion;
- build reproducible detector-backend harness;
- emit canonical replay dumps.

Required bbox metrics:

- center jitter;
- width jitter;
- height jitter;
- area jitter;
- temporal bbox IoU;
- FP/FN once GT exists.

Fusion must not merge two distinct nearby people.

A1 does not own tracker behavior, enhancement scheduling, or global runtime scheduling.

After A5 publishes a frozen usable corpus revision, compare suitable licensed/runtime-compatible candidates such as:

- current YOLO baseline;
- RF-DETR variants;
- RT-DETRv2 variants.

Do not use published COCO mAP as the SpectraTrack decision metric.

### A2 Tracking

Immediate scope:

- audit current `MultiObjectTracker`;
- build detection-replay benchmark;
- diagnose ID switches, fragmentation, crossings, occlusion recovery, dormant/reactivation behavior, appearance failures, and camera-motion failures.

Important existing behavior:

- current tracker already has high/low-confidence two-stage association;
- low-confidence detections may support an existing track;
- low-confidence detections do not create new tracks.

Do not implement ByteTrack-style rescue again before measuring why the current implementation fails.

Then compare the current tracker with reference ByteTrack / BoT-SORT / OC-SORT-style approaches only when dependency and license constraints are acceptable.

Temporal bbox smoothing is researched as post-association state, not as a detector hack.

A2 must consume identical frozen replay detections across tracker candidates.

### A3 Enhancement

Immediate scope:

Profile each current quality operation independently:

- low-light;
- bilateral denoise/deblock;
- mild sharpen;
- combinations.

For every operation record:

- execution cost;
- where/why it is enabled;
- whether detector output changes;
- whether a real GT detection is recovered;
- how many FPs are introduced.

Design selective enhancement research:

- enhancement only on suspect ROIs;
- no enhancement for already-good regions;
- preserve raw evidence;
- enhanced-only person detections without raw corroboration remain forbidden.

A3 does not own tile-grid geometry, final NMS/fusion, or runtime scheduling.

### A4 Performance / Scheduler

Immediate scope:

Quantify current compute for:

- 720p;
- 1080p;
- 1440p;
- 4K.

Break down:

- tile count;
- raw tile inference calls;
- enhanced inference calls;
- preprocess cost;
- ONNX inference cost;
- postprocess cost;
- seconds processing per source second.

For 1080p with `tile=640`, `overlap=0.20`, explicitly verify the current adaptive path: full-frame pass + raw tile probes + any enhanced tile passes. Distinguish theoretical maximum calls from observed calls.

Do not micro-optimize until the bottleneck is measured.

Research an experimental compute-budget scheduler:

- cheap global pass;
- selective ROIs;
- track-guided ROI rescans;
- periodic global rescan;
- immediate rescan on scene change;
- configurable detector cadence;
- hard inference-call budget;
- adaptive-work budget.

Research batching only if DirectML/ONNX benchmark evidence shows a real win.

A4 does not change enhancement algorithms, detector-fusion semantics, or tracker-association semantics.

### A5 CCTV QA / Golden Set

Immediate scope:

Use existing `qa_benchmark.py` as the canonical basis and create a small representative golden-corpus workflow covering:

- tiny people;
- distant people;
- night/dark;
- blur;
- compression;
- partial/heavy occlusion;
- high camera angle;
- crossings;
- moving camera;
- negative frames.

Auto-labeling may be used only for pre-annotation. It is not ground truth. Final GT must be human-confirmed.

Required comparison metrics:

- person recall;
- precision;
- FP;
- FN;
- recall by bbox height;
- ID switches;
- fragmentations;
- bbox jitter;
- uninterrupted track length;
- recovery latency;
- actual inference calls/frame;
- processing seconds/source second.

A5 owns experiment comparison/leaderboard infrastructure, not feature candidates.

## Frozen-corpus synchronization

When A5 publishes the first usable corpus snapshot:

1. assign a corpus revision identifier and immutable manifest;
2. freeze the annotated snapshot for the comparison round;
3. all agents rerun relevant baseline/candidates on that exact revision;
4. A5 assembles one comparison table;
5. architect review decides what evidence justifies production integration.

No candidate enters production directly from a specialist branch.

## Decision gates

A candidate can advance only if:

- it has reproducible provenance;
- it is compared against the exact immutable baseline or a clearly defined isolated parent;
- the same input/GT revision is used;
- quality and compute cost are both reported;
- known regressions are explicit;
- licenses/runtime compatibility are verified;
- improvements are not based only on synthetic data;
- no semantic ownership boundary is violated.

The architect may reject a candidate that improves one metric if cost, regressions, complexity, or maintainability are unacceptable.

## Prohibited shortcuts

Do not:

- integrate all candidates at once;
- resolve semantic conflicts with blind `ours/theirs`;
- change detector + tracker + enhancement simultaneously and call the result proof;
- use synthetic metrics as CCTV-quality evidence;
- use other-hardware FPS as RX 5700 XT evidence;
- lower tracker strong threshold merely to create more IDs;
- treat `global_object_id` as biometric identity;
- promise quality gains without annotated data.
