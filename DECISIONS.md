# DECISIONS.md

Architectural decisions are recorded here so parallel agents do not repeatedly reopen settled questions without evidence.

## 2026-09-25 — Repository is also shared agent memory

Decision: keep code and agent coordination documentation in the same SpectraTrack repository.

Reason:

- architecture/task decisions version with the code they describe;
- new agents can bootstrap from repository state without chat history;
- Git commits/branches/PRs provide a durable coordination trail.

After setup is merged, prefer:

```text
main <- integration <- agent/*
```

Do not commit agent feature work directly to `main`.

## 2026-09-25 — Local-only runtime

Decision: PC and Android runtime remain local-only by default with no telemetry/cloud inference/model downloader.

Reason: privacy, reproducibility, deterministic model provenance, and simpler runtime trust boundaries.

## 2026-09-25 — Model weights are separate from application releases

Decision: do not silently bundle or download arbitrary model weights.

Reason: model licensing/provenance varies. The repository has SHA-256 and manifest support so the exact model can be identified.

## 2026-09-25 — Fixed-size YOLO ONNX is the baseline detector contract

Decision: support verified raw YOLO `xywh + class scores` and common end-to-end `xyxy + score + class` layouts; reject unknown layouts rather than guessing.

Reason: silent decoder guesses can produce plausible but incorrect boxes/classes.

## 2026-09-25 — DirectML preference on Windows

Decision: prefer `DmlExecutionProvider` when available, with CPU fallback.

Reason: the project targets Windows PCs including AMD GPUs without requiring CUDA/NVIDIA.

## 2026-09-25 — Analysis frame and display frame are separate

Decision: operator display modes do not automatically alter detector input. Detector enhancement is an explicit separate switch.

Reason: visual filters useful to humans can reduce detector accuracy; evaluation must know which pixels the model actually received.

## 2026-09-25 — Non-generative live enhancement first

Decision: live enhancement uses CLAHE/denoise/sharpen-style processing. Generative SR remains optional for selected snapshots and is marked AI-enhanced.

Reason: generative reconstruction can hallucinate detail and consumes detection/tracking compute budget.

## 2026-09-25 — Two-stage tracker semantics

Decision:

- high-confidence detections may create tracks;
- lower-confidence detections may maintain/recover existing tracks;
- skipped detector frames use `predict_only()` and do not count as misses.

Reason: this reduces ID churn and lets lower-confidence evidence help continuity without spawning many false identities.

## 2026-09-25 — Tracker association is currently dependency-light and greedy

Decision: current `MultiObjectTracker` uses class gating, predicted geometry, confidence and optional appearance cue with greedy candidate selection.

Reason: simple, testable baseline. Global assignment/Kalman/ByteTrack/BoT-SORT upgrades require measured regression benefit before replacement.

## 2026-09-25 — CMC is affine image-space motion

Decision: use Shi-Tomasi + LK optical flow + RANSAC affine-partial transform with hard plausibility gates.

Reason: camera pan/rotation/scale changes otherwise contaminate target velocity and association.

CMC is not physical camera pose.

## 2026-09-25 — Stabilization and CMC must not double-compensate

Decision: when stabilization is active, tracker CMC compensation is disabled for that stabilized path.

Reason: applying both to the same motion would shift predictions twice.

## 2026-09-25 — Appearance cue is non-biometric

Decision: live tracker appearance uses color histograms. Cross-video person links are `same_appearance_candidate`, not identity.

Reason: useful continuity without face recognition/biometric identity claims.

## 2026-09-25 — Selected-target LockRefiner re-anchors on detector observations

Decision: on fresh selected-target detections, initialize/re-anchor optical flow. Use flow between detector observations or during short misses.

Reason: prevents repeatedly integrating stale optical flow and prevents double-applying motion to an already-current detector box.

Current implementation is display-side; it is not yet a tracker measurement source.

## 2026-09-25 — Source FPS is preserved for processed file output

Decision: when source FPS is available, use it for the output writer instead of measured processing throughput.

Reason: processing speed must not alter playback duration.

## 2026-09-25 — Camera backends apply only to camera sources

Decision: DirectShow/MSMF selection is used for integer camera sources, not ordinary video files.

Reason: forcing camera backends onto MP4/MKV files can make valid files fail to open.

## 2026-09-25 — Cross-video grouping is conservative and reviewable

Decision:

- compare only compatible classes;
- keep a small diverse appearance gallery per tracklet;
- use complete-link strong grouping instead of transitive chain merging;
- expose ambiguous candidates in HTML;
- allow explicit SAME / DIFFERENT / UNSURE review.

Reason: cross-video visual matching is uncertain; the system should preserve ambiguity instead of inventing certainty.

## 2026-09-25 — Person cross-video semantics are appearance-only

Decision: person cross-video relations never claim biometric identity or a named individual.

Reason: current descriptor is visual appearance only and the project does not implement face recognition.

## 2026-09-25 — No fabricated sensor data

Decision:

- pseudo-thermal is labeled false color;
- monocular RGB does not produce fake metric target range;
- image velocity remains pixels/frame unless real calibration/depth supports physical units;
- target GPS is not inferred from pixels.

Reason: distinguish measured/calibrated quantities from visualization/model estimates.

## 2026-09-25 — Small-person recall work starts with a benchmark, not threshold guessing

Decision: the next detection milestone must first establish a representative annotated validation set for poor high-angle/night/compressed footage.

Preferred implementation path after baseline measurement:

1. offline quality-max mode;
2. dedicated high-resolution person pass;
3. tiled/sliced inference;
4. class-specific person thresholds;
5. temporal confirmation/recovery of weak evidence;
6. benchmark model/preprocessing variants;
7. optimize for live performance only after recall is demonstrated.

Reason: lowering one global confidence threshold cannot recover objects the model never resolves, and it can flood the tracker with false positives.


## 2026-09-25 — QA regression comparisons require identical evaluation inputs

Decision:

- annotated QA results record the exact ground-truth SHA-256, model SHA-256, provider list, detector settings, and evaluation IoU;
- comparisons are rejected when ground-truth hashes, target labels, or matching IoU differ;
- the default regression gate treats any object detected by baseline but missed by a candidate as a `NEW FALSE NEGATIVE`;
- recall/precision drops and increases in ID switches/fragmentation also fail by default unless an explicit tolerance is supplied;
- VRAM is recorded only when a real measurement source is provided; the benchmark must not estimate or fabricate it.

Reason: branch comparisons are useful only when they measure the same annotated evidence under known settings. A strict new-miss gate catches visually convincing demos that quietly lose previously detected people.
## 2026-09-25 — Confirmed tracks may enter a bounded dormant reactivation window

Decision:

- keep the existing two-stage live tracker and CMC contract;
- after a confirmed local track exceeds `max_missed`, allow a bounded dormant window instead of immediately making its ID unrecoverable;
- dormant reactivation requires the same object class plus agreement from several cues: appearance, box shape/size, spatial plausibility, and motion direction when available;
- do not reactivate a dormant track when appearance information is unavailable;
- detector-skipped `predict_only()` frames do not age the dormant window.

Reason: the existing tracker already follows ByteTrack-like high/low confidence semantics and CMC. A conservative dormant pool targets the measured long-occlusion fragmentation failure without replacing the tracker wholesale or adding a heavy dependency.

A local `track_id` remains a video-local trajectory identifier. Cross-video/global object grouping is a separate layer.

## 2026-09-25 — Cross-video Re-ID uses multi-signal scores and explicit global object IDs

Decision:

- evolve the existing cross-video graph rather than create a second Re-ID subsystem;
- keep legacy `entity_id` and `similarity` fields for compatibility;
- add an explicit `global_object_id` that is independent from each member's local `track_id`;
- combine appearance/gallery evidence with color, shape, relative size, and—only when comparable—temporal/direction continuity;
- allow non-overlapping same-video track fragments to become candidates, while overlapping same-video tracks remain incompatible unless manually reviewed as SAME;
- expose score components and state explicitly that the combined score is an engineering similarity score, not a calibrated probability;
- a future learned embedding may be added as one signal only after provenance and benchmark evidence; it must not become the sole match criterion.

Reason: the current graph already provides conservative complete-link grouping and manual SAME/DIFFERENT/UNSURE review. Extending it minimizes schema and architecture duplication while preserving ambiguity.

## 2026-09-26 — vNext QA measurements remain backward-compatible with QA result schema v1

Decision:

- keep `qa_benchmark` result `schema_version = 1` and preserve all existing fields/comparator behavior;
- add vNext measurement fields only additively:
  - exact input-video SHA-256/dimensions;
  - ONNX calls/frame and processing seconds/source second;
  - GT-relative bbox-stability metrics;
  - uninterrupted-track length and recovery latency;
- old consumers may ignore these new fields without changing prior semantics;
- keep frozen-corpus manifests, stamped experiment runs, replay validation, and leaderboard output in separate research schemas rather than overloading production/session formats;
- auto-label output is never accepted as golden ground truth without explicit human confirmation.

Reason: A1-A4 need richer common evidence, but the existing QA result contract is already used by regression tooling. Additive measurement fields let vNext experiments share quality/compute evidence without invalidating existing QA consumers or creating a second incompatible evaluator.

## 2026-09-26 — public pedestrian datasets supplement, not replace, private CCTV validation

Decision:

- add isolated importers for MOT17 train ground truth and CrowdHuman validation annotations;
- importers convert into the existing `qa_benchmark.py` JSONL format; there is no second evaluator or GT schema;
- public dataset files remain external/local and are never committed by A5;
- frozen public corpus manifests include dataset/version/split, source-file SHA-256 values, conversion settings, selected sequences, and importer source commit;
- MOT17 keeps stable pedestrian identities; target-like distractor classes (person-on-vehicle, static person, distractor, reflection) become canonical `ignore=true`; unrelated classes do not enter person scoring;
- MOT17 original 1-based frame number and source sequence are preserved as source provenance while canonical evaluator frame indices remain zero-based;
- CrowdHuman uses the official validation split only for detection evaluation;
- CrowdHuman full-body `fbox` is the default A5 evaluation policy because CrowdHuman explicitly defines a full-body detection task; visible-body `vbox` is an explicit opt-in and must be frozen as a different corpus revision;
- CrowdHuman objects intentionally have no stable canonical IDs, so tracking metrics remain disabled for its independent images;
- imported full-body boxes may extend beyond image bounds when the source annotation policy does; validator still requires them to intersect the image;
- public datasets are a reproducible benchmark layer, not a substitute for the user-owned human-confirmed private CCTV holdout.

Licensing/terms constraints recorded by the importer:

- MOTChallenge datasets: CC BY-NC-SA 3.0;
- CrowdHuman images: non-commercial research/education only and may not be redistributed.

Reason: public annotations reduce manual labeling cost and improve repeatability, but domain shift means a small private CCTV holdout remains necessary for product-specific evidence.

