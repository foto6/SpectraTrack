# A2 vNext tracking research

Research base:

`vnext-base @ d03af3ae6425d3ea2e4d52e25389fecc09957394`

Immutable product baseline:

`integration @ 3ebc4d50213593cac62b97399447fccf6bbc1755`

This directory documents the A2 research harness. It does not replace the production tracker.

## Canonical replay

A2 consumes JSONL schema:

`spectratrack-detection-replay-v1`

The parser is `spectratrack.detection_replay`.

A1's current vNext replay writer was checked against this reader. The canonical metadata and frame fields are accepted directly. A2 additionally understands optional research-only frame fields:

- `detector_ran=false` for an intentionally skipped detector frame;
- `camera_motion=[dx,dy]`;
- `camera_transform=[a,b,tx,c,d,ty]`.

These fields are optional. Their absence keeps the canonical A1 replay behavior.

Every tracker candidate receives fresh `Detection` objects reconstructed from the same immutable replay records. The replay canonical SHA-256 is recorded in output. Tracker research performs zero detector policy runs and zero ONNX inference calls.

## Commands

From `pc/`:

```powershell
python -m spectratrack.tracking_research --synthetic --performance-repeats 100 --output benchmarks/vnext/tracking/synthetic-report.json
```

Replay an A1 dump through the current tracker with association diagnostics:

```powershell
python -m spectratrack.tracking_research --replay path\to\detections.jsonl --candidate current --audit-current --output benchmarks/vnext/tracking/current.json
```

Run the same replay through one research candidate:

```powershell
python -m spectratrack.tracking_research --replay path\to\detections.jsonl --candidate current-global-assignment --output benchmarks/vnext/tracking/current-global.json
python -m spectratrack.tracking_research --replay path\to\detections.jsonl --candidate byte-global-reference-style --output benchmarks/vnext/tracking/byte-style.json
python -m spectratrack.tracking_research --replay path\to\detections.jsonl --candidate botsort-reference-style --output benchmarks/vnext/tracking/botsort-style.json
python -m spectratrack.tracking_research --replay path\to\detections.jsonl --candidate ocsort-style --output benchmarks/vnext/tracking/ocsort-style.json
```

## Synthetic failure catalog

The deterministic pre-A5 suite covers:

- crossings (including an exact-overlap ambiguity audit);
- a nearby same-class asymmetric-bbox counterexample where greedy local choice is worse than the global one-to-one assignment;
- partial occlusion;
- full occlusion;
- short detector dropout;
- long detector dropout;
- dormant reactivation;
- appearance mismatch;
- camera pan;
- affine camera transform;
- intentionally detector-skipped frames;
- weak detection sequences;
- false weak detections;
- nearby same-class people;
- changing bbox scale;
- sudden direction change.

Synthetic probes are mechanism tests, not CCTV-quality evidence.

## Current-tracker audit

`AuditedCurrentTracker` subclasses the current production `MultiObjectTracker` only inside the research harness. It records:

- high-stage candidate geometry and score;
- low/unmatched-stage candidate geometry and score;
- selected greedy matches;
- dormant reactivation candidates and selected matches.

Production `tracker.py` is unchanged.

Important: the current tracker already has ByteTrack-like high/low-confidence rescue. The research suite explicitly verifies that weak detections can maintain an existing track and cannot create a new ID.

## Candidate meaning

The isolated `current-global-assignment` candidate preserves the current SpectraTrack score/gates, high/low stages, CMC and dormant lifecycle; only greedy one-to-one selection is replaced by maximum-score global assignment. It is the cleanest test of the assignment hypothesis.

The three named external-style candidates are dependency-free clean-room mechanism probes. They are **not bit-for-bit copies of official repositories**:

- `byte-global-reference-style`: high/low association + strong-only creation + global assignment + constant-velocity geometry;
- `botsort-reference-style`: global assignment + appearance + replay CMC;
- `ocsort-style`: global assignment + observation-direction motion cue.

This split is intentional. It isolates mechanisms before adding external stacks to Windows production.

External reference feasibility observed during this research:

- official ByteTrack is MIT but its repository setup brings YOLOX-oriented dependencies and Cython/pycocotools tooling;
- official BoT-SORT is MIT but its setup is substantially heavier, including PyTorch/FastReID and optional FAISS-style appearance dependencies;
- official OC-SORT is MIT and motion-centric, but the official repository is also coupled to YOLOX/filterpy tooling;
- Deep OC-SORT adds appearance/Re-ID complexity and is not justified before same-replay evidence on the frozen SpectraTrack corpus.

No external tracker dependency is added by A2.

## Metrics

Identity/tracking:

- ID switches;
- fragmentations;
- tracking recall;
- false track creations;
- mean uninterrupted track length;
- recovery latency;
- recovered same-ID count;
- wrong recovery count;
- CPU time;
- wall time.

Post-association bbox stability:

- center jitter;
- width jitter;
- height jitter;
- area jitter;
- motion-compensated temporal IoU;
- response lag after a real direction change;
- mean IoU to truth.

The smoothing probe compares raw boxes, bounded EMA, and a motion-state alpha/beta smoother. Smoothing never changes association or replay detections.

## A5 synchronization

At the time this harness was created, A5 had not published a frozen real CCTV corpus revision in its role state.

When A5 publishes an immutable corpus/hash, rerun all finalists on exactly that revision before recommending production integration.
