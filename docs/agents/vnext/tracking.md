# A2 — vNext Tracking Research

Branch:

`agent/vnext-tracking`

Start from the exact `vnext-base` SHA in the architect handoff.

## Immediate objective

Audit the current `MultiObjectTracker` and explain observed ID loss/fragmentation before proposing a replacement.

Important: the current tracker already has high/low-confidence two-stage association. Low-confidence detections may maintain an existing track but may not create a new one.

Do not reimplement ByteTrack rescue as if it were missing.

## Required diagnostics

Using frozen detection replay, measure and classify:

- ID switches;
- fragmentation;
- crossing failures;
- short-occlusion recovery failures;
- long-occlusion/dormant-reactivation failures;
- appearance-cue failures;
- camera-motion/CMC failures;
- uninterrupted track length;
- recovery latency.

Build a replay harness that consumes `spectratrack-detection-replay-v1` and performs no detector inference.

Every tracker candidate must receive identical detections.

## Candidate research

Only after current failure modes are measured, compare current behavior with reference approaches where dependency/license constraints are reasonable:

- ByteTrack;
- BoT-SORT;
- OC-SORT-style approaches.

Do not infer superiority from published MOT results alone.

## Bbox smoothing

Research temporal bbox smoothing only as post-association track state.

Keep separate metrics for:

- detector bbox quality;
- tracker identity quality;
- displayed/smoothed bbox quality.

Do not hide detector errors by rewriting replay detections.

## Ownership boundaries

A2 owns:

- tracking replay harness;
- tracker diagnostics;
- tracker-candidate experiments;
- post-association temporal smoothing research.

A2 does not own:

- detector backend/fusion;
- tile grid;
- enhancement;
- runtime scheduler;
- GT annotation policy.

Do not lower the tracker strong threshold merely to create more IDs.

`global_object_id` remains a non-biometric appearance-grouping concept, not identity.
