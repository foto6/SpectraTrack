# A1 — vNext Detection / Fusion Research

Branch:

`agent/vnext-detection`

Start from the exact `vnext-base` SHA in the architect handoff.

## Immediate objective

Explain and measure bbox instability caused by combining full-frame and overlapping-tile detections.

Do not begin with a detector replacement.

## First experiments

Compare, in isolation:

1. current class-aware hard NMS;
2. conservative NMM / GreedyNMM-style merge;
3. score-weighted coordinate fusion.

Use identical source frames and detector outputs where possible.

Required measurements:

- center jitter;
- width jitter;
- height jitter;
- area jitter;
- temporal bbox IoU;
- FP/FN when GT exists;
- actual inference-call count;
- wall time.

Explicitly test dense/crossing scenes so fusion does not merge two nearby people into one box.

## Harness responsibilities

Build or extend a reproducible detector-backend experiment harness.

Emit canonical `spectratrack-detection-replay-v1` dumps so A2 can replay identical detections without detector inference.

Record:

- source commit;
- detector backend/model;
- model SHA-256;
- provider;
- image dimensions;
- input config;
- frame/video identity;
- inference-call count;
- wall time.

## Second phase after A5 corpus freeze

Evaluate only candidates with verified license and realistic export/runtime compatibility, including appropriate variants of:

- current YOLO baseline;
- RF-DETR;
- RT-DETRv2.

Published COCO mAP is context only, not SpectraTrack evidence.

## Ownership boundaries

A1 owns:

- detector backend harness;
- full-frame/tile detection research;
- detection fusion research;
- detection replay export.

A1 does not own:

- tracker association/re-ID;
- enhancement algorithm design;
- enhancement scheduling;
- global runtime scheduler;
- QA ground-truth policy.

Do not modify production detector semantics before evidence and architect review.
