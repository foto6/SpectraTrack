# A3 — vNext Enhancement Research

Branch:

`agent/vnext-enhancement`

Start from the exact `vnext-base` SHA in the architect handoff.

## Immediate objective

Quantify whether current adaptive enhancement provides enough real detection benefit to justify its cost.

Profile each operation separately:

- low-light;
- bilateral denoise/deblock;
- mild sharpen;
- supported combinations.

For each operation record:

- operation cost;
- tiles/ROIs on which it activates;
- activation reason/quality signal;
- detector-output delta;
- GT detections recovered;
- FP introduced;
- extra inference calls caused downstream;
- wall-time impact.

## Selective enhancement research

Design experiments for:

- enhancement only on suspect ROIs;
- no enhancement on already-good regions;
- raw-evidence preservation;
- enhanced-only person detections without raw corroboration remaining forbidden.

Do not apply enhancement everywhere simply because adaptive mode is enabled.

## Evidence requirements

Before claiming quality improvement:

- use the same frozen corpus revision;
- report recall/precision/FP/FN deltas;
- report compute cost and actual inference calls;
- separate per-operation effects from combinations.

## Ownership boundaries

A3 owns:

- enhancement transforms;
- quality/operation profiling;
- selective-enhancement candidate logic at research level.

A3 does not own:

- tile-grid geometry;
- final NMS/fusion;
- tracker association;
- runtime scheduling/global budgets.

Do not change production enhancement behavior before evidence and architect review.
