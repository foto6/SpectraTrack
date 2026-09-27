# A8 performance↔tracking Pareto gate

Research-only branch based exactly on `d03af3ae6425d3ea2e4d52e25389fecc09957394`. No production scheduler/runtime file is modified.

## Frozen producer inputs

A2 tracking source: `agent/vnext-tracking @ d73b5d5a8268586ee963df9de901d3fd6ff6fcca`.

- exact smoke manifest: `mot17-04-smoke400-v1.json`
- manifest SHA-256: `cc26aa3a37f5830912e576d9475d83b231b1842519bcd9a22baab8c06c3c4844`
- source replay file SHA-256: `b2719a2c93c353123437497ac9513033898d3d65750a32ed633d05e8fb65b2d4`
- source replay canonical SHA-256: `d27d45172a2df0a5c81ff83be2ceed03d2ed1eef922d3fde06f72b3126a89c59`
- GT SHA-256: `28dcb9d197e0a098a1efb097f1589177350192a8f5f1be3e2ab5cd18d8f205c7`
- exact windows: frames 0–199 and 400–599, resetting tracker state between windows

The producer Git tree does not contain the canonical 600-frame replay or scored GT payload bytes. It contains hashes,
the frozen manifest, the prior A2 control/result, and replay tooling. A8 therefore refuses to synthesize candidate
quality from aggregate A2 metrics.

A4 cost source: `agent/vnext-performance @ 5c880267498a18b51b2e5c5f92f6c5360fafc212`.

Two schedules were locked before any A8 quality scoring:

1. `detect_every=3, max_calls=1, global_period=8`
2. `detect_every=5, max_calls=3, global_period=5`

These are exact previously measured RX 5700 XT / DirectML bounded cost points. No parameter sweep is allowed.

## Locked quality gates

- tracking recall loss must be <= 0.25 percentage points;
- fragmentation increase must be <= 5%;
- ID switches may not increase;
- mean recovery latency may increase by at most 1 frame;
- discovery latency must be observed and no greater than the candidate periodic-global bound.

There is no scalar score.

## Current result

Both locked schedules show substantial detector-cost reduction in A4 evidence. Neither can be promoted or called a
Pareto survivor because the exact A2 replay/GT payload required to score the schedules is absent from the producer
Git tree available to this branch.

The checked-in Pareto report therefore stops both candidates with
`STOP_INSUFFICIENT_QUALITY_EVIDENCE`. It does not request a larger replay, and it never uses NightOwls as tracking
ground truth.

To complete the experiment later, supply bytes matching the pinned replay/GT hashes and score CONTROL plus both
locked schedules on the same two contiguous windows. The lock file and gates must not change after those outcomes
are observed.
