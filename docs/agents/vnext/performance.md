# A4 — vNext Performance / Scheduler Research

Branch:

`agent/vnext-performance`

Start from the exact `vnext-base` SHA in the architect handoff.

## Immediate objective

Build a quantitative compute model of the current pipeline before optimizing it.

Measure at:

- 720p;
- 1080p;
- 1440p;
- 4K.

For each workload record:

- tile count;
- detector policy runs;
- raw tile inference calls;
- enhanced inference calls;
- preprocess time;
- ONNX inference time;
- postprocess time;
- tracker/other pipeline time where relevant;
- total wall time;
- processing seconds per source second;
- provider;
- image dimensions.

For 1080p with `tile=640`, `overlap=0.20`, explicitly verify how many calls current adaptive actually performs: full-frame, raw probes, and enhanced passes. Report theoretical upper bound separately from observed calls.

Do not micro-optimize until measurements identify the bottleneck.

## Scheduler research

Model and benchmark an experimental budget scheduler with:

- cheap global pass;
- selective ROIs;
- track-guided ROI rescans;
- periodic global rescan;
- immediate scene-change rescan;
- detector cadence;
- hard inference-call budget;
- adaptive-work budget.

The scheduler simulator must make its quality/cost tradeoffs explicit.

## Batching

Investigate batching only if ONNX Runtime + DirectML benchmarks on the target-relevant stack show a real end-to-end benefit.

Do not assume CUDA batching behavior applies to DirectML.

## Ownership boundaries

A4 owns:

- instrumentation/reports;
- compute model;
- scheduler simulation;
- performance experiments.

A4 does not own:

- enhancement algorithm semantics;
- detector fusion semantics;
- tracker association semantics;
- QA GT definitions.

Do not present hosted-runner or other-GPU FPS as RX 5700 XT evidence.
