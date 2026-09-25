# Detection agent state

## Ownership

Branch: `agent/detection`

Primary scope:

- detector inference and decode;
- high-recall person detection;
- tiled/sliced inference;
- class-specific detector thresholds;
- detector-level regression/benchmark tooling;
- detector call paths in live and batch modes only as required.

Normally outside scope:

- tracker association/Re-ID;
- broad preprocessing architecture;
- performance pipeline refactors;
- QA framework ownership.

## Current objective

Reduce false negatives for small/distant/poor-quality people without silently degrading standard detector behavior.

## Important invariants

- Standard detector mode must remain compatible.
- Tile boxes must map correctly back to full-frame coordinates.
- Overlapping tile/full-frame detections must be merged without class-crossing suppression.
- Batch and live detector behavior must stay consistent.
- No recall/precision improvement may be claimed without a real annotated dataset.

## Current branch state

Original shared base: `integration @ 573087cdf60c1a00bbc93b8764733b2ade3cbdd8`

Observed HEAD when this file was created: `9a21c495baa258c55d74ddc6a4cdde6c88e2867a`

Observed work includes:

- tiled high-recall person detector path;
- `people-recall` mode and preset;
- lower class-specific threshold for `person`;
- merge/NMS handling for combined detections;
- detector-level person regression harness;
- detector benchmark metadata fixes/documentation.

## Integration hotspots

Likely overlap with enhancement/performance in:

- `pc/spectratrack/detector.py`
- `pc/spectratrack/app.py`
- `pc/spectratrack/runtime_config.py`

Do not resolve those by discarding another branch. Integrator must reconcile behavior intentionally.

## Handoff checklist

Update before final handoff:

- HEAD SHA:
- commits owned:
- tests run:
- real benchmark data:
- unverified items:
- conflicts/overlaps:
- ready for integration: yes/no
