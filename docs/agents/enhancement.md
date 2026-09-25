# Enhancement / tiny-object agent state

## Ownership

Branch: `agent/enhancement`

Primary scope:

- frame quality estimation;
- adaptive non-generative preprocessing;
- low-light/denoise/deblock/mild sharpen routing;
- tiny-object/tiled detection support where enhancement is required;
- raw-frame corroboration of enhancement-originated candidates;
- enhancement tests.

Normally outside scope:

- general tracker architecture;
- Re-ID;
- broad performance refactors;
- QA framework ownership.

## Current objective

Help the detector recover small/poor-quality people while preventing enhancement artifacts from becoming unsupported accepted detections.

## Important invariants

- Enhancement must not be treated as new ground-truth evidence.
- An enhanced-only person candidate must not be accepted without raw-frame corroboration.
- Normal/well-lit frames should not be needlessly processed.
- Processing must preserve image shape/type unless an explicitly tested detector path requires otherwise.

## Current branch state

Original implementation work began from `2012eaae2f4ffe820a66d12e40346d911616cd03`, not the canonical parallel-agent base.

Canonical shared base is `integration @ 573087cdf60c1a00bbc93b8764733b2ade3cbdd8`.

Observed HEAD when this file was created: `9a9d6a65fc0487753b23c5d3d35f661f60c0d51b`

Observed work includes:

- frame quality assessment: blur/darkness/compression/low-resolution/noise;
- adaptive low-light enhancement, denoise/deblock and mild sharpening;
- tiled person detection helpers;
- weak raw probe + enhanced candidate corroboration;
- people-recall runtime configuration;
- tracker-side weak-person temporal confirmation/floor changes.

## Required cleanup before integration

This branch currently diverges from the canonical base and overlaps the tracking agent.

Before handoff:

- preserve enhancement work;
- reconcile onto the canonical base without force-pushing other branches;
- justify every remaining change in `tracker.py` / `test_tracker.py`;
- prefer handing tracker semantics to `agent/tracking` rather than maintaining a second tracker policy.

## Integration hotspots

- `pc/spectratrack/detector.py`
- `pc/spectratrack/app.py`
- `pc/spectratrack/runtime_config.py`
- `pc/spectratrack/tracker.py`
- `pc/tests/test_tracker.py`

## Handoff checklist

Update before final handoff:

- corrected base/history:
- HEAD SHA:
- commits owned:
- tests run:
- raw-vs-enhanced validation:
- tracker changes still required and why:
- conflicts/overlaps:
- ready for integration: yes/no
