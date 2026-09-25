# Tracking / Re-ID agent state

## Ownership

Branch: `agent/tracking`

Primary scope:

- track association and lifecycle;
- temporary-loss recovery/reactivation;
- CMC interaction with tracking;
- track continuity and ID-switch reduction;
- cross-video multi-signal object linking;
- global object IDs and their semantics;
- tracking/Re-ID tests.

Normally outside scope:

- detector threshold/model strategy;
- frame enhancement;
- performance pipeline refactors;
- QA framework ownership.

## Current objective

Preserve object identity through short/long detector dropouts and provide conservative cross-video object grouping without presenting visual similarity as biometric identity.

## Important invariants

- Same-class association remains mandatory.
- Person cross-video links are appearance candidates, not identity claims.
- Re-ID scores are not probabilities unless calibrated.
- CMC must not be double-applied.
- Dormant/reactivated tracks must not create easy ID hijacking between similar nearby objects.

## Current branch state

Original shared base: `integration @ 573087cdf60c1a00bbc93b8764733b2ade3cbdd8`

Observed HEAD when this file was created: `ccf2b14e1606049a313d3471fc9e762723b1e095`

Observed work includes:

- dormant confirmed tracks and later reactivation;
- appearance/geometry-based reactivation gates;
- dormant-geometry updates under CMC;
- multi-signal cross-video scoring using appearance/color/shape/size/direction/time;
- global object IDs for grouped tracklets;
- additional tracker/Re-ID regression tests.

## Integration hotspots

Likely overlap with enhancement in:

- `pc/spectratrack/tracker.py`
- `pc/tests/test_tracker.py`

Enhancement-originated weak-person temporal confirmation must be reconciled with tracker lifecycle rules instead of overwriting them.

## Handoff checklist

Update before final handoff:

- HEAD SHA:
- commits owned:
- tests run:
- ID-switch/fragmentation evidence:
- unverified items:
- conflicts/overlaps:
- ready for integration: yes/no
