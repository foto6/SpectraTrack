# QA / regression agent state

## Ownership

Branch: `agent/qa`

Primary scope:

- ground-truth format and validation;
- detector/tracker quality evaluation;
- TP/FP/FN, precision, recall;
- size/tag/attribute breakdowns;
- ID switches and fragmentation;
- baseline-vs-candidate comparison;
- new-false-negative regression gates;
- benchmark documentation and tests.

Normally outside scope:

- implementing detector/tracker/enhancement/performance features for other agents.

## Current objective

Provide an independent, reproducible gate for deciding whether agent branches actually improve or regress SpectraTrack.

## Important invariants

- Baseline and candidate comparisons must use identical ground truth and compatible evaluation settings.
- A candidate must not silently introduce new false negatives.
- Synthetic evaluator fixtures validate the evaluator, not real detector quality.
- Real quality claims require a representative annotated real dataset.
- VRAM must not be fabricated or inferred without a trustworthy measurement source.

## Current branch state

Original shared base: `integration @ 573087cdf60c1a00bbc93b8764733b2ade3cbdd8`

Observed HEAD when this file was created: `5ff444758ad48ec3d5b52bf1226f181da75c5945`

Observed work includes:

- annotated JSONL ground-truth evaluator;
- TP/FP/FN + precision/recall;
- per-tag/per-size/per-attribute reporting;
- tracking recall, ID switches and fragmentation;
- saved baseline/candidate comparison;
- explicit NEW FALSE NEGATIVE gate;
- benchmark docs/tests and result/video gitignore handling.

## Current limitation

No representative validated real CCTV dataset has yet been established. Until that exists, the harness is infrastructure, not proof that one detector branch is objectively better.

## Integration role

QA should provide commands and evidence for the integrator. It should not merge or rewrite other agent branches.

## Handoff checklist

Update before final handoff:

- HEAD SHA:
- commits owned:
- evaluator tests run:
- exact commands for integration comparison:
- real dataset status:
- known metric limitations:
- ready for integration: yes/no
