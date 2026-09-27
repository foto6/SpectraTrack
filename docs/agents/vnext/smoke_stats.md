# A6 — Smoke Statistics and Promotion Gate

Branch: `agent/vnext-smoke-stats`

## Purpose

The 400-frame smoke path is a triage instrument, not a smaller authoritative benchmark. Its job is to reject clearly bad/too-small candidates cheaply, promote clearly material candidates to the frozen full corpus, and label the rest `AMBIGUOUS`.

The implementation is `pc/spectratrack/smoke_stats.py`. The machine-readable policy is `pc/benchmarks/vnext/smoke_stats/promotion_policy.v1.json`.

## Ordering constraint

The smoke frame selection must be frozen before any candidate scoring and before this analysis consumes full-corpus composition for representativeness checks. A6 never chooses NightOwls frames and never changes A5 corpus/selection bytes.

At implementation start, A5's local smoke400 worktree was still at `95d0b389efcd46030634ec6005c0bc18098f4aa5` with no smoke400 commit. Therefore the committed example fixture is deliberately synthetic. NightOwls-specific observed coverage must only be added after A5 publishes the exact selection/hash.

## What 400 vs 5000 means

Frame cost is deterministic: 400 / 5000 = 8%, so one smoke run processes 12.5x fewer frames than one full run.

For a rare condition that occupies exactly R distinct frames in the frozen 5000-frame corpus, the module reports the without-replacement probability that a 400-frame outcome-independent smoke sample contains at least one such frame:

`1 - C(5000 - R, 400) / C(5000, 400)`.

This is a coverage calculation, not a claim that objects within frames are independent.

## Coverage statistics

After selection is frozen, A6 accepts both full-corpus and observed smoke composition:

- total frames;
- positive frames;
- scored target objects.

Positive-frame coverage gets its exact finite-population expected count and hypergeometric standard deviation. The observed positive-frame fraction gets a Wilson interval.

Object coverage gets an expected count from the full-corpus object/frame rate plus a Poisson-style rate interval for the observed smoke count. The result explicitly labels that interval descriptive because multiple objects can cluster in the same frame.

## Candidate-effect intervals

Three candidate-independent metric forms are supported:

- `paired_binary_delta`: preferred for the same GT opportunities. Record candidate improvements and regressions relative to control. Conservative benefit bounds are built from Wilson intervals for the two transition rates.
- `binomial_delta`: fallback when only baseline/candidate successes over the same denominator are available. It is wider because pairing information is discarded.
- `poisson_rate_delta`: count-rate differences such as FP/frame. Count uncertainty uses a Wilson-Hilferty/Byar-style Poisson approximation.

All intervals are oriented so positive means "better candidate" after applying metric direction. No p-values are used for promotion.

## Promotion logic

Thresholds are declared per metric before candidate scoring:

- `minimum_material_improvement`: smallest benefit worth paying for on the full benchmark;
- `maximum_tolerable_harm`: largest allowed adverse change.

Primary metric states:

- `PROMISING`: lower bound reaches the minimum material improvement;
- `TOO_SMALL`: the interval excludes unacceptable harm but its upper bound cannot reach the material-improvement threshold;
- `HARMFUL`: even the interval upper bound is beyond the allowed harm boundary;
- `AMBIGUOUS` / `INSUFFICIENT_EVIDENCE`: unresolved.

Guardrails use `SAFE`, `HARMFUL`, `AMBIGUOUS`, or `INSUFFICIENT_EVIDENCE`.

Overall categories:

- `CLEAR_REJECT`: any metric is `HARMFUL`, or every primary metric is confidently `TOO_SMALL`;
- `PROMOTE_TO_FULL`: every primary metric is `PROMISING` and every guardrail is `SAFE`;
- `AMBIGUOUS`: everything else.

`PROMOTE_TO_FULL` is never a final quality claim. Full frozen-corpus evaluation remains mandatory for acceptance.

## Deltas smoke400 cannot resolve well

Zero observations are not evidence of zero risk. At 95% confidence, `zero_event_upper_probability(n)` reports the event probability still compatible with seeing zero events in n opportunities.

This matters especially for:

- new false negatives or other regressions that occur on only a handful of frames;
- rare strata with few positive frames;
- recall changes when the smoke slice contains only tens of scored people;
- small FP-rate changes when both control and candidate produce few events;
- clustered outcomes where several objects share the same scene/frame.

The exact resolution limits depend on A5's frozen smoke composition. The report records them from the observed positive-frame and object counts instead of pretending all 400 frames are independent person opportunities.

## Compute budget

If a fraction s of candidates survives smoke and each survivor is rerun on all 5000 frames, expected cost relative to running every candidate full immediately is:

`0.08 + s`.

Therefore expected savings are `0.92 - s`:

- 0% survive: 92% savings;
- 10% survive: 82%;
- 25% survive: 67%;
- 50% survive: 42%;
- 75% survive: 17%;
- 92% survive: break-even;
- 100% survive: 8% overhead.

This model counts frames, not wall-clock seconds. A4 can substitute measured per-frame cost while preserving the same survival-rate algebra.

## Reuse by A1 / A3 / A4

Prepare one `spectratrack-smoke-promotion-input-v1` JSON after the smoke selection and candidate config are locked. Prefer paired GT transitions for recall-like metrics. Use count-rate metrics for FP/invocation/rare-event rates where appropriate.

Run:

```powershell
cd pc
python -m spectratrack.smoke_stats --input benchmarks\vnext\smoke_stats\fixtures\example_input.json
```

The output schema is `pc/benchmarks/vnext/smoke_stats/smoke_promotion_result.schema.json`. The example result fixture is generated deterministically from the example input.

## Statistical limits

The intervals are deliberately conservative engineering bounds for triage. They do not repair selection bias, scene clustering, temporal dependence, or subgroup sparsity. Hypergeometric representativeness calculations assume the fixed deterministic selection behaves like an outcome-independent without-replacement sample for the property being discussed.

If that assumption is not defensible for a metric/subgroup, the correct category is `AMBIGUOUS` and the full frozen corpus should be run.


## A5 smoke400 publication validated during A6 run

A5 published the exact slice while A6 was in progress. A6 did not choose or modify any frame.

Published/verified provenance:

- A5 branch head: `2c40b522433b986a213689f7152834d2933e29c5`;
- parent revision: `nightowls-public-slice5000-r1`;
- parent corpus SHA-256: `1ba30ef5adad0f6bedba4319d1c3b5f246d4c576c92b26f8b98b68b4b94a0ae8`;
- parent canonical GT SHA-256: `6cf1533e9c86cf9bc536373c3b707098cc30353b2e264f61c931887d213a9abc`;
- smoke revision: `nightowls-public-smoke400-r1`;
- smoke corpus SHA-256: `70c29ecd91ede9239ebed2949ea46e4b07b63e842aef630c0752ee41b9620162`;
- smoke JSONL SHA-256: `45ba10895c8a98f433bc7bad8e488b311a31ad8acf8459656d0cd775fe3fc0e8`;
- smoke manifest file SHA-256: `3f65339609de78e7ef255e94875727a4bb26c4769231763a2660e1bb8f15905e`;
- selection proof SHA-256: `0838ecfd1341eb5f0193ad369516098be5a7d8b53d77302f8e2e4a4089c34940`;
- selection identity SHA-256: `2a4da42830400b14405de01a82e3b3bee3d17a07db0be067f752545664fc6186`.

A6 independently loaded both canonical JSONL files with the existing `qa_benchmark` loader and re-hashed them:

- FULL5000: 5000 frames, 605 positive frames, 969 scored pedestrians;
- SMOKE400: 400 frames, 109 positive frames, 183 scored pedestrians.

These counts exactly match A5's post-freeze composition.

### Representativeness result

Under a simple-random 400-of-5000 reference model, the parent composition would imply:

- 48.4 expected positive frames versus 109 observed;
- 77.52 expected scored pedestrians versus 183 observed.

The positive-frame standardized deviation under that simple-random reference is about 9.69 SD. This is **not** treated as a selection failure because A5 did not use a simple-random design: it intentionally takes one deterministic metadata-only hash-ranked frame from each of 143 logical sequences, then fills 257 slots by global hash rank. GT labels and candidate outcomes were not used.

Accordingly, A6 marks all simple-random hypergeometric capture calculations for this exact slice as `reference_only=true`. The smoke is useful for broad sequence coverage and is strongly positive-enriched, but it must not be used to estimate FULL5000 prevalence directly.

### Actual smoke resolution limits

With 400 frames / 109 positive frames / 183 scored-person opportunities, the 95% zero-observation ceilings are:

- any frame-level event: about **0.746%**;
- an event conditional on the 109 positive frames: about **2.71%**;
- an event across the 183 scored-person opportunities: about **1.62%**.

For the conservative paired-transition interval used by the promotion gate, even **zero** observed improvements and **zero** regressions leave approximately:

- **±0.95 percentage points** at 400 frame opportunities;
- **±3.40 pp** at 109 positive-frame opportunities;
- **±2.06 pp** at 183 object opportunities.

Therefore sub-2 pp person-recall deltas are below the practical smoke400 resolution floor in the best zero-discordance case; subgroup deltas are worse when their opportunity count is smaller. A zero FP count over 400 frames still has a two-sided Poisson-style upper rate of about **0.00917/frame**, so "zero new smoke FPs" cannot support a zero-regression claim for FULL5000.

For rare full-corpus frame events, the simple-random reference capture probability is:

- 1 event frame: 8.0%;
- 2: 15.36%;
- 5: 34.10%;
- 10: 56.60%;
- 20: 81.19%;
- 50: 98.49%.

Under that reference model at least 36 full-corpus event frames are needed for a 95% chance that smoke400 contains one. Because A5 is stratified by logical sequence, these numbers are sensitivity references rather than exact inclusion probabilities.

Committed exact analysis artifacts:

- `pc/benchmarks/vnext/smoke_stats/nightowls_smoke400_stats.input.json`;
- `pc/benchmarks/vnext/smoke_stats/nightowls_smoke400_stats.result.json`.


## Stage-aware anti-leakage research policy

Repeated smoke use is governed separately from the unchanged statistical interval gate above. The versioned policy is `pc/benchmarks/vnext/smoke_stats/research_stage_policy.v1.json`; the validator is `pc/spectratrack/research_stage_policy.py`; and the strict input contract is `research_stage_history.schema.json`.

The research cycle is ordered `r1 -> r2 -> holdout4200 -> external_future_corpus`. Each candidate freezes a candidate-spec digest, semantic-config digest, timestamp, freeze-record digest, data exposure, lineage, and declared next stage before scoring. A candidate frozen before r1 may use r1 for triage. `CLEAR_REJECT` stops that candidate. `PROMOTE_TO_NEXT` and `AMBIGUOUS` may advance only to the next eligible blinded stage; `AMBIGUOUS` is not a positive claim.

Once r1 results are exposed, any newly tuned hypothesis or hyperparameter must record `data_exposure=["r1"]` and cannot use r1 for promotion; r2 is the next eligible evidence. Once r2 is exposed, a newly tuned candidate can next use only holdout4200. Once holdout4200 is exposed, further tuning requires a new external/future corpus. The validator rejects omitted exposure, repeated stage use, stage skipping, same-candidate parameter mutation, mismatched lineage hashes, scoring before freeze, and promotion inputs carrying p-values/significance labels.

`full5000_characterization` is explicitly non-promotional aggregate characterization. After component-slice exposure it cannot be represented as independent held-out evidence and cannot carry `CLEAR_REJECT`, `PROMOTE_TO_NEXT`, or `AMBIGUOUS`.

The only freeze-eligibility exception is an independently motivated semantics-preserving performance change. It requires an unchanged `semantic_config_sha256`, no quality-stage inspiration, an explicit independence attestation, and hashed equivalence evidence. It never permits reuse of a stage already consumed by the parent lineage.

The existing smoke statistics policy remains unchanged. At this layer, an existing statistical `PROMOTE_TO_FULL` outcome is represented as `PROMOTE_TO_NEXT`; the next stage is determined by exposure history rather than by reinterpreting the uncertainty calculation.

A5 r1 provenance remains exactly bound to the published r1 hashes already recorded above. During this implementation A5 then published the disjoint partition at source commit `a339da324ea0df682179097d7d6810b6fc250240`. The policy is now hash-bound to r2 revision `nightowls-public-smoke400-r2` (GT/JSONL `75eba2c9dc3b36d0a2389bfbb080a709621ef690680ae655d85d8732e7bc6097`, corpus `9d145b4dda780052388f3b663203c519ded48f61457adef14654f84fb5549eff`, manifest `9270d46c2776aa531e1b979a1a7ebc16eaed0f6483b2c1095da834af383c4e83`, proof `df11e9dc6ca1f63019cba071ed82420de27c706b41d1d3b46df880bb3ce50faf`, hash-bundle identity `9f2afa165f294a4bce1e7c842b8a135a98876e1a0c4db4b0cbd0b8c583151cce`) and holdout revision `nightowls-public-holdout4200-r1` (GT/JSONL `cda56abf7bd48b9849d5b10467be7e192d0fea9abc3ae31fac5a2f180605c621`, corpus `c2e5091fe4e1c146301f6d411c7a9c4d816380f94515e226b25e28e90dfc190a`, manifest `19ea73abf8d97265b3b0a3e46657c09bac370c31835b7414886156994fefc901`, proof `4f1f8e977871ce3c134a4c72029035760f4377d3bddfd7a99cd3cebe568dab3d`, hash-bundle identity `011242dbdc155e28dbae90e48cb2f6162abaf49ec793619936f86b78f41ebc9a`). A5 reports r1/r2/holdout pairwise disjoint with union exactly 5000; this provenance update does not change stage semantics.
