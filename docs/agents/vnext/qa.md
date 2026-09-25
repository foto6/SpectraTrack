# A5 — vNext CCTV QA / Golden Set Research

Branch:

`agent/vnext-qa`

Start from the exact `vnext-base` SHA in the architect handoff.

## Immediate objective

Use the existing `qa_benchmark.py` as the canonical basis and establish a small representative, human-confirmed CCTV golden corpus plus comparison workflow.

## Required corpus coverage

Include representative examples of:

- tiny people;
- distant people;
- night/dark;
- blur;
- compression;
- partial occlusion;
- heavy occlusion;
- high camera angle;
- crossings;
- moving camera;
- negative frames.

Keep a corpus manifest/revision identifier so comparison rounds can freeze exact evidence.

Do not commit large source-video binaries when repository policy excludes them; store manifests/hashes and documented acquisition paths instead.

## Ground truth rules

Auto-labeling is allowed only as pre-annotation.

Auto-label output is never final GT.

Final scored annotations must be reviewed/confirmed by a human.

Record annotation provenance and corpus revision.

## Required leaderboard metrics

At minimum:

- person recall;
- precision;
- FP;
- FN;
- recall by bbox height;
- ID switches;
- fragmentations;
- bbox jitter;
- uninterrupted track length;
- recovery latency;
- actual inference calls/frame;
- processing seconds/source second.

Comparison rows must include source commit, model/backend, model hash when applicable, provider, config, image dimensions, and corpus revision.

## Bbox stability metrics

Define bbox-stability metrics consistently and document the matching method so A1 comparisons are reproducible.

Avoid treating genuine person motion as detector jitter. Prefer GT-relative or motion-compensated error measures where the annotation density supports them.

## Synchronization role

Before the first frozen real corpus:

- build corpus/annotation tooling and experiment-table schema;
- validate evaluator behavior with synthetic fixtures only as tooling tests.

After freeze:

1. publish immutable corpus revision;
2. require A1-A4 to rerun relevant candidates on it;
3. collect results;
4. reject incomparable runs;
5. assemble one shared comparison table for architect review.

## Ownership boundaries

A5 owns:

- GT/annotation workflow;
- corpus revisioning;
- evaluator/leaderboard research tooling;
- cross-role comparability checks.

A5 does not implement detector, tracker, enhancement, or scheduler feature candidates.

Synthetic data validates the evaluator; it is not CCTV-quality evidence.
