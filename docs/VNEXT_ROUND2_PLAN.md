# SpectraTrack vNext Round 2 — Coordination Plan

Status: evidence-driven research only.

Start from the current specialist branch heads listed in `docs/VNEXT_ROUND2_EVIDENCE.md`. Do not rebase onto another specialist branch. Do not merge specialist branches together.

## Shared rules

Every role must:

- `git fetch origin` first;
- verify its exact starting HEAD;
- read `AGENTS.md`, `ARCHITECTURE.md`, `TASKS.md`, `DECISIONS.md`, `docs/VNEXT_RESEARCH_PLAN.md`, `docs/VNEXT_ROUND2_EVIDENCE.md`, and its role handoff;
- keep production behavior unchanged unless explicitly promoted later by `agent/vnext-integrator`;
- use the same frozen corpus/replay bytes for comparisons;
- record exact source commit, corpus revision, GT/replay/model hashes, provider, config, wall time, and real inference-call counts;
- prefer bounded experiments and reject candidates that only look good on one clip;
- keep large datasets/models/results out of Git;
- finish with a clear handoff: reject / continue research / candidate for later integrator review.

## A1 — Detection / Fusion

Starting branch/head:

`agent/vnext-detection @ 943abee566c45116cee2b0e72b7d2c48753891ef`

Primary question:

**Can cross-pass evidence recover the precision lost at person threshold 0.12 without giving back the recall/jitter gains?**

Required work:

1. Keep `0.12` as the weak-person research floor. Do not lower it further.
2. Add an isolated evidence-aware acceptance/scoring candidate:
   - distinguish full-frame, tile-only, and corroborated multi-pass detections;
   - low-confidence single-source detections must not be treated the same as corroborated detections;
   - retain deterministic class-aware behavior;
   - preserve the non-transitive close-person safety gates;
   - do not use tracker state inside A1.
3. Compare at minimum:
   - hard NMS;
   - conservative NMM;
   - weighted;
   - the new evidence-aware/hybrid candidate.
4. Run on more than the existing MOT17-04 600-frame subset:
   - use a fixed MOT17 sequence split for development vs validation;
   - report per-sequence and aggregate metrics;
   - use CrowdHuman validation for dense detection safety where tracking is not applicable.
5. Emit canonical replay artifacts for surviving fusion policies so A2 can consume byte-identical detections.
6. Measure:
   - precision / recall / F1 / FP / FN;
   - bbox localization IoU;
   - center/size/area jitter and temporal IoU;
   - fusion mistakes / duplicate count;
   - inference calls and wall time.

Promotion gate for A1 candidate:

- must improve recall or stability over hard NMS while recovering a material part of the precision loss;
- must not regress dense/nearby-person safety;
- must survive held-out MOT17 sequences, not only the tuning subset.

Do not spend this round on a new detector family unless an exact locally runnable model is already available with provenance. The immediate blocker is evidence policy, not model fashion.

## A2 — Tracking

Starting branch/head:

`agent/vnext-tracking @ 2a46322c9d4eb6b0bd86b3126670dc424f9abe11`

Primary conclusion from round 1:

The current tracker remains the control. The tested global-assignment / Byte-style / BoT-SORT-style / OC-SORT-style candidates are rejected as wholesale replacements.

Required work:

1. Build a failure miner around the current tracker:
   - enumerate ID-switch windows;
   - fragmentation/recovery windows;
   - close crossings;
   - competing detections for the same tracks;
   - dormant/reactivation failures;
   - camera-motion/geometry anomalies when present.
2. Categorize failures from GT/replay evidence; do not guess from aggregate counts.
3. Implement at most one narrow candidate at a time. Priority:
   - ambiguity/hysteresis guard for close competing associations;
   - conservative short-occlusion continuity;
   - optional post-association bbox smoothing, strictly separated from association identity.
4. Do not replace the two-stage high/low semantics.
5. Run every tracker candidate on byte-identical A1 replays.
6. Re-run on A1 round-2 surviving replay(s), not only the old hard/NMM/weighted set.

Promotion gates:

- tracking recall must not fall by more than 0.25 percentage points from the current tracker on the same replay;
- ID switches must decrease materially, target >=10% on the validation set;
- fragmentation must not increase by more than 5%;
- false track creations must not increase materially;
- bbox smoothing, if used, must improve stability without changing track IDs or hiding misses.

If a targeted candidate cannot beat the current tracker under these gates, keep the current tracker.

## A3 — Enhancement Efficiency

Starting branch/head:

`agent/vnext-enhancement @ 86b1460b6f9879038de92eb1e1341a96221e6c03`

Round-1 conclusion:

Broad adaptive enhancement is not a viable default. It is too expensive and introduces too much FP evidence.

Required work:

1. Treat `no enhancement` as the baseline candidate.
2. Drop broad quality-only activation as the expensive-inference trigger.
3. Research a strict weak-evidence gate using detector/track context:
   - raw weak-person evidence near threshold;
   - suspect/track ROI context;
   - quality evidence as a secondary condition, not the sole trigger;
   - at most one enhanced inference per source frame in the first candidate.
4. Compare only a compact finalist set:
   - no enhancement;
   - bilateral;
   - current_adaptive_cached;
   - sharpen only if the strict gate makes it competitive.
5. Preserve the rule that enhanced-only evidence without acceptable raw corroboration cannot create a trusted new person.
6. Evaluate **post-fusion** quality, not only pre-fusion FP deltas.
7. Report recovered GT per extra inference call and FP cost per recovered GT.
8. Low-light/gamma+CLAHE work is deferred until the private night-CCTV review corpus exists; do not tune it on a corpus where the darkness gate never activates.

Promotion gates:

- strict call budget satisfied;
- post-fusion recall gain is real;
- FP increase is bounded and explicitly reported;
- no jitter regression large enough to undermine tracking;
- if no candidate clears the gate, recommend enhancement-off for the vNext integrator.

## A4 — Performance / Runtime Scheduler

Starting branch/head:

`agent/vnext-performance @ 1cd9f38a09d35f3b125c4a214d8514200ee0f76d`

Primary question:

**What scheduler is actually usable on RX 5700 XT once wall time, detector cadence, and discovery latency are measured together?**

Required work:

1. Replace simulation-only conclusions with target-PC measurements using the exact YOLO11x/960 model and DirectML environment used by the current evidence.
2. Include detector cadence in the search. Sweep bounded combinations such as:
   - `detect_every` 2/3/4/5;
   - max detector calls per detector frame 1/2/3;
   - periodic global rediscovery bounds around 0.5–1.0 source seconds;
   - track-guided/suspect ROI limits.
3. Measure real:
   - processing seconds/source second;
   - ONNX calls/source second;
   - per-call latency distribution;
   - preprocess/inference/postprocess;
   - queue/backpressure behavior where measurable.
4. Keep global new-person discovery bounded. ROI-only indefinite blindness is forbidden.
5. Start from `TRACK_GUIDED` and `BUDGETED_ADAPTIVE`, but add a lower-call `LITE` candidate if needed.
6. Consume A3's strict enhancement-call budget rather than assuming all eligible ROIs are enhanced.
7. Produce a Pareto table against A5/A1 quality evidence when available.

Research targets, not release promises:

- drive average detector work toward <=15 ONNX calls/source-second on 1080p;
- target <=2 processing seconds per source second on the RX 5700 XT if quality permits;
- retain <=1 second periodic global discovery bound.

If those targets are impossible with YOLO11x/960, state that explicitly and hand the model/input-size bottleneck back to A1 rather than hiding it with micro-optimizations.

## A5 — QA / Experiment Control

Starting branch/head:

`agent/vnext-qa @ e7251c964c63da496caac506d4d7da1876977230`

Required work:

1. Preserve `mot17-public-r1` and `crowdhuman-val-fbox-r1` as separate immutable public evidence sets.
2. Add a round-2 evidence bundle/leaderboard that can ingest A1-A4 artifacts from the same public revision without inventing an overall score.
3. Prepare a **private CCTV review pack** from the existing local user test videos:
   - `1234.mp4`;
   - `12345.mp4`;
   - `video_2026-09-25_18-24-46.mp4`.
4. Minimize manual work:
   - stratified frame extraction;
   - current-detector pre-annotation as draft only;
   - prioritize detector disagreement, small people, crossings, dark/compressed/blurred frames, and negatives;
   - produce one compact review batch.
5. Do **not** freeze AI-only boxes as GOLDEN.
6. After explicit human confirmation, freeze `cctv-golden-r1` and rerun/stamp surviving A1-A4 candidates.
7. Keep public and private evidence separate in reporting.

The private CCTV freeze is the production integration gate.

## Architect / integrator gate

Do not start production integration yet.

`agent/vnext-integrator` is unlocked only after:

1. A1 has a surviving evidence-aware fusion policy;
2. A2 either has a targeted tracker improvement or explicitly retains current tracker;
3. A3 has a strict enhancement decision, including the possibility of enhancement-off;
4. A4 has real target-PC wall-time evidence and a bounded scheduler candidate;
5. A5 has `cctv-golden-r1` human-confirmed and the surviving candidate table is complete.

At that point create `agent/vnext-integrator` from the immutable product baseline and integrate only the accepted minimum changes with full regression/packaging tests.


## Mandatory Git-persistent execution log

Project-owner requirement for all Round-2 roles and the architect:

1. Before a major experiment/code step, ensure the intended step and current state are represented in a Git-tracked role/coordination file.
2. After the step, commit the outcome before moving to the next major step.
3. Use explicit states: `DONE`, `IN PROGRESS`, `FAILED`, `BLOCKED`, `NOT DONE`.
4. Failed attempts stay in history with the exact failure cause; do not rewrite them away.
5. For local-only large artifacts, record in Git:
   - artifact path;
   - source branch/commit;
   - corpus/model/provider/config provenance;
   - hash when practical;
   - measured summary;
   - whether the artifact is accepted or rejected evidence.
6. Never leave the only copy of important state in chat, terminal history, process memory, or a local temp log.
7. Each role must update its own handoff/status documentation before claiming the branch is ready for architect review.
8. The architect maintains:
   - `docs/VNEXT_ROUND2_WORKLOG.md` — chronological append-only coordination record;
   - `docs/VNEXT_ROUND2_STATUS.md` — compact current state and remaining gates.

A session restart must be recoverable from Git alone, except for re-verifying whether local long-running processes are still alive.
