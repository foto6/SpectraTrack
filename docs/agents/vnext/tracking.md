# A2 — vNext Tracking / MOT Research Handoff

Branch:

`agent/vnext-tracking`

Exact research base:

`vnext-base @ d03af3ae6425d3ea2e4d52e25389fecc09957394`

Immutable product baseline:

`integration @ 3ebc4d50213593cac62b97399447fccf6bbc1755`

Validated research-code HEAD before this state-only handoff commit:

`ce6d3d1c07e702abd047eef4e523af151e0db9cd`

No specialist branch was merged or cherry-picked into A2. `integration`, `main`, RC and other agent branches were not modified.

## 1. Scope and result

A2 stayed research-only.

Implemented:

- strict reader/writer support for canonical `spectratrack-detection-replay-v1`;
- immutable replay records with canonical SHA-256 and fresh per-candidate `Detection` objects;
- deterministic current-`MultiObjectTracker` audit harness;
- 15 deterministic failure probes covering the required MOT cases;
- weighted ID/recovery/continuity metrics;
- a deterministic greedy-assignment counterexample;
- an isolated `current-global-assignment` candidate that changes only one-to-one assignment selection while preserving current tracker gates/lifecycle/CMC/appearance/dormant logic;
- dependency-free ByteTrack-, BoT-SORT-, and OC-SORT-style mechanism probes;
- post-association raw / bounded-EMA / motion-state bbox stability probes;
- reproducibility notes under `pc/benchmarks/vnext/tracking/`.

Not changed:

- production `pc/spectratrack/tracker.py`;
- detector/fusion behavior;
- enhancement;
- runtime scheduler;
- QA algorithm implementation;
- `integration` / `main`.

The current production tracker already has ByteTrack-like high/low association. A2 did **not** add a second low-confidence rescue path.

## 2. Files changed

Only research/tests/docs:

- `pc/spectratrack/detection_replay.py`
- `pc/spectratrack/tracking_research.py`
- `pc/tests/test_detection_replay.py`
- `pc/tests/test_tracking_research.py`
- `pc/benchmarks/vnext/tracking/README.md`
- `pc/benchmarks/vnext/tracking/.gitignore`
- this role-state file

No production tracker file is in the branch diff.

## 3. Commits owned before handoff state

From `vnext-base` to validated research-code HEAD:

1. `487a020994c2ffb8e7e7f2cfa1e57ef9639a7d8f` — research: add canonical detection replay reader
2. `4a4637a38b36e300710a8b2f583d4dcb0e6c275d` — tests: validate deterministic detection replay
3. `d37a14c3fb2d17c9cf893ba2210aaa266c213033` — research: add deterministic MOT replay bakeoff
4. `2cbbf8cd0c9e671d9fdc181d3c60388a142314b6` — tests: cover MOT research replay and smoothing
5. `7ca4a97a4758339312fa89faaa40c86ead1745ce` — research: count same-id recovery after tracking gaps
6. `1c41b73130e8831730c579ca5e5dbe67565a3958` — docs: document vNext tracking research harness
7. `c19b606817740ce594b4987b0eb2999773ca5fd8` — chore: ignore generated tracking research artifacts
8. `7d2d8748437bfda63a8b0aff9ea4d54dff14f4c2` — research: add greedy assignment conflict probe
9. `c5fb05fc8950eff3eda488cf7beb49c3391c5ce9` — tests: pin greedy assignment counterexample
10. `08ea0e6dd41fd7d59096a68da034a2b58df88120` — research: isolate global assignment candidate
11. `01b91fa27da4ba0dc1966c9e3e7a11c185855776` — tests: validate isolated global assignment candidate
12. `8126f39ed85d1cb33d4d3867439e8cb1180196bf` — docs: record global assignment research candidate
13. `a7cd4bb58d82979229ebfb9ff326100952fb37bb` — research: weight aggregate tracking means by events
14. `ce6d3d1c07e702abd047eef4e523af151e0db9cd` — tests: verify weighted research metrics

The final handoff commit updating this file is reported externally because a commit cannot contain its own SHA.

## 4. Detection replay compatibility and determinism

Canonical schema:

`spectratrack-detection-replay-v1`

A2 replay guarantees:

- source bytes receive SHA-256 when loaded from file;
- canonical serialization has a deterministic SHA-256;
- frame indices must be strictly increasing;
- metadata/video identity and dimensions are validated;
- bbox/score/class/label are validated;
- every candidate receives newly constructed `Detection` objects, so one candidate cannot mutate the next candidate's input;
- tracker-only research performs **0 detector policy runs** and **0 ONNX inference calls**.

Determinism tests actually verify:

- canonical write -> load -> write identity;
- repeated load equality;
- fresh `Detection` object isolation;
- repeated current-tracker replay produces byte-equivalent frame results and the same canonical replay hash;
- all candidate rows consume the same 15 scenario replay fingerprints.

A1 compatibility was checked against the current A1 replay writer. Its metadata/frame records are directly accepted by A2.

Important A5 comparability constraint discovered during cross-agent review:

- A5 requires full 40-hex source commit provenance;
- A5 requires explicit versioning such as `config.appearance_schema` when non-null appearance vectors are present;
- A2's parser is intentionally more permissive than A5's common-evidence gate.

Therefore a future common-corpus A2 result must pass A5 validation/stamping even if A2 itself can parse the replay.

## 5. Synthetic audit input

Pre-A5 deterministic suite:

- 15 replay scenarios;
- 183 total frames;
- synthetic frame size 640x360;
- source commit recorded as `d03af3ae6425d3ea2e4d52e25389fecc09957394`;
- detector ID `synthetic-deterministic-v1`;
- detector policy runs: **0**;
- ONNX inference calls: **0**.

Scenarios:

- crossing;
- partial occlusion;
- full occlusion;
- short dropout;
- long dropout;
- dormant reactivation;
- appearance mismatch;
- camera pan;
- affine camera transform;
- detector-skipped frames;
- weak detection sequence;
- false weak detections;
- nearby same-class people;
- changing bbox scale;
- sudden direction change.

Synthetic probes are mechanism evidence only, not CCTV-quality evidence.

## 6. Exact current-tracker failure analysis

### Crossing

Observed: **0 ID switches, 0 fragmentation, recall 1.0** in the simple symmetric crossing probe.

Decision path:

- both detections stay inside current class/geometry candidate gates;
- greedy association happens to preserve the trajectory ordering in this specific geometry.

Conclusion:

- this probe does **not** prove crossings are generally safe;
- the nearby asymmetric same-class probe below exposes the actual greedy failure deterministically.

### Partial occlusion

Observed: **0 switch, 0 fragmentation, recall 1.0**.

At the weak/narrow frame:

- detection score = 0.28, below `high_conf=0.45`;
- high stage has no match;
- existing low/unmatched stage evaluates it with loose gates;
- IoU = **0.433333** vs loose IoU gate **0.065**;
- center ratio = **0.019670** vs loose center gate **2.375**;
- appearance = **1.0**;
- candidate score = **2.623718**.

Result: existing ID is maintained by the low-confidence rescue that already existed before vNext.

### Full occlusion

Synthetic GT is intentionally invisible during the short full-occlusion interval.

Observed: **0 switch, 0 fragmentation, visible-GT recall 1.0**.

Decision path:

- missing detections increment `missed`;
- gap is shorter than `max_missed=14`;
- predicted live track remains available;
- reappearance associates back to the same live track.

### Short detector dropout

GT stays visible but detections are absent for two frames.

Observed: **0 switch, 0 fragmentation, tracking recall 1.0**.

Decision path:

- only two misses, below deletion budget;
- prediction remains spatially close enough to GT/detection;
- the same live ID resumes.

### Long detector dropout without appearance

16 consecutive detector misses while GT remains visible.

Observed:

- tracking recall **22/24 = 0.916667**;
- **1 ID switch**;
- **1 fragmentation**;
- recovered same ID = 0;
- wrong recovery = 1;
- recovery latency = 2 frames.

Decision path:

- on the 15th miss, confirmed track exceeds `max_missed=14`;
- because `track.appearance is None`, it is not inserted into dormant recovery;
- live output is absent for two frames;
- later strong detection is unmatched and creates a new identity.

Root cause: lifecycle expiry plus the intentional “no appearance -> no dormant reactivation” safety rule, not missing ByteTrack rescue.

### Dormant reactivation with matching appearance

Same long gap, but a stable appearance vector is present.

Observed:

- tracking recall **22/24 = 0.916667**;
- **0 ID switches**;
- **1 fragmentation**;
- recovered same ID = 1;
- wrong recovery = 0;
- recovery latency = 2 frames.

Decision path:

- expired confirmed track enters dormant pool;
- reappearing strong detection has same class;
- appearance similarity = 1.0, above `reactivation_min_appearance=0.90`;
- shape/size/spatial gates pass;
- combined dormant score passes `reactivation_min_score=0.90`;
- original local ID is restored.

Important limitation: dormant recovery preserves identity but does not fill the two-frame output gap, so fragmentation/recall loss still exists.

### Appearance mismatch after long gap

Same geometry, but orthogonal appearance vector after the gap.

Observed:

- tracking recall **0.916667**;
- **1 ID switch**;
- **1 fragmentation**;
- recovered same ID = 0;
- wrong recovery = 1.

Decision path:

- dormant appearance cosine is below 0.90;
- reactivation is rejected;
- unmatched strong detection creates a new ID.

This is expected conservative behavior and prevents an unsafe same-ID claim when appearance disagrees.

### Camera pan

Observed: **0 switch, recall 1.0**.

With +30 px replay camera motion:

- CMC shifts the predicted box first;
- IoU = **1.0**;
- center ratio = **0.0**;
- association score = **3.535**.

Result: camera translation is not interpreted as target motion.

### Affine camera transform

Observed: **0 switch, recall 1.0**.

At the first transformed frame:

- affine CMC transforms the previous bbox before scoring;
- IoU = **1.0**;
- center ratio = **0.0**;
- score = **3.535**.

Result: replayed scale/translation camera transform preserves the ID in this deterministic probe.

### Detector-skipped frames

Observed: **0 switch, 0 fragmentation, recall 1.0**.

Decision path:

- `detector_ran=false` calls `predict_only()`;
- `predict_only()` advances geometry but does not increment `missed`;
- intentional detector cadence therefore does not consume the dropout budget.

### Weak detection sequence

After initial strong detections, score drops to 0.20.

Observed: **0 switch, 0 fragmentation, recall 1.0**.

At first weak frame:

- high stage: no match;
- loose stage: IoU **0.898634** vs **0.065** gate;
- center ratio **0.019828** vs **2.375** gate;
- score **3.178373**;
- same ID survives.

This directly confirms the ByteTrack-like rescue already existed.

### False weak detections

Only score-0.20 false detections, no existing tracks.

Observed: **0 false track creations**.

Decision path:

- weak detections can participate only in association with existing tracks;
- creation loop uses unmatched **high** detections only;
- therefore weak evidence alone never spawns a new ID.

### Nearby same-class people — confirmed greedy failure

This is the clearest current-tracker association failure.

Observed current tracker:

- **4 ID switches**;
- 0 fragmentation;
- recall 1.0;
- mean uninterrupted length **2.333 frames**.

At conflict frame 3, current candidate matrix includes:

- T1 -> D0 = **1.292033**
- T1 -> D1 = **0.857607**
- T2 -> D0 = **1.300685**
- T2 -> D1 = **1.146425**

Greedy current selection chooses the locally highest pair:

- T2 -> D0 = 1.300685
- then only T1 -> D1 = 0.857607 remains
- total = **2.158292**

Global maximum-score assignment chooses:

- T1 -> D0 = 1.292033
- T2 -> D1 = 1.146425
- total = **2.438458**

That isolated change produces **0 ID switches** in this scenario and uninterrupted length **7.0**.

Root cause: greedy one-to-one selection, not the IoU/center candidate gate itself.

### Changing bbox scale

Observed: **0 switch, recall 1.0**.

Final frame diagnostic:

- IoU **0.855625**;
- center ratio **0.0**;
- appearance **1.0**;
- score **3.7385**.

Result: incremental scale change is safe in this probe. No claim is made about abrupt detector box collapse/expansion on real CCTV.

### Sudden direction change

Observed: **0 switch, recall 1.0**, but the audit exposes an important two-stage boundary.

At the reversal frame:

Strict high stage:

- IoU = 0;
- center ratio = **1.958346**;
- strict center gate = **1.9**;
- candidate rejected by geometry gate.

The same unused high-confidence detection then enters the second loose stage:

- loose center gate = **2.375**;
- candidate accepted;
- score = **0.310433**;
- same ID is preserved.

So the tracker can survive this reversal specifically because the second stage also considers unmatched high-confidence detections. This is existing behavior, not a new A2 rescue path.

## 7. Greedy vs global assignment result

The isolated candidate `current-global-assignment` subclasses the current tracker and changes only `_associate()` selection from sorted greedy edges to maximum-total-score one-to-one assignment.

Everything else stays current:

- same candidate scoring;
- same high/low stages;
- same gates;
- same CMC;
- same appearance;
- same confirmation/deletion;
- same dormant recovery.

On the 15-scenario synthetic suite:

| Candidate | ID switches | Fragmentations | Tracking recall | False track creations | Mean uninterrupted length | Recovered same ID | Wrong recovery |
|---|---:|---:|---:|---:|---:|---:|---:|
| current | 6 | 3 | 0.967742 | 0 | 7.826 | 2 | 2 |
| current-global-assignment | **2** | 3 | 0.967742 | 0 | **9.474** | 2 | 2 |

The four-switch difference comes from the deterministic nearby-same-class greedy conflict. The remaining two aggregate switches are long-dropout/no-appearance and appearance-mismatch cases; assignment strategy cannot fix those lifecycle/recovery cases.

Interpretation:

- **finding:** greedy selection can create avoidable ID switches even when the candidate-score matrix already contains a better globally consistent solution;
- **hypothesis:** replacing only selection with global assignment may improve real CCTV MOT;
- **not established:** global assignment is not a production winner until the exact A5 frozen corpus shows the same benefit without new regressions/cost problems.

## 8. Reference-style bake-off actually executed

All rows below used the exact same 15 replay fingerprints and no detector inference.

These are clean-room mechanism probes, **not executions of the official upstream repositories**.

| Candidate | What was actually tested | ID switches | Fragmentations | Recall | Mean uninterrupted length | Same-ID recoveries | Wrong recoveries |
|---|---|---:|---:|---:|---:|---:|---:|
| current | production `MultiObjectTracker` unchanged | 6 | 3 | 0.967742 | 7.826 | 2 | 2 |
| current-global-assignment | current tracker, only assignment selector changed | **2** | 3 | 0.967742 | **9.474** | **2** | **2** |
| byte-global-reference-style | high/low + strong-only creation + global assignment + constant-velocity geometry | 3 | 3 | 0.967742 | 9.474 | 1 | 3 |
| botsort-reference-style | global assignment + appearance + replay CMC | 3 | 3 | 0.967742 | 9.474 | 1 | 3 |
| ocsort-style | global assignment + observation-direction cue | 9 | 3 | 0.967742 | 7.200 | 1 | 3 |

Why Byte/BoT-style probes have worse recovery counts than current:

- the clean-room probes intentionally do not copy SpectraTrack's dormant reactivation subsystem;
- the dormant-recovery scenario therefore becomes a new-ID recovery for those probes.

Why OC-SORT-style is not advanced from synthetic evidence:

- it produced **9** switches in this specific suite, worse than current;
- the simplified observation-direction probe is not an official OC-SORT implementation and must not be used to claim upstream OC-SORT itself is worse.

Official ByteTrack / BoT-SORT / OC-SORT / Deep OC-SORT repositories were **not** installed or benchmarked in this cycle. A2 reviewed their deployment/dependency shape, but did not add them to production.

Required future comparison after A5 freeze:

- exact official/reference implementation adaptation only if license/runtime/dependency cost remains reasonable;
- feed the same A1 replay bytes;
- isolate detector from tracker;
- report the same ID/recovery/stability/cost metrics.

## 9. Bbox smoothing research

Smoothing is post-association only. It never changes replay detections or ID association.

Synthetic direction-change/jitter probe:

| Smoother | Center jitter px | Width jitter px | Height jitter px | Area jitter frac | Temporal IoU | Response lag px | Mean IoU to truth |
|---|---:|---:|---:|---:|---:|---:|---:|
| raw | 5.759 | 2.110 | 2.062 | 0.05699 | 0.7444 | **2.833** | 0.8522 |
| bounded EMA | **2.318** | **1.202** | **1.183** | **0.02850** | **0.8870** | 7.919 | 0.7086 |
| motion-state alpha/beta | 2.536 | 1.360 | 1.373 | 0.03366 | 0.8775 | 6.098 | **0.8923** |

Status:

- bounded EMA strongly reduces jitter, but its response lag nearly triples and mean IoU to moving truth drops badly; **reject as current finalist**;
- motion-state smoothing reduces jitter substantially and has better mean truth IoU in this synthetic probe, but reversal lag remains more than 2x raw; **retain only as an A5 rerun hypothesis**;
- raw boxes remain the no-lag baseline.

No smoothing candidate is allowed to hide an association failure. Identity metrics are computed before smoothing.

## 10. Tracker-only performance cost

A separate 1000-repeat synthetic run was executed from the exact `ce6d3d1c...` CI source artifact in a Linux/Python 3.13.5 container.

Each repeated suite contains all 15 scenarios / 183 frames.

This is CPU/Python mechanism cost only; detector policy runs = 0 and ONNX inference calls = 0.

| Candidate | Wall ms / 183-frame suite | CPU ms / suite | Wall delta vs current |
|---|---:|---:|---:|
| current | 1.174 | 1.165 | baseline |
| current-global-assignment | 1.427 | 1.420 | +21.6% |
| byte-global-reference-style | 1.085 | 1.079 | -7.6% |
| botsort-reference-style | 1.189 | 1.182 | +1.3% |
| ocsort-style | 1.177 | 1.170 | +0.2% |

Interpretation:

- global assignment has measurable relative overhead in this tiny Python suite;
- absolute time is still tiny here;
- this does **not** establish dense-scene production cost, because Hungarian/global matching scales worse than greedy and these scenarios contain very few simultaneous tracks;
- these numbers are not RX 5700 XT / DirectML detector performance evidence.

## 11. Validation actually run

Validated research-code HEAD:

`ce6d3d1c07e702abd047eef4e523af151e0db9cd`

GitHub Actions:

- workflow run: `36168022989`
- job: `108180433197`
- conclusion: **success**
- Ruff: **PASS** — `All checks passed!`
- compileall + pytest: **164 passed in 1.60s**
- existing synthetic tracker smoke: **PASS**
  - 500 frames
  - 24 targets
  - elapsed 0.363 s
  - 1376.3 tracker FPS
  - 24 active tracks
  - crossing probe 0 switches
  - reappearance probe 0 switches
- diagnostics: **PASS**
- self-check: **PASS**, `SELF_CHECK=PASS`
- PyInstaller standalone build: **PASS**
- `SpectraTrack-PC.exe --help`: **PASS**
- `SpectraTrack-PC.exe batch --help`: **PASS**
- source and Windows artifact upload: **PASS**

PyInstaller still reports the pre-existing optional `onnxruntime.quantization` warning because `onnx` is not installed; build/smoke completed successfully.

Earlier research CI caught and fixed one evaluator bug: same-ID recovery after a detector/track gap was initially counted only when GT disappeared. The metric now counts recovery after an unmatched tracking gap as well.

## 12. A5 frozen corpus status

A5 current handoff:

`agent/vnext-qa @ c52abcdf3f156bd01cbe782cc16aeb0f0deffe5f`

Current frozen corpus:

- revision: **NOT ASSIGNED**
- corpus SHA-256: **NOT AVAILABLE**
- real CCTV baseline: **NOT MEASURED**

Reason: no real user CCTV set has yet been supplied, human-confirmed and frozen.

Therefore:

- A5 frozen corpus revision used by A2: **NONE**
- no real-CCTV tracker winner can be declared;
- official/reference finalists still require the common frozen-corpus rerun.

## 13. Recommendation vs hypothesis

### Recommendation now

Keep the production tracker unchanged.

Advance `current-global-assignment` as the **first A2 finalist for frozen-corpus evaluation** because:

- it preserves all current SpectraTrack lifecycle/gates/CMC/appearance/dormant semantics;
- it isolates one proven synthetic failure source;
- it reduces aggregate synthetic switches 6 -> 2;
- it fixes the nearby-same-class counterexample 4 -> 0 switches;
- it does not change synthetic recall, fragmentation, false-track creation, or recovery counts.

This is a research recommendation for the next comparison round, **not a production integration recommendation**.

### Secondary hypothesis

Retain motion-state alpha/beta bbox smoothing for A5 stability testing only.

It is not ready for integration because response lag after genuine motion reversal is materially higher than raw.

### Rejected / not advanced from current synthetic evidence

Bounded EMA smoothing:

- rejected as a finalist because lag increases strongly and mean IoU to moving truth falls from 0.8522 to 0.7086 despite smoother boxes.

OC-SORT-style mechanism probe:

- not advanced because this simplified probe produced 9 switches vs current 6;
- this does not reject official OC-SORT.

Byte-global / BoT-SORT-style probes:

- not preferred over the isolated current-global candidate because they produced 3 switches and lose SpectraTrack dormant-recovery behavior in this implementation;
- official implementations remain untested.

### Still only hypotheses until A5 freeze

- that global assignment reduces real crossing/nearby-person ID switches;
- that motion-state smoothing improves operator-visible bbox stability without unacceptable lag;
- that an official ByteTrack/BoT-SORT/OC-SORT integration would outperform the isolated candidate enough to justify deployment complexity.

## 14. Dependency / deployment risks

No external tracker dependency was added.

If official references are evaluated later:

- ByteTrack-style upstream stacks are more detector/training-framework coupled than the isolated assignment experiment;
- BoT-SORT-style full appearance/Re-ID stacks add substantial model/runtime dependencies;
- OC-SORT-style motion code is lighter conceptually, but upstream tooling still needs adaptation to SpectraTrack replay/runtime;
- Deep OC-SORT adds further appearance/Re-ID complexity.

For Windows local-only deployment, dependency size, model provenance, CPU/DirectML compatibility and standalone packaging must be measured, not assumed.

The current research `_maximize_assignment` implementation is pure Python and suitable for mechanism isolation. It is not automatically the production implementation; dense-scene complexity/cost must be benchmarked.

## 15. Integration risks

If the architect later chooses global assignment for production:

- do not replace current candidate scoring at the same time;
- do not alter high/low rescue semantics at the same time;
- preserve strong-only track creation;
- preserve CMC and `predict_only()` missed semantics;
- preserve dormant reactivation;
- test tie-breaking/determinism explicitly;
- benchmark dense simultaneous-person scenes for assignment cost;
- rerun current + global on exactly the same A1 replay bytes and A5 corpus;
- keep bbox smoothing as a separate experiment so identity gains are not conflated with visual smoothing.

Replay integration:

- A1 replay format is compatible;
- common evidence must satisfy stricter A5 provenance/versioning;
- A2 must add full candidate config/provenance to real-corpus evidence before A5 stamping.

No branch should merge this research directly into production. Production integration remains owned by later `agent/vnext-integrator`.

## 16. Readiness

A2 research tooling / pre-A5 failure audit: **READY FOR ARCHITECT REVIEW**

Production tracker replacement: **NOT READY**

Winner declaration: **NOT ALLOWED / NO FROZEN REAL CCTV CORPUS**

Next unlock condition:

1. A5 publishes human-confirmed frozen corpus revision/hash;
2. A1 supplies validated canonical replay bytes for that exact corpus;
3. A2 reruns current, `current-global-assignment`, and any approved official/reference finalists on the same replay;
4. A2 reports ID/fragmentation/recovery/stability/wall+CPU metrics with full config;
5. A5 validates/stamps the evidence;
6. architect decides whether any candidate proceeds to integration.


## Round 2 supplement — real failure-window miner

A research-only current-tracker failure miner now reports exact ID-switch and fragmentation/recovery windows, nearby-GT context, ambiguous association score margins, dormant reactivation matches, and first-seen false-track IDs. It does not alter production tracker behavior.

On MOT17-04 first 600 frames with the existing hard-NMS replay, the miner found:

- 335 ID-switch events;
- 281 fragmentation/recovery events;
- 468 association events where two accepted track candidates were within the configured 0.08 score margin for one detection;
- 334/335 ID switches occurred while another GT person was within the miner's close-competition context heuristic;
- 214 false track IDs.

On the first round-2 evidence-aware weighted replay, the miner found 643 ID switches and 321 fragmentation/recovery events, so that fusion policy is not a tracker-safe replacement despite its detector precision/jitter improvement.

The close-competition count is a diagnostic heuristic, not causal proof. The next candidate must target ambiguity/crossing continuity and be evaluated on identical replay bytes under the round-2 recall/IDSW/fragmentation gates.
