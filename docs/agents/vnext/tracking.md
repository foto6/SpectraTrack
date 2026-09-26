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


## Round 2 supplement — ambiguity-scoped assignment candidate

Research-only candidate:

`current-ambiguity-guard`

Validated code HEAD before this documentation update:

`c552b0f45e033628e188f91df11839025c2ed359`

Purpose:

- preserve the current tracker everywhere association scores are not ambiguous;
- detect score-close competition using the same 0.08 diagnostic margin as the failure miner;
- construct a local connected competition context through each participating node's two best accepted alternatives;
- apply maximum-total-score one-to-one assignment only inside that ambiguous component;
- keep current greedy ordering for all remaining edges.

Unchanged:

- production `tracker.py`;
- high/low two-stage semantics;
- strong-only new-track creation;
- candidate gates/scoring;
- CMC;
- appearance handling;
- dormant reactivation/lifecycle;
- detector/replay bytes.

### Failed first implementation

The first implementation only connected alternatives that were themselves within the 0.08 margin.

CI run `36240587164` exposed that this was too narrow:

- target test: `nearby_same_class`;
- expected ambiguity guard to resolve the known greedy conflict;
- observed: 4 ID switches, unchanged from current.

Reason:

- the ambiguous detection connected both tracks, but the second detection needed for the 2x2 one-to-one alternative was outside the margin;
- the candidate therefore formed a 2-track/1-detection component and did not invoke the local global solve.

That result is rejected and retained here as a failed research attempt.

### Fixed component context

Commit:

`c552b0f45e033628e188f91df11839025c2ed359`

The component is still seeded only by close-score ambiguity, but expands through each participating node's top two accepted alternatives. This supplies the missing one-to-one context without switching every frame to global assignment.

Deterministic tests now verify:

- the nearby-same-class greedy conflict is resolved with 0 ID switches;
- non-ambiguous camera-pan, weak-detection, and dormant-reactivation scenarios remain metric-identical to the current tracker.

### CI

GitHub Actions run:

`36240708158`

Result: **SUCCESS**

Observed:

- ruff: passed;
- compile + pytest: **166 passed in 2.09 s**;
- tracker smoke: 500 frames / 24 targets / 11970 observations;
- tracker smoke elapsed: 0.304 s;
- tracker smoke throughput: 1643.4 tracker FPS;
- crossing ID switches: 0;
- reappearance ID switches: 0;
- diagnostics/self-check: passed;
- standalone Windows build/smoke/package/upload: passed.

### Decision status

**CONTINUE RESEARCH / NOT A PRODUCTION CANDIDATE YET.**

The candidate now passes the targeted mechanism test and preserves selected non-ambiguous probes, but it still needs evaluation on the exact real replay artifacts under the Round-2 gates:

- tracking recall loss <= 0.25 percentage points vs current;
- target >=10% ID-switch reduction;
- fragmentation increase <=5%;
- no material false-track increase.

Do not integrate into production until those replay results exist and A5 stamps the surviving evidence.


## Round 2 public-first tracking assignment — 2026-09-26

Status: **IN PROGRESS / WAITING ON A5 DANCETRACK FREEZE**

Current tracker remains CONTROL. The only targeted candidate for this round remains:

`current-ambiguity-guard`

Do not return to wholesale global/Byte/BoT/OC replacements.

### MOT17

Finish current vs ambiguity-guard on byte-identical A1 canonical replays with the existing gates:

- tracking recall loss <= 0.25 percentage points;
- ID-switch improvement target >= 10%;
- fragmentation increase <= 5%;
- no material false-track increase.

### DanceTrack — association-only first

After A5 publishes `dancetrack-public-r1`:

1. derive tracker observations directly from official GT geometry/identity under a documented deterministic detection-emulation policy;
2. feed the exact same observation bytes to current and `current-ambiguity-guard`;
3. measure ID switches, fragmentation, uninterrupted length, recovery, wrong/same-ID recovery, false track creation where meaningful;
4. use no YOLO inference in this first phase.

Purpose: isolate association/crossing behavior from detector misses and bbox noise.

Only if ambiguity-guard survives this association-only gate should A2 consume a detector-generated DanceTrack replay in a second phase.

Do not modify production `tracker.py` before final integrator review.


### DanceTrack association-only observation contract

For the first DanceTrack gate, tracker input must not contain GT identity information.

For each official scored GT box on an annotated frame, construct a deterministic research `Detection`:

- bbox = canonical GT geometry;
- label/class = person;
- score = `1.0`;
- appearance = `null`;
- detector_ran = true for evaluated frames.

GT track ID is retained **only in the evaluator**, never copied into tracker input or appearance metadata.

Frames without a GT observation for a person naturally create an observation gap. Do not synthesize detections through occlusion.

This phase performs 0 ONNX inference calls and isolates geometry/lifecycle/association behavior. Any later detector-replay DanceTrack phase is a separate experiment with its own provenance.


## Round 2 execution checkpoint — 2026-09-26 replay promotion gate

Status: **COMPLETED — RETAIN CURRENT TRACKER**

Starting A2 HEAD verified before this execution step:

`372c96e71e504a54ea2ac027ab988dfba2bd1918`

Canonical coordination state read from:

`coord/vnext-round2 @ 937ca1244583369355b969fa3c01577f00a51d9f`

### DONE — reproducible current vs ambiguity-guard scorer

Research commits:

- `10f9a72e835ea5f489fdcab21010b99e12822304` — add canonical-GT + canonical-replay Round-2 scorer and promotion gates;
- `fb7809d67851640a35c7ee848418497c8985c65c` — cover identical-replay comparison and gate decisions;
- `f8906a65b8bbe51c68dc0187aa0d335901656f71` — suppress canonical ignored person regions from false-track accounting and reject incomplete GT coverage;
- `6895a39812e035bcf4dcde64788c28349cd987c5` — regression tests for ignored regions and missing replay-frame GT.

The scorer:

- consumes one immutable `spectratrack-detection-replay-v1` artifact for both candidates;
- loads the existing canonical A5 QA JSONL rather than creating a new GT format;
- keeps stable GT identity evaluator-only;
- does not mutate replay detections;
- records replay source SHA-256, canonical replay SHA-256, GT SHA-256 and subject commit;
- executes 0 detector policy runs and 0 ONNX inference calls;
- reports tracking recall, ID switches, fragmentation, false track creations, recovery events/latency, same-ID recovery, wrong-ID recovery and mean uninterrupted track length.

Promotion gates are encoded exactly as assigned:

- recall loss <= 0.25 percentage points;
- ID-switch improvement >= 10%;
- fragmentation increase <= 5%;
- no false-track increase (conservative interpretation because no numeric "material" tolerance was specified).

Failure returns exactly:

`RETAIN CURRENT TRACKER`

Pass returns only:

`ADVANCE TO DANCETRACK ASSOCIATION-ONLY`

It is not a production promotion.

Validation on `fb7809d67851640a35c7ee848418497c8985c65c`:

- GitHub Actions run `36254888877`: **SUCCESS**;
- Ruff: PASS;
- compile + pytest: **169 passed in 1.57 s**;
- existing tracker smoke: **1364.0 tracker FPS**, crossing/reappearance smoke switches 0;
- diagnostics/self-check: PASS;
- standalone Windows build and both CLI smokes: PASS.

The ignored-region/missing-GT follow-up at `6895a398...` is still awaiting its own final CI result at the time of this checkpoint.

### DONE — mandatory identical real A1 replay comparison

The target PC returned online after the earlier blocker. A2 did **not** restart detector inference. It inspected existing local artifacts and reused the already-produced canonical A1 control replay.

Execution worktree:

- `C:\Users\foto6\SpectraTrack-worktrees\a2`
- branch: `agent/vnext-tracking`
- exact scored-comparison subject commit: `9b733fdc40d71a81c190b7a3eefaa8ae9cc62fac`
- `git fetch origin --prune`: completed before the run
- worktree was clean before fast-forward to the exact remote A2 head.

Canonical control evidence:

- frozen corpus revision: `mot17-public-r1`
- corpus SHA-256: `8bfa6e54ab7a0160c133d8c7d0a2896b7ba254a23cf06afee2d4c9c453836759`
- GT path: `C:\Users\foto6\SpectraTrack-data\imports\mot17-public.jsonl`
- GT SHA-256: `28dcb9d197e0a098a1efb097f1589177350192a8f5f1be3e2ab5cd18d8f205c7`
- scored sequence: `golden/public/mot17/MOT17-04`
- replay frames: 600
- A1 replay path: `C:\Users\foto6\SpectraTrack-data\runs\a1-replay-mot17-04-600-hard-nms.jsonl`
- replay source-file SHA-256: `b2719a2c93c353123437497ac9513033898d3d65750a32ed633d05e8fb65b2d4`
- replay canonical SHA-256: `d27d45172a2df0a5c81ff83be2ceed03d2ed1eef922d3fde06f72b3126a89c59`
- replay producer source commit: `943abee566c45116cee2b0e72b7d2c48753891ef`
- detector/model: `current-yolo-onnx-prefusion`
- model SHA-256: `e84cbad768b218d74ecc85e3e52d84631123719a6951b3ddf6eddc850d5b3f73`
- replay provider: `DmlExecutionProvider,CPUExecutionProvider`
- input size: 960
- person threshold: 0.12
- fusion: hard NMS
- original replay-generation work: 600 detector policy runs / 5400 ONNX calls.
- **A2 tracker-only comparison itself:** 0 detector policy runs / **0 ONNX inference calls**.

Both candidates consumed the same immutable replay object/bytes and the same canonical GT. Stable GT identity stayed evaluator-only.

Generated scored artifact:

- path: `C:\Users\foto6\SpectraTrack-data\runs\a2-round2-current-vs-ambiguity-mot17-04-hard-nms.json`
- size: 3976 bytes
- local timestamp: 2026-09-26 23:30
- artifact SHA-256: `bdb5dacb673cc80c2d3a02f977d4d7a0a6c44dd50c685b4c174c9578f001927b`
- schema: `spectratrack-tracking-round2-comparison-v1`
- matching IoU: 0.30.

#### Current control

- tracking recall: **0.8584821095** (22985 / 26774 GT observations)
- ID switches: **335**
- fragmentations: **281**
- false track creations: **147**
- recovery events: **281**
- mean recovery latency: **7.2811 frames**
- same-ID recoveries: **165**
- wrong-ID recoveries: **116**
- mean uninterrupted track length: **41.1181 frames**
- uninterrupted segments: 559
- tracker-only wall time: 6.2113 s
- tracker-only CPU time: 6.1563 s.

#### current-ambiguity-guard

- tracking recall: **0.8582953612** (22980 / 26774)
- ID switches: **397**
- fragmentations: **289**
- false track creations: **147**
- recovery events: **289**
- mean recovery latency: **7.1349 frames**
- same-ID recoveries: **166**
- wrong-ID recoveries: **123**
- mean uninterrupted track length: **36.9453 frames**
- uninterrupted segments: 622
- tracker-only wall time: 6.8605 s
- tracker-only CPU time: 6.8281 s.

#### Promotion gates

Measured delta candidate - control:

- tracking recall: -0.0001867483 = **-0.01867 percentage points** -> PASS vs max 0.25 pp loss;
- ID switches: **+62**, equivalent to **-18.51% improvement** (18.51% worse) -> **FAIL** vs target >=10% improvement;
- fragmentations: +8 = **+2.847%** -> PASS vs max +5%;
- false track creations: +0 -> PASS;
- recovery events: +8;
- same-ID recoveries: +1;
- wrong-ID recoveries: **+7**;
- mean uninterrupted track length: **-4.1727 frames**.

Promotion result:

`RETAIN CURRENT TRACKER`

This is the assigned fail-safe result, not an inconclusive outcome. The ambiguity guard does not advance.

#### Failure-mechanism diff

A second read-only diagnostic artifact compared exact GT switch-event keys between control and guard on the same replay:

- path: `C:\Users\foto6\SpectraTrack-data\runs\a2-round2-switch-diff-mot17-04-hard-nms.json`
- artifact SHA-256: `c5b0f5cd92a20ba443e183d3c81b8b8cf9f8bf5a45344be5df7a08054803d62c`
- replay source SHA-256: `b2719a2c93c353123437497ac9513033898d3d65750a32ed633d05e8fb65b2d4`
- current switch events: 335
- guard switch events: 397
- common switch events: 277
- current-only switch events removed by guard: **58**
- guard-only new switch events: **120**.

Interpretation: ambiguity-scoped maximum-score assignment does correct some current greedy mistakes, but on this dense real sequence it creates roughly twice as many new identity discontinuities as it removes. The candidate optimizes local tracker association score, which is not sufficiently aligned with GT identity continuity to justify promotion.

No alternate replay was searched for a more favorable result after this gate failed.

### DONE — evaluator hardening / validation

Two evaluator-only correctness issues were found before claiming the real result:

1. canonical ignored person regions must suppress false-track counts;
2. non-person tracks present in the A1 replay must not count as false **person** track creations.

Fix commits:

- `f8906a65b8bbe51c68dc0187aa0d335901656f71` — ignore-region support + strict replay-frame GT coverage;
- `6895a39812e035bcf4dcde64788c28349cd987c5` — migrate/add ignore and missing-GT tests;
- `8c5d5884a5f1ec54734d535cdf9aa806008a5c2b` — score false tracks only for the evaluation label;
- `9b733fdc40d71a81c190b7a3eefaa8ae9cc62fac` — regression test for non-person false-track exclusion.

Retained failed CI attempt:

- run `36255182487` at `f8906a65...`: Ruff PASS, **1 failed / 168 passed**;
- cause: an existing stable-ID test fixture supplied only one GT row, so the newly-correct missing-frame guard fired before the expected stable-ID error;
- this was a test migration failure, not a tracker metric result;
- fixed by `6895a398...`.

Validation:

- `6895a398...`, run `36255218128`: **171 passed**, full Windows CI green;
- final scored code HEAD `9b733fdc...`, run `36255611233`: **172 passed in 1.47 s**, Ruff PASS, compile PASS, tracker smoke **1352.3 FPS**, crossing/reappearance smoke switches 0, diagnostics PASS, `SELF_CHECK=PASS`, standalone build + both CLI smokes PASS;
- target-PC focused tests on exact `9b733fdc...`: **29 passed in 0.34 s**, focused Ruff PASS.

### WAITING — DanceTrack association-only

Current A5 branch re-checked after importer work:

`agent/vnext-qa @ bf63820f81cd13fdfae8680a200e25f030d314fc`

A5 has implemented the isolated DanceTrack importer and canonical provenance plumbing, but its tracked state still reports:

- real DanceTrack import: **NOT RUN**;
- frozen `dancetrack-public-r1`: **NOT FROZEN**;
- real artifact path/hash: **NOT AVAILABLE**;
- frozen corpus hash: **NOT AVAILABLE**.

Therefore A2 has not started the association-only DanceTrack experiment.

Additionally, `current-ambiguity-guard` already failed the mandatory identical-A1-replay promotion gate above, so it is **not advanced** to DanceTrack in this Round-2 state. Even if A5 freezes DanceTrack later, architect direction would be required to override the explicit `RETAIN CURRENT TRACKER` gate outcome.

When/if explicitly re-opened after a valid freeze, phase 1 remains exactly:

- canonical GT bbox -> tracker Detection;
- person class/label;
- score 1.0;
- appearance null;
- detector_ran=true;
- GT ID evaluator-only;
- no synthesized observations through occlusion;
- **0 ONNX inference calls**;
- current and `current-ambiguity-guard` consume identical observation bytes.

Detector-replay DanceTrack remains forbidden until the ambiguity guard first survives this association-only gate.


## Round 2 continuation checkpoint — 2026-09-26 final A2 gate state

Exact A2 remote HEAD re-verified before this documentation update:

`agent/vnext-tracking @ 9b733fdc40d71a81c190b7a3eefaa8ae9cc62fac`

The user-provided earlier checkpoint `372c96e71e504a54ea2ac027ab988dfba2bd1918` is an ancestor. Seven subsequent A2 commits already completed and hardened the assigned identical-replay promotion experiment; they were preserved and not replayed or overwritten.

Code validation at exact scored/evaluator HEAD `9b733fdc...`:

- GitHub Actions run `36255611233`: **SUCCESS**;
- compile + pytest: **172 passed in 1.47 s**;
- tracker smoke: **1352.3 FPS**, crossing/reappearance smoke ID switches 0;
- diagnostics: PASS;
- self-check: PASS;
- standalone Windows build and both CLI smoke tests: PASS.

### Final Round-2 decision for current-ambiguity-guard

Mandatory identical canonical A1 replay comparison already completed on `mot17-public-r1` / MOT17-04 first 600 frames.

Control artifact provenance:

- canonical replay file SHA-256: `b2719a2c93c353123437497ac9513033898d3d65750a32ed633d05e8fb65b2d4`;
- canonical replay semantic SHA-256: `d27d45172a2df0a5c81ff83be2ceed03d2ed1eef922d3fde06f72b3126a89c59`;
- canonical GT SHA-256: `28dcb9d197e0a098a1efb097f1589177350192a8f5f1be3e2ab5cd18d8f205c7`;
- corpus revision: `mot17-public-r1`;
- corpus SHA-256: `8bfa6e54ab7a0160c133d8c7d0a2896b7ba254a23cf06afee2d4c9c453836759`;
- comparison artifact SHA-256: `bdb5dacb673cc80c2d3a02f977d4d7a0a6c44dd50c685b4c174c9578f001927b`;
- switch-diff artifact SHA-256: `c5b0f5cd92a20ba443e183d3c81b8b8cf9f8bf5a45344be5df7a08054803d62c`;
- tracker-only ONNX inference calls: **0**.

Promotion-gate result remains:

`RETAIN CURRENT TRACKER`

Reason:

- recall loss: **0.01867 percentage points** -> PASS;
- ID switches: **335 -> 397** -> **FAIL**, 18.51% worse instead of >=10% improvement;
- fragmentation: **281 -> 289** (+2.847%) -> PASS;
- false track creations: **147 -> 147** -> PASS;
- wrong-ID recovery: **116 -> 123** (worse);
- mean uninterrupted track length: **41.1181 -> 36.9453 frames** (worse).

Failure-window diff confirms the mechanism rather than only the aggregate:

- current switch events: 335;
- guard switch events: 397;
- common: 277;
- current-only switches removed by guard: 58;
- new guard-only switches: 120.

Interpretation: the ambiguity guard repairs some greedy mistakes but introduces about twice as many new GT identity discontinuities as it removes. Local maximum-score ambiguity resolution is not aligned strongly enough with identity continuity on this dense real replay.

No alternative replay was searched and no guard threshold was retuned after observing the held-out result.

### DanceTrack gate status

A5 was re-checked at:

`agent/vnext-qa @ bf63820f81cd13fdfae8680a200e25f030d314fc`

A5 still reports:

- real DanceTrack validation import: **NOT RUN**;
- real artifact hashes: **NOT AVAILABLE**;
- `dancetrack-public-r1`: **NOT FROZEN**;
- frozen corpus hash: **NOT AVAILABLE**.

A2 therefore has no valid DanceTrack artifact to consume.

More importantly, the assigned prerequisite for advancing `current-ambiguity-guard` has already failed on the identical A1 replay. Under the Round-2 promotion contract, A2 does **not** advance the rejected guard to DanceTrack and does **not** start detector-replay DanceTrack. Running it anyway would override the explicit fail-safe gate rather than continue the assigned protocol.

If the architect explicitly re-opens DanceTrack for CONTROL-only characterization or for a new narrow candidate later, the association-only input contract remains:

- canonical GT bbox copied to tracker Detection;
- person class/label;
- score 1.0;
- appearance null;
- detector_ran=true;
- GT identity evaluator-only;
- 0 ONNX inference calls;
- no synthetic detections through occlusion.

### A2 Round-2 readiness

- current tracker: **RETAIN AS CONTROL**;
- `current-ambiguity-guard`: **REJECTED BY PROMOTION GATE**;
- DanceTrack guard evaluation: **NOT ELIGIBLE under current gate outcome and also blocked by missing freeze**;
- production `tracker.py`: unchanged;
- next action: architect review / explicit new narrow hypothesis only. Do not resume wholesale global/Byte/BoT/OC replacement work.
