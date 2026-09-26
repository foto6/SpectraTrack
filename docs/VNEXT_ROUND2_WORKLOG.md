# SpectraTrack vNext Round 2 — Live Worklog

Purpose: persistent engineering log for every meaningful coordination/research action, including completed, failed, blocked, and pending work.

Rules for this log:

- append every meaningful engineering action before moving on;
- record both completed and not-completed work;
- record failures and fixes, not only successful outcomes;
- include exact branch/commit/artifact when known;
- do not rewrite history to hide failed attempts;
- large local benchmark artifacts remain outside Git, but their path/hash/result summary is recorded here;
- this file is coordination history only and does not imply production approval.

## 2026-09-26 — Round 2 setup

### DONE — review round-1 evidence

Reviewed public-corpus evidence from the completed local benchmark cycle:

- detector threshold sweep on MOT17 and CrowdHuman;
- MOT17-04 fusion comparison;
- A2 tracking bake-off on identical replay bytes;
- A3 enhancement quality/cost evidence;
- A4 scheduler call-count simulation;
- final local evidence summary.

Key conclusion recorded in `docs/VNEXT_ROUND2_EVIDENCE.md`: do not production-integrate the round-1 candidates yet.

### DONE — create Round 2 coordination branch

Created:

`coord/vnext-round2`

from:

`vnext-base @ d03af3ae6425d3ea2e4d52e25389fecc09957394`

Added:

- `docs/VNEXT_ROUND2_EVIDENCE.md`
- `docs/VNEXT_ROUND2_PLAN.md`

Created draft coordination PR #24.

### DONE — assign specialist work

Created GitHub issues:

- #18 — A1 evidence-aware person fusion
- #19 — A2 tracker failure mining / targeted fixes
- #20 — A3 strict enhancement gating or enhancement-off
- #21 — A4 target-PC wall-time scheduler search
- #22 — A5 public evidence bundle + private CCTV review pack
- #23 — architect/integrator gate

Also posted round-2 instructions into the existing A1-A5 PRs.

### DONE — update background watch

Updated the existing SpectraTrack automation to watch Round 2 branches/issues/PRs while preserving these safety boundaries:

- no merge/change to `main`;
- no merge/change to `integration`;
- no RC/release;
- no branch deletion;
- no AI-only private annotations frozen as GOLDEN.

## 2026-09-26 — specialist progress discovered

### DONE — A1 new research commit discovered

A1 advanced to:

`agent/vnext-detection @ b468836c9be1fd3669b46887a56ece4e9aafcd7f`

Commit:

`research(a1): add evidence-aware weak-person fusion`

The new research candidate distinguishes strong, full-frame medium-confidence, and corroborated weak evidence.

A1 branch-focused tests run locally:

- 16 passed.

### DONE — A2 new research commit discovered

A2 advanced to:

`agent/vnext-tracking @ d83017d52011b1c728b6bb86b2fd519041da2032`

Commit:

`research(a2): mine real tracker failure windows`

A2 branch-focused tests run locally:

- 13 passed.

### DONE — A5 new research commit discovered

A5 advanced to:

`agent/vnext-qa @ b62509b6bbcf90e138b71fe48de51eb3c43c0787`

Commit:

`qa(vnext): seed human review from draft annotations`

Verified that machine draft boxes can seed annotation UI while human-saved rows take precedence and private freeze still requires explicit human review confirmation.

## 2026-09-26 — private CCTV review preparation

### DONE — extract review frames

Extracted a compact review batch from the three local user videos:

- `E:\SpectraTrack-PC\test-videos\1234.mp4`
- `E:\SpectraTrack-PC\test-videos\12345.mp4`
- `E:\SpectraTrack-PC\test-videos\video_2026-09-25_18-24-46.mp4`

Local artifact root:

`C:\Users\foto6\SpectraTrack-data\private-review`

Frame extraction results:

- 1234.mp4: 11 review frames;
- 12345.mp4: 11 review frames;
- vertical clip: 24 review frames;
- total: 46 review frames.

These are review artifacts only, not GOLDEN.

### DONE — create detector draft pre-annotations and triage

Ran current YOLO11x with DirectML on the 46 extracted review frames.

Provider:

- `DmlExecutionProvider`
- CPU fallback present in provider priority.

Generated local artifacts:

- `C:\Users\foto6\SpectraTrack-data\private-review\draft-preannotation.jsonl`
- `C:\Users\foto6\SpectraTrack-data\private-review\triage.json`
- `C:\Users\foto6\SpectraTrack-data\private-review\triage.csv`

Important rule: draft boxes and triage signals are not GT and are not frozen.

Observed triage examples:

- 12345.mp4 contains multiple weak 0.12–0.35 person detections and receives the highest review priority;
- several vertical-video frames have low Laplacian variance and no detector person boxes, making them useful potential miss/blur review cases;
- no semantic GOLDEN tags were invented from these heuristics.

### NOT DONE — human confirmation of private CCTV

`cctv-golden-r1` is **not frozen**.

Reason:

- human visual confirmation/correction is still required;
- AI draft labels cannot satisfy this gate.

This remains the main production-integration blocker.

## 2026-09-26 — A1 evidence-aware follow-up

### DONE — reproduce default evidence-aware result

On the existing MOT17-04 first-600-frame development subset, default A1 evidence-aware policy:

- weak floor: 0.12
- full-frame solo threshold: 0.20
- strong threshold: 0.35
- minimum independent sources: 2

Measured:

- precision: 0.60954
- recall: 0.74998
- F1: 0.67251
- bbox IoU: 0.82311
- center jitter: 1.6979 px
- fusion mistakes: 60

Interpretation: precision/stability improved substantially versus prior weighted/NMM, but the NMM/weighted recall gain was not preserved.

### FAILED ATTEMPT — first threshold-grid evaluation

A first local threshold-grid script evaluated the saved pre-fusion dump without reattaching GT objects.

Observed impossible metrics:

- precision 0.0
- recall 1.0
- zero FN with tens of thousands of FP

Cause:

- pre-fusion dump does not carry GT objects needed by `evaluate_fusion`.

The result was rejected and not used for decisions.

### DONE — correct threshold-grid evaluation

Re-ran the grid after joining the frozen MOT17 GT by `(video, frame)`.

Local artifact:

`C:\Users\foto6\SpectraTrack-data\runs\a1-evidence-grid-mot17-04.json`

Selected observations:

- solo 0.22 / strong 0.35: precision 0.61033, recall 0.74927, F1 0.67270, jitter 1.6995 px;
- solo 0.20 / strong 0.35: precision 0.60954, recall 0.74998, F1 0.67251;
- solo 0.14 / strong 0.25: precision 0.57923, recall 0.76044, F1 0.65758.

Conclusion: lowering evidence thresholds trades precision back for only modest recall recovery. Threshold tuning alone does not recover the prior ~0.77 NMM recall.

### DONE — emit evidence-aware replay artifacts

Generated local canonical replay artifacts for:

- default evidence-aware thresholds;
- a deliberately looser evidence-aware setting.

Paths:

- `C:\Users\foto6\SpectraTrack-data\runs\a1-replay-mot17-04-600-evidence-default.jsonl`
- `C:\Users\foto6\SpectraTrack-data\runs\a1-replay-mot17-04-600-evidence-loose.jsonl`

### NOT DONE — held-out multi-sequence A1 validation

Still required:

- fixed MOT17 dev/validation sequence split;
- CrowdHuman dense-safety comparison for the new evidence-aware candidate;
- aggregate and per-sequence metrics;
- identical replay handoff to A2.

## 2026-09-26 — A4 target-PC runtime work

### FAILED ATTEMPT — target-probe CLI

The first Round-2 target-PC probe failed with:

`TypeError: target_scheduler_probe() missing 1 required keyword-only argument: 'source_commit'`

Cause:

- CLI parsed `--source-commit` but did not pass it to `target_scheduler_probe()`.

### DONE — fix A4 target-probe wiring

Fixed the CLI wiring in A4.

Validation:

- ruff: passed;
- branch full tests: 164 passed.

### DONE — real RX 5700 XT / DirectML execution-cost grid

Measured on real local clip:

`E:\SpectraTrack-PC\test-videos\12345.mp4`

Exact model:

`E:\SpectraTrack\yolo11x.onnx`

Model SHA-256:

`e84cbad768b218d74ecc85e3e52d84631123719a6951b3ddf6eddc850d5b3f73`

Input:

- 1920x1080;
- 25 FPS;
- 125 frames;
- input size 960;
- person threshold 0.12;
- DirectML provider.

Measured execution-cost frontier:

| detect_every | max calls | global period | calls/source-s | processing s/source-s | global bound |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 1 | 12 | 12.6 | 1.495 | 0.96 s |
| 3 | 1 | 8 | 8.4 | 0.994 | 0.96 s |
| 4 | 1 | 6 | 6.4 | 0.755 | 0.96 s |
| 4 | 2 | 6 | 12.8 | 1.496 | 0.96 s |
| 5 | 2 | 5 | 10.0 | 1.165 | 1.00 s |
| 5 | 3 | 5 | 15.0 | 1.751 | 1.00 s |

Per-call median latency was about 113–115 ms, p95 about 133–138 ms.

This is execution-cost-only evidence. It does not establish scheduler recall/quality.

### DONE — commit A4 Round-2 progress

Committed and pushed:

`agent/vnext-performance @ b3afa9d3843faa2f74b101cf2e40beea27caf6a9`

Commit:

`research(a4): measure bounded DirectML scheduler costs`

Also posted the measured frontier to issue #21.

### NOT DONE — scheduler quality gate

Still required:

- evaluate candidate cadence/call budgets against frozen quality evidence;
- quantify miss latency/new-person discovery impact;
- combine A3 strict enhancement budget with A4 scheduler budget.

## 2026-09-26 — A3 strict enhancement experiment

### DONE — prepare strict weak-evidence ROI manifest

Built a local research manifest from A1 pre-fusion weak tile evidence:

`C:\Users\foto6\SpectraTrack-data\derived\MOT17-04-first300-roi-weak1.jsonl`

Rules:

- weak evidence means tile candidate score `0.12 <= score < 0.35`;
- at most one selected ROI per source frame;
- no GT used for ROI selection;
- signal is research-only `weak_person`.

Result:

- 2400 total ROI rows;
- 1955 weak candidate ROIs existed before the per-frame cap;
- 300 frames selected exactly one weak ROI.

### IN PROGRESS / STATUS UNKNOWN AT LAST LOCAL CHECK — A3 weak-person profile

Started:

- operations: bilateral, current_adaptive_cached, sharpen;
- gate: weak-person;
- max one selected ROI per source frame by manifest construction;
- GT: MOT17-04 first 300;
- model: YOLO11x / DirectML environment.

Expected local output:

`C:\Users\foto6\SpectraTrack-data\runs\a3-round2-weak1.json`

At the last successful control-channel check the process was still running. Do not claim completion until the artifact is verified.

## 2026-09-26 — A2 failure mining runs

### DONE — A2 tooling update present and focused tests pass

A2 current Round-2 head:

`d83017d52011b1c728b6bb86b2fd519041da2032`

Focused tests:

- 13 passed.

### IN PROGRESS / STATUS UNKNOWN AT LAST LOCAL CHECK — current-tracker failure audits

Started `--audit-current` runs for the current tracker on:

- hard-NMS replay;
- conservative-NMM replay;
- weighted replay.

Expected local outputs:

- `a2-audit-hard-nms.json`
- `a2-audit-conservative-nmm.json`
- `a2-audit-weighted.json`

At the last successful control-channel check the processes were still active and output files had not yet appeared. Do not claim completion until verified.

## Current integration gate

### DONE

- public MOT17/CrowdHuman research corpora exist;
- Round-1 public evidence exists;
- A1 Round-2 evidence-aware candidate exists;
- A2 failure-mining tooling exists;
- A5 draft-seeded human review workflow exists;
- A4 real RX 5700 XT execution-cost frontier exists.

### NOT DONE / REQUIRED BEFORE PRODUCTION INTEGRATION

1. A1 held-out multi-sequence evidence-aware validation.
2. A2 targeted tracker candidate result or explicit retain-current conclusion based on mined failures.
3. A3 strict-gate result with post-fusion quality/cost conclusion.
4. A4 quality validation for low-call scheduler candidates.
5. Human-reviewed private CCTV corpus freeze: `cctv-golden-r1`.
6. Cross-role evidence stamping/leaderboard on the same private frozen revision.
7. Architect decision to unlock `agent/vnext-integrator`.

No production merge/release is authorized yet.


## 2026-09-26 — coordination rule tightened by project owner

### DONE — Git-persistent step logging made mandatory

New rule from this point forward:

**Every meaningful step, completed item, failed attempt, blocker, and still-pending item must be written into Git-tracked project files.**

Required behavior for architect and all A1-A5 roles:

- before starting a major step, ensure the intended step is represented in the role/status/task ledger;
- after the step, append the exact outcome to this worklog or the role handoff/log and update the live status file;
- failed attempts are logged with cause and whether their results were rejected;
- unfinished work is explicitly listed as NOT DONE / BLOCKED / IN PROGRESS;
- exact branch/head, tests, benchmark inputs, artifact paths/hashes, and measured results are recorded when applicable;
- local-only generated binaries/results may stay out of Git, but their existence, provenance, path/hash, and result summary must be recorded in Git;
- no important project state may exist only in chat, terminal history, local logs, or memory.

This rule is additive to the existing AGENTS/coordination rules and applies for the rest of Round 2 and later integrator work.


## 2026-09-26 — restart recovery verification

### DONE — recovered durable Round-2 state from Git

Used the Git-tracked coordination files as the source of truth after another chat/session restart.

Verified current specialist branches against the last recorded Round-2 heads:

- A1 `agent/vnext-detection @ b468836c9be1fd3669b46887a56ece4e9aafcd7f` — identical, 0 commits ahead/behind.
- A2 `agent/vnext-tracking @ d83017d52011b1c728b6bb86b2fd519041da2032` — identical, 0 commits ahead/behind.
- A3 `agent/vnext-enhancement @ 86b1460b6f9879038de92eb1e1341a96221e6c03` — identical, 0 commits ahead/behind.
- A4 `agent/vnext-performance @ b3afa9d3843faa2f74b101cf2e40beea27caf6a9` — identical, 0 commits ahead/behind.
- A5 `agent/vnext-qa @ b62509b6bbcf90e138b71fe48de51eb3c43c0787` — identical, 0 commits ahead/behind.

Conclusion: no specialist branch silently advanced during the interruption.

### DONE — verify target PC connectivity

Remote device `DESKTOP-64LCMQ8` answered a direct ping and is online.

### FAILED / UNVERIFIED — detailed process/artifact probe after restart

Several detailed Remote Desktop Commander queries for process/GPU state and the expected A2/A3 output files timed out before returning a reliable result.

Rejected behavior:

- do not infer that A2/A3 finished because the GPU became idle;
- do not infer that they failed because a control query timed out;
- do not restart those experiments until the existing local artifacts/process state can be checked without risking duplicate work.

Current state therefore remains:

- A2 audit artifacts: **NEEDS VERIFICATION**.
- A3 strict weak-person profile: **NEEDS VERIFICATION**.
- target-PC GPU/process activity: **NEEDS VERIFICATION** beyond the successful online ping.

### DONE — persistence rule reconfirmed

Project-owner instruction reconfirmed: every subsequent meaningful coordinator action, including failures and not-done work, must be persisted in Git-tracked project files before the next major step.


### DONE — propagate persistent logging rule to the Round-2 plan and all role issues

Updated `docs/VNEXT_ROUND2_PLAN.md` to make Git-persistent execution logging a formal Round-2 requirement.

Commit:

`e58dc3046df96b78d2f9c18cd06a62f5fca7277e`

The rule now requires each role to persist major-step intent/outcome, failed attempts, blockers, local artifact provenance/path/hash/summary, and not-done work before claiming readiness.

Posted the same requirement to issues #18, #19, #20, #21, #22, and #23 so the specialist assignments themselves carry the persistence contract.


## 2026-09-26 18:20 +07 — user status check

### DONE — re-read live coordination state

Re-read the current Git-tracked Round-2 status/worklog and GitHub issues #18-#23 before reporting progress.

Confirmed:

- no newer specialist commits are recorded beyond the current Round-2 heads in the status file;
- A4 target-PC execution-cost evidence remains completed and committed;
- A1 held-out multi-sequence validation is still not done;
- A2 audit outputs still require local verification after the interrupted run;
- A3 strict weak-person profile still requires local verification after the interrupted run;
- A5 private 46-frame review pack exists, but human confirmation and `cctv-golden-r1` are still not done;
- architect/integrator gate remains locked.

### FAILED / UNVERIFIED — live local artifact probe

Attempted another direct read of the expected A3 local result file through Remote Desktop Commander.

The control request timed out again.

No completion/failure claim was made for A2/A3 based on this timeout.

### NEXT

1. regain reliable local artifact/process inspection;
2. verify A2 audit outputs and A3 strict-gate result before restarting anything;
3. finish A1 held-out multi-sequence validation;
4. combine A3 quality budget with A4 scheduler quality evaluation;
5. present the private review pack for human confirmation;
6. freeze `cctv-golden-r1` only after human review;
7. only then evaluate the architect integration gate.


## 2026-09-26 — unattended execution authorized

### IN PROGRESS — autonomous Round-2 continuation on target PC

Project owner explicitly authorized continued unattended work on the connected PC while away.

Immediate execution order:

1. verify whether interrupted A2 audit and A3 strict-gate artifacts already completed;
2. do not duplicate completed work;
3. if missing and no matching process is alive, resume only the missing A2/A3 jobs;
4. complete A1 held-out multi-sequence MOT17 validation and CrowdHuman dense-safety evidence;
5. close A2 with a targeted candidate or explicit retain-current decision;
6. close A3 with strict-gated candidate or enhancement-off;
7. run A4 quality/cadence validation using the surviving A1/A3 evidence;
8. prepare A5 private review materials for later human confirmation, but do not freeze AI-only annotations;
9. keep integrator/production/release locked until all documented gates are met.

Every major step and failure will be appended to this Git worklog/status before moving on.


### FAILED / BLOCKED — Remote Desktop Commander control channel

After unattended execution was authorized, the target device still answered basic connectivity earlier, but all substantive Remote Desktop Commander operations attempted in this session timed out:

- multi-file read of the expected A2/A3 result artifacts;
- device config query;
- prior process/GPU/file probes from the recovery session.

This is treated as a control-channel blocker, not as an A2/A3 benchmark failure.

Safety response:

- no A2/A3 job was restarted blindly;
- no production branch was modified;
- no benchmark completion was fabricated;
- no local artifact was overwritten.

### NEXT while the PC control channel is unavailable

- continue Git-side verification and coordination;
- keep A1-A5 exact heads and remaining gates explicit;
- retry the target-PC control channel on the next watch cycle;
- once local control recovers, first verify existing artifacts/processes, then resume only missing jobs.


### IN PROGRESS — prepare A1 multi-sequence corpus runner while target-PC control is blocked

Git-side code review found that A1 already has canonical image/image-sequence source resolution in `vnext_detector_backend_lab.py`, while `vnext_detection_fusion.py` still only collects from a single video file.

Planned isolated A1 step:

- add corpus-aware pre-fusion collection for canonical A5 JSONL records;
- reuse the existing A1 source-resolution helpers instead of creating another source-format implementation;
- support MOT17 image sequences and CrowdHuman direct images;
- preserve existing single-video `collect` behavior;
- add tests using temporary image fixtures;
- rely on GitHub CI while the target-PC control channel is unavailable;
- after CI, use the command for held-out multi-sequence collection when local control recovers.

No production detector/runtime files are in scope.


### IN PROGRESS — A1 resumable corpus runner implemented, CI running

A1 branch advanced with research-only public-corpus tooling:

- `15d9df37ba78482da2825f9e517eef9ce71734b8` — add resumable public-corpus fusion runner;
- `6c06a87cbe3c60423370be9827c8b8117ecf832b` — add source-resolution / GT / aggregate/per-video tests;
- `fd9cd0f776d1abb86b7063731af9a77343fc37c8` — pin frame-limit settings in resumable provenance and split new vs reused inference-call accounting.

New module:

`pc/spectratrack/research/vnext_detection_corpus.py`

Purpose:

- use the canonical A5 GT JSONL plus the existing A1 source-resolution helpers;
- support MOT17 image sequences and CrowdHuman direct images;
- evaluate hard-NMS / conservative-NMM / weighted / evidence-aware on identical candidate evidence;
- optionally emit per-video pre-fusion artifacts;
- support resume only when model/GT/corpus/config/source-commit provenance matches;
- report aggregate and optional per-video metrics without changing production runtime.

GitHub Actions run `36239750252` is currently in progress for A1 HEAD `fd9cd0f...`.

Do not mark this step DONE until CI completes successfully.


### DONE — A1 resumable public-corpus runner validated

A1 research code HEAD:

`fd9cd0f776d1abb86b7063731af9a77343fc37c8`

A1 role-handoff docs commit:

`9c52142f4d8413f2c56f8b77d5dcb31df9fc2598`

GitHub Actions PC CI run:

`36239750252`

Validated result:

- ruff: passed;
- compile + pytest: **163 passed in 1.68s**;
- synthetic tracker benchmark: 1369.3 tracker FPS, crossing/reappearance ID switches 0;
- diagnostics: `DmlExecutionProvider,CPUExecutionProvider`, `directml=yes`;
- self-check: `SELF_CHECK=PASS`;
- standalone Windows build / CLI smoke / packaging / artifact upload: passed.

The new A1 runner can resume per-video prefusion evidence only under matching source/model/GT/corpus/config provenance. It supports MOT17 image sequences and CrowdHuman direct images and can compare hard-NMS / NMM / weighted / evidence-aware on the same detector evidence.

This closes the tooling gap only. The actual held-out MOT17 and CrowdHuman runs are still NOT DONE because target-PC control is currently blocked.


## 2026-09-26 — quick status check

### DONE

- Confirmed the target PC still answers Remote Desktop Commander ping.
- A1 resumable public-corpus runner remains CI-green at research code HEAD `fd9cd0f776d1abb86b7063731af9a77343fc37c8`; role handoff docs are at `9c52142f4d8413f2c56f8b77d5dcb31df9fc2598`.
- Git-side Round-2 coordination remains intact.

### BLOCKED / NOT DONE

- Detailed Remote Desktop Commander file/process reads still time out, so local A2/A3 artifact state cannot yet be safely verified.
- A1 held-out MOT17/CrowdHuman target-PC runs are still not done.
- No A2/A3 jobs are being blindly restarted while local artifact/process state is unverified.
- Production integration remains locked.


## 2026-09-26 — continue Round 2 while target-PC control is degraded

### IN PROGRESS — A2 targeted-association candidate design from Git source

Because target-PC artifact/process inspection is still unreliable, the next safe step is Git-side A2 research that does not require rerunning detector inference.

Plan:

1. inspect current `agent/vnext-tracking` failure-miner/candidate code and tests;
2. identify the smallest targeted ambiguity/hysteresis change that preserves current two-stage high/low semantics;
3. add it only to research tooling, not production `tracker.py`;
4. add deterministic tests for the specific close/competing-association case;
5. push to A2 and use GitHub CI for code validation;
6. defer real replay acceptance/rejection until target-PC artifacts are again readable.

No production branch or runtime behavior will be changed.


### IN PROGRESS — A2 ambiguity-scoped assignment candidate implemented

A2 research branch advanced:

- code commit: `662da384551c9573016da41403019340f6022252`
- test commit / current head: `711ed1c476d26db7beca99a3d049ec4741202a8d`

New research-only candidate:

`current-ambiguity-guard`

Design:

- keeps current SpectraTrack two-stage high/low association, gates, lifecycle, CMC, appearance, dormant recovery, and strong-only creation;
- preserves current greedy association when no close score competition exists;
- detects connected ambiguous association components where top alternatives are within the existing research margin (0.08);
- applies global maximum-score assignment only inside those ambiguous components;
- falls back to current greedy ordering for the remaining edges.

This specifically targets the observed failure pattern without adopting wholesale global assignment.

Added deterministic tests:

- nearby-same-class synthetic greedy conflict must be resolved with 0 ID switches;
- non-ambiguous camera-pan / weak-detection / dormant-reactivation scenarios must remain metric-identical to current tracker.

GitHub Actions run:

`36240587164`

State: **IN PROGRESS**. Do not mark candidate validated until CI completes.


### FAILED — first A2 ambiguity-guard implementation did not trigger on the synthetic conflict

GitHub Actions run `36240587164` failed one deterministic test:

`test_global_assignment_probe_exposes_current_greedy_conflict`

Observed:

- current tracker: known greedy conflict;
- full global candidate: 0 switches;
- first ambiguity-guard candidate: **4 switches**, so it did not improve the target case.

Cause:

- the first component builder only connected score-close edges;
- in the known 2x2 conflict, one detection had two near-equal track scores, but the alternative second detection was not itself within the 0.08 margin;
- the candidate therefore saw a 2-track/1-detection component and never invoked the local global solve.

Rejected result: the failed implementation is not evidence of candidate quality.

### IN PROGRESS — A2 ambiguity component fix

A2 advanced to:

`c552b0f45e033628e188f91df11839025c2ed359`

Fix:

- ambiguity is still seeded only by close-score competition;
- component context now expands through each participating node's two best accepted alternatives;
- this supplies the missing one-to-one alternative without globally replacing assignment on every frame.

GitHub Actions run `36240708158`:

- lint: passed;
- compile + pytest: passed;
- synthetic benchmark: passed;
- diagnostics/self-check: passed;
- Windows build was still running at the last check.

Do not mark fully validated until the workflow concludes successfully.


### DONE — A2 ambiguity-scoped candidate CI validation

Final A2 role-handoff head:

`8c4c9b69f9d3f332ed2d655ff5f7bc9cb69cb6e4`

Research code head:

`c552b0f45e033628e188f91df11839025c2ed359`

GitHub Actions run `36240708158` completed successfully:

- ruff passed;
- compile + pytest: **166 passed in 2.09 s**;
- tracker benchmark: 1643.4 tracker FPS on the synthetic smoke;
- crossing/reappearance smoke ID switches: 0;
- diagnostics and self-check passed;
- Windows standalone build/smoke/package/upload passed.

The candidate is now mechanism-valid but remains **NOT YET ACCEPTED** for production. Real replay comparison is still required once target-PC artifact access is reliable.


### IN PROGRESS — make A3 strict enhancement budget reproducible in tooling

The existing local weak-person experiment enforced one selected ROI per source frame by preparing a special manifest. That is useful evidence, but the budget is not yet encoded in the A3 profiler itself.

Safe Git-side step:

- add an explicit research-only per-frame enhanced-ROI cap to the A3 profiler;
- default to unlimited so existing behavior/tests remain backward-compatible;
- enforce the cap independently per operation after operation/selective gates;
- record cap and budget-skipped ROI counts in result provenance;
- add deterministic tests proving that two eligible ROIs in one frame produce only one enhanced call when the cap is 1;
- do not change production enhancement/runtime code.

This makes the Round-2 strict budget reproducible instead of depending on a hand-crafted manifest.


### DONE — A3 explicit strict enhancement budget is CI-green

A3 role-handoff head:

`a5481b1f4bbcc67e77b5eecefb3e8daea783ba5c`

Research code head:

`898caa9c6ecff4d15b0d5cbabaae9390b8e86fd8`

GitHub Actions run `36241047867`:

- result: SUCCESS;
- ruff passed;
- compile + pytest: **154 passed in 2.66 s**;
- synthetic tracker smoke: 1798.9 tracker FPS;
- crossing/reappearance smoke switches: 0;
- diagnostics/self-check/build/package/upload passed.

The profiler now enforces a reproducible per-operation/source-frame enhanced-inference cap and records budget-skipped ROIs. The quality decision remains pending target-PC evidence.


### IN PROGRESS — freeze A1 Round-2 MOT17 development/held-out split

A1 already supports exact repeated `--include-video` selection. To prevent accidental held-out leakage, the sequence split itself will now be committed as immutable research metadata before the target-PC run.

Split policy:

- development: `MOT17-04` only, because Round-2 fusion thresholds/policy were already developed on its first 600 frames;
- held-out validation: `MOT17-02`, `MOT17-05`, `MOT17-09`, `MOT17-10`, `MOT17-11`, `MOT17-13`;
- CrowdHuman validation remains a separate dense detection-safety corpus and is not used for MOT17 tuning.

No held-out result may be used to retune the policy in this Round-2 cycle.


### DONE — A1 fixed held-out split committed before validation

A1 branch head:

`002b698b06c439694906c9c5b3e765e543bd0029`

Added:

`pc/benchmarks/vnext/detection/round2_mot17_split.json`

Frozen policy:

- development: MOT17-04 only;
- held-out: MOT17-02/05/09/10/11/13;
- CrowdHuman validation remains separate dense-safety evidence.

The A1 README now contains the exact resumable held-out command. Held-out metrics may not be used to silently retune Round-2 thresholds. Actual target-PC inference remains NOT DONE while the command relay is timing out.


### IN PROGRESS — make A1 held-out runner emit canonical A2 replays

The resumable A1 corpus runner currently records metrics and per-video prefusion artifacts, but the Round-2 handoff requires surviving policies to be passed to A2 as canonical replay bytes.

Safe research-only change:

- add optional `--replay-dir`;
- require `--prefusion-dir` when replay export is requested;
- after evaluation, convert each selected per-video prefusion artifact into canonical `spectratrack-detection-replay-v1` for every requested fusion method;
- record generated replay paths in the corpus report;
- add deterministic tests;
- do not rerun detector inference for replay conversion;
- do not change production detector behavior.


### FAILED then FIXED — A1 canonical replay export CI lint

GitHub Actions run `36241720488` failed at lint before tests because the new replay-export test used `json.loads` without importing `json`.

Observed failure:

- `F821 Undefined name json`
- file: `pc/tests/test_vnext_detection_corpus.py`

Fix commit on A1:

`704c9a645d276ceb1d2acd26f0dc44ad296196ce`

The failure is retained here rather than hidden. Awaiting the replacement CI run before marking replay-export tooling validated.


## 2026-09-26 19:20 +07 — parallel specialist dispatch

### DONE — dispatch A1-A5 Round-2 roles in parallel

Posted explicit START instructions to GitHub issues:

- #18 A1 — held-out MOT17 + CrowdHuman evidence-aware fusion validation;
- #19 A2 — real identical-replay ambiguity-guard vs current tracker evaluation;
- #20 A3 — strict weak-evidence enhancement profile and enhancement-off decision;
- #21 A4 — scheduler quality/cadence validation with miss/discovery latency;
- #22 A5 — finish review-progress tooling/tests/docs and keep private review pack ready.

All roles retain the mandatory Git-persistent logging rule and the no-integration/main/RC boundary.

### BLOCKED / UNVERIFIED — actual local Codex worker launch

Attempted to query the target PC with `codex --version` before starting local autonomous workers.

Remote Desktop Commander timed out again, so actual Codex worker availability/limits are currently unverified.

No claim is made that Codex processes were launched. Git tasks are dispatched; local worker launch will be attempted only after the control channel responds.


## 2026-09-26 19:30 +07 — coordinator recovery and verification

### DONE — re-verify immutable production refs and specialist heads

After re-reading the Round-2 coordination source of truth, verified GitHub branch state:

- `integration @ 3ebc4d50213593cac62b97399447fccf6bbc1755` — unchanged;
- `main @ 2012eaae2f4ffe820a66d12e40346d911616cd03` — unchanged;
- A1 code HEAD `704c9a645d276ceb1d2acd26f0dc44ad296196ce` matched the expected branch state before the coordinator documentation commit;
- A2 `8c4c9b69f9d3f332ed2d655ff5f7bc9cb69cb6e4` — exact match;
- A3 `a5481b1f4bbcc67e77b5eecefb3e8daea783ba5c` — exact match;
- A4 `b3afa9d3843faa2f74b101cf2e40beea27caf6a9` — exact match;
- A5 had advanced two research commits beyond the previously recorded `b62509b...` and was recovered at code/test HEAD `afc6baf1cc0216ec21fa71f298be88e76519cdfa`.

No production branch was changed.

### DONE — A1 replacement CI after replay-export lint fix

A1 commit:

`704c9a645d276ceb1d2acd26f0dc44ad296196ce`

GitHub Actions:

`36241884895` — **SUCCESS**

The earlier `F821 Undefined name json` failure remains recorded. The fixed replay-export tooling is now CI-validated. Held-out MOT17/CrowdHuman evidence remains **NOT DONE** and must not be retuned on held-out data.

A1 role documentation was updated in commit:

`32ae82052b6de37ae938e2e2cded27c9787118d5`

A docs-only CI run `36242653285` started for that head and was still in progress at this checkpoint.

### DONE — recover A5 private-review progress work

Recovered A5 research code/test HEAD:

`afc6baf1cc0216ec21fa71f298be88e76519cdfa`

GitHub Actions:

`36241763043` — **SUCCESS**

Verified behavior:

- human-review progress is based only on rows saved to the human output GT;
- draft preannotations never count as reviewed;
- pending frames are separated into with-draft vs without-draft;
- duplicate review-batch frame keys are rejected;
- `cctv-golden-r1` remains **NOT DONE** until explicit human confirmation.

A5 role documentation was updated in commit:

`8faf6c9180c5d84555589e2ed7600832b036ac78`

A docs-only CI run `36242654534` started for that head and was still in progress at this checkpoint.

### FAILED / BLOCKED — target-PC substantive control remains unavailable

The target PC still answered a direct ping, but substantive Remote Desktop Commander calls timed out again, including:

- `git fetch origin --prune` / repository status command;
- process/session enumeration attempt;
- direct file metadata query for `C:\\Users\\foto6\\SpectraTrack-data\\runs\\a3-round2-weak1.json`.

Interpretation:

- this is a control-channel failure, not benchmark evidence;
- A2/A3 completion state remains **NEEDS VERIFICATION**;
- A1 held-out run is not started blindly because existing local process/artifact state cannot be positively checked;
- no local artifact was overwritten or duplicated.

### NEXT

1. wait for/verify the docs-only A1/A5 CI runs;
2. retry a minimal target-PC process/artifact probe on a later control cycle;
3. if control recovers, verify process/artifact/marker/timestamp before any restart;
4. then run A1 held-out without retune and evaluate A2/A3 from existing or resumed artifacts;
5. A4 remains parked until surviving quality policies exist;
6. stop for the project owner only when the compact private CCTV pack is ready for human review or a non-recoverable blocker remains.


## 2026-09-26 19:40 +07 — coordinator stop gate reached

### DONE — docs-only specialist CI completed

A1 documentation head:

`32ae82052b6de37ae938e2e2cded27c9787118d5`

GitHub Actions run `36242653285`: **SUCCESS**.

A5 documentation head:

`8faf6c9180c5d84555589e2ed7600832b036ac78`

GitHub Actions run `36242654534`: **SUCCESS**.

These runs do not replace the already recorded code/evidence CI; they confirm the current branch heads remain green after the mandatory state-documentation updates.

### FAILED / BLOCKED — control channel remains unusable

A later Remote Desktop Commander meta/history probe also timed out. Combined with the earlier process/session/file-info timeouts, the target PC can currently be pinged but cannot be safely inspected or commanded for substantive work.

Safety decision:

- do not restart A1 held-out, A2 replay, or A3 strict profile blindly;
- keep their local completion state at **NEEDS VERIFICATION / NOT DONE** as applicable;
- do not invent target-PC artifacts or benchmark results.

### BLOCKED — human private CCTV review gate reached

A5 already prepared the compact 46-frame private review pack from the three user CCTV clips plus machine draft preannotations/triage. This is the point where project-owner human confirmation is explicitly required.

`cctv-golden-r1` remains **NOT DONE** and must not be frozen until the human-reviewed output exists.

Coordinator stop condition reached for two independent reasons:

1. human review is now required for the private CCTV pack;
2. the target-PC substantive control channel is persistently unavailable and prevents safe verification/resumption of long local jobs.

Production integration, `agent/vnext-integrator`, merge to `integration/main`, RC changes and release all remain locked.


## 2026-09-26 20:17 +07 — continuation after owner authorization

### DONE — retry minimum target-PC control probe

Target PC `DESKTOP-64LCMQ8` answered ping successfully.

A minimal substantive PowerShell probe containing only:

`Write-Output 'ok'; Get-Date`

still timed out through Remote Desktop Commander.

This confirms the current blocker is not caused by a long Git/benchmark command. The control channel can resolve ping while failing even trivial substantive execution.

### BLOCKED — target-PC experiment continuation remains unsafe

Because the minimum command failed, no A1/A2/A3 local job was restarted and no artifact was overwritten. Existing local A2/A3 results remain **NEEDS VERIFICATION**; A1 held-out remains **NOT DONE**.

### DONE — GitHub specialist issue review

Reviewed issues #18-#23 after the retry. No newer A1/A2/A3/A4 benchmark result had been posted beyond the already recorded states. A1/A2/A3/A5 assignments remain active; A4 remains deferred until surviving quality policies are available.

### BLOCKED — private CCTV human review is now the actionable non-compute gate

A5's 46-frame private review pack and machine draft preannotations are prepared. Human confirmation/correction is required before any `cctv-golden-r1` freeze. This is intentionally not bypassed by AI review.


## 2026-09-26 — public-first benchmark strategy revision

### DONE — owner changed benchmark priority

The project owner explicitly moved private CCTV out of the primary benchmark role. Public human-annotated datasets now carry the main quantitative evidence. The current 46-frame private review pack is retained but human review is deferred until public finalists exist.

### DONE — official dataset/terms research

DanceTrack official repository findings:

- public train and validation annotations provide bbox + stable identity in MOT-style `gt.txt`;
- annotations are CC BY 4.0;
- image/video dataset use is non-commercial research only;
- code is MIT;
- result: **ACCEPTED for isolated research import**, no raw dataset bytes in Git/product release.

NightOwls official dataset findings:

- night pedestrian dataset, 279k frames / 40 sequences;
- PNG/JSON and Caltech-compatible distributions;
- annotations include pedestrian boxes plus occlusion/difficulty/pose and tracking information;
- official license allows academic/non-academic non-commercial research/personal experimentation, requires citation, and prohibits redistribution of the dataset or modified versions;
- result: **ACCEPTED as primary isolated night research corpus**, no raw/modified dataset redistribution.

LLVIP official repository findings:

- visible + infrared paired low-light dataset;
- non-commercial research/teaching/personal experimentation license;
- result: **ACCEPTED as optional secondary research evidence using visible/RGB side only**; infrared is not SpectraTrack production input.

KAIST:

- visible + thermal pedestrian benchmark with dense manual annotations and temporal correspondence;
- retained as **fallback**, not primary Round-2 night corpus, to avoid unnecessary multispectral-origin complexity; visible/RGB only if used.

### DONE — architectural dataset decision

Round-2 public evidence contract is now:

- MOT17 -> A1 detection/fusion + A2 tracking;
- CrowdHuman -> A1 dense/crowd/fusion safety, no tracking metrics;
- DanceTrack -> A2 association/ID/crossing stress, association-only first;
- NightOwls -> A1 night detection/fusion + A3 enhancement; A2 tracking only if importer validates stable IDs;
- LLVIP visible -> optional secondary low-light detector/enhancement evidence;
- private CCTV -> final reduced domain sanity check only.

### IN PROGRESS — specialist dispatch update

A5 is assigned DanceTrack + NightOwls importer/freeze work. A2 is assigned DanceTrack association-only after A5 freeze. A1/A3 are assigned NightOwls validation after A5 freeze. A4 remains parked until public quality finalists exist.

The target-PC substantive control channel remains blocked; no local long-running job is restarted blindly.


### DONE — public-first role directives committed

Role handoff commits:

- A1 detection: `bccf087df01b74bc463dcc3525e5665110ff8225` — NightOwls public validation assignment;
- A2 tracking: `35de7cbdb4023c66cfdba27698bd8b62dd106e78` — DanceTrack association-only gate;
- A3 enhancement: `4b13f8eaec4e9339ea3247fc759f85400e1a13b4` — NightOwls strict enhancement gate;
- A4 performance: `45bd0bc85f7e677cd602b435108b2095ab0ee78b` — explicitly parked until public quality finalists;
- A5 QA: `abc269f5ebe062c80d1fba8eb5c06b1b1450c6b9` — DanceTrack + NightOwls isolated importer/freeze assignment.

Issues #18-#23 were updated with the same role boundaries and persistence requirements. No production branch or specialist branch was merged into another.


### DONE — held-out split / leakage rules for new public corpora

Pinned before candidate evaluation:

- DanceTrack validation is the primary held-out association gate; no current-ambiguity-guard retuning from DanceTrack validation results.
- NightOwls validation is the primary held-out night gate; no A1/A3 threshold/gate retuning after candidate results are observed.
- A deterministic stratified NightOwls Round-2 slice is permitted for compute control only if frozen before candidate results and sampled solely from source sequence/official annotation metadata, not model output.
- Any slice must preserve positive and negative/background evidence, use identical bytes for every candidate, and be reported as a slice rather than a full-validation result.

This prevents compute-driven cherry-picking while keeping YOLO11x/960 multi-pass evaluation tractable.


### BLOCKED — target-PC substantive relay still unavailable after public-first dispatch

Target `DESKTOP-64LCMQ8` still answers ping, but a minimal substantive PowerShell probe:

`Write-Output 'probe-ok'`

timed out again.

Safety decision unchanged:

- do not restart A1 held-out inference blindly;
- do not restart A2/A3 interrupted jobs blindly;
- do not infer completion/failure from GPU idleness or relay timeout;
- continue Git-side public dataset coordination only until process/artifact inspection works.


### DONE — unified public evidence contract committed

Added:

`docs/VNEXT_ROUND2_PUBLIC_EVIDENCE.md`

Commit:

`727a259d1ceaff5fe33bb4138514a3ddc0eeee2a`

The contract pins:

- corpus purpose/roles;
- official source and research-use constraints;
- DanceTrack association-only input semantics;
- NightOwls held-out and deterministic-slice policy;
- LLVIP visible-only and KAIST visible-only fallback rules;
- canonical import/freeze provenance;
- required A1-A4 metrics;
- private CCTV deferred to the final reduced domain sanity pack.


### DONE — hourly coordinator watch enabled

An hourly condition watch is enabled for Round-2 coordination.

Each cycle is instructed to:

- recover state from `coord/vnext-round2`;
- verify specialist heads/CI/issues and immutable production refs;
- test target-PC substantive control safely;
- verify process/artifact/marker/log/timestamps before any restart;
- continue only research/coordination work within A1-A5 ownership boundaries;
- persist every meaningful result/failure/blocker into Git;
- keep `main`, `integration`, RC and release untouched;
- notify the project owner only when human action is required, a blocker has no autonomous workaround, or the architect gate is complete.

This provides restart-safe ongoing coordination without treating chat memory as project state.


## 2026-09-26 — manual specialist chat launch confirmed

### DONE — owner launched A1/A2/A3/A5 chats manually

The project owner confirmed the current Round-2 prompts were sent to the A1, A2, A3 and A5 agent chats after Codex limits prevented assuming autonomous worker startup.

At this checkpoint, Git verification showed no new specialist commits yet:

- A1 `bccf087df01b74bc463dcc3525e5665110ff8225`
- A2 `372c96e71e504a54ea2ac027ab988dfba2bd1918`
- A3 `4b13f8eaec4e9339ea3247fc759f85400e1a13b4`
- A5 `f4f406182df54335e2c26a34091915408204789c`

All four current heads are CI-green.

Production refs re-verified unchanged:

- `integration @ 3ebc4d50213593cac62b97399447fccf6bbc1755`
- `main @ 2012eaae2f4ffe820a66d12e40346d911616cd03`

Next architect action is to inspect every new specialist head as it appears: ownership/diff -> CI/tests -> artifact/provenance -> metrics/gates -> role/status/worklog update. No specialist feature coding is delegated to the architect unless a coordination-only fix is required.


## 2026-09-26 — target-PC control restored and interrupted artifact verification

### DONE — substantive target-PC control restored

Device `DESKTOP-64LCMQ8` is online through Remote Desktop Commander. Verified:

- device status: online;
- ping: success;
- substantive PowerShell command: success.

This clears the prior relay-only blocker.

### FAILED — first recovery inspection command syntax

The first read-only PowerShell inspection command failed with a parser error caused by piping directly from a `foreach` statement.

No benchmark process was launched, no artifact was changed, and no repository state was modified.

The failure was corrected with a read-only object collection command.

### DONE — process verification before restart

No active SpectraTrack benchmark Python process was found.

Observed Python processes were VS Code Black formatter services plus:

- `C:\Users\foto6\SpectraTrack-control\supervisor.py run`

No A1/A2/A3 benchmark command line was active.

Therefore old interrupted results could be inspected safely without assuming a job was still running.

### DONE — A2 old audit artifacts recovered

Verified existing local artifacts:

- `C:\Users\foto6\SpectraTrack-data\runs\a2-audit-hard-nms.json`
  - 836,448,758 bytes
  - modified 2026-09-26 09:50:26 UTC
  - replay canonical SHA-256 `d27d45172a2df0a5c81ff83be2ceed03d2ed1eef922d3fde06f72b3126a89c59`
  - replay source SHA-256 `b2719a2c93c353123437497ac9513033898d3d65750a32ed633d05e8fb65b2d4`
  - ONNX calls 0
  - wall 19.1857 s
- `a2-audit-conservative-nmm.json`
  - 945,546,052 bytes
  - modified 2026-09-26 09:50:55 UTC
  - canonical replay SHA-256 `a34f67210de56ca63ab03060876116eb3ab026376a58ce3d6f38dfd2f19785b6`
  - replay source SHA-256 `ffcd3f5a1812afc83f392e4410a63e2cef3e5bab3491455a826ee65fac25cea1`
  - ONNX calls 0
  - wall 22.9854 s
- `a2-audit-weighted.json`
  - 953,219,839 bytes
  - modified 2026-09-26 09:51:00 UTC
  - canonical replay SHA-256 `d796c01a22fc93e8510db6420fd5799cfe92c69ae14e83a061e5eb41a1abd93d`
  - replay source SHA-256 `91a3b4cedde639e04478a06be5a57697aa22651a7a02d61e87685b712eb7d007`
  - ONNX calls 0
  - wall 25.7062 s

These are valid recovered old current-tracker audit artifacts, but they do **not** close Round-2 A2 because they do not contain the required `current-ambiguity-guard` real-replay comparison.

### DONE — A3 strict weak-person artifact recovered

Verified:

`C:\Users\foto6\SpectraTrack-data\runs\a3-round2-weak1.json`

- 8,933 bytes;
- modified 2026-09-26 09:56:06 UTC;
- schema `spectratrack-vnext-enhancement-profile-v1`;
- corpus `mot17-public-r1`;
- source `MOT17-04-first300.mp4`;
- source commit `86b1460b6f9879038de92eb1e1341a96221e6c03`;
- model SHA-256 `e84cbad768b218d74ecc85e3e52d84631123719a6951b3ddf6eddc850d5b3f73`;
- providers DML + CPU fallback;
- selective gate `weak-person`;
- raw corroboration required;
- 2,400 raw probe calls;
- actual low-level ONNX calls across shared experiment: 2,726;
- sampled source seconds: 10;
- wall: 437.343 s.

Recovered candidate deltas:

- bilateral:
  - 30 enhanced calls;
  - 15 recovered GT;
  - +372 pre-fusion FP observations;
  - ~0.500 recovered GT / extra call;
  - ~24.8 FP observations / recovered GT;
  - 34.906 processing s / sampled source s.
- current_adaptive_cached:
  - 158 enhanced calls;
  - 101 recovered GT;
  - +2,170 pre-fusion FP observations;
  - ~0.639 recovered GT / extra call;
  - ~21.49 FP observations / recovered GT;
  - 36.747 processing s / sampled source s;
  - bbox center/size residual jitter worsened vs raw.
- sharpen:
  - 138 enhanced calls;
  - 87 recovered GT;
  - +1,890 pre-fusion FP observations;
  - ~0.630 recovered GT / extra call;
  - ~21.72 FP observations / recovered GT;
  - 36.409 processing s / sampled source s;
  - bbox center/size residual jitter worsened vs raw.

A3 status changes from `NEEDS VERIFICATION` to **artifact verified** for this MOT17 strict-profile run.

However this artifact reports **pre-fusion** FP deltas and A3 explicitly does not own final fusion. It therefore cannot by itself choose enhancement ON/OFF. Post-fusion quality plus NightOwls held-out evidence remain required.
