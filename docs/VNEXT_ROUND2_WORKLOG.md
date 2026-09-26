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
