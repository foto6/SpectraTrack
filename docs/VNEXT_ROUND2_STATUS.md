# SpectraTrack vNext Round 2 — Live Status

This file is the current coordination snapshot. Update it whenever a task changes state.

Status keywords:

- **DONE** — completed and evidence recorded.
- **IN PROGRESS** — active work started; completion not yet verified.
- **BLOCKED** — cannot safely continue without resolving a concrete blocker.
- **NOT DONE** — required work has not yet been completed.
- **LOCKED** — intentionally prohibited until prerequisites are satisfied.

## Mandatory persistence rule

Every meaningful step and every completed, failed, blocked, in-progress, or not-done item must be represented in Git-tracked project files. Important state must not exist only in chat, local logs, or terminal history.

## Global gates

- **DONE** — public `mot17-public-r1` frozen.
- **DONE** — public `crowdhuman-val-fbox-r1` frozen.
- **DONE** — Round-1 detector/fusion/tracking/enhancement/scheduler evidence collected.
- **DONE** — Round-2 evidence review and assignments created.
- **NOT DONE** — A1 held-out multi-sequence Round-2 validation.
- **IN PROGRESS / needs verification** — A2 failure-audit runs and targeted candidate work.
- **IN PROGRESS / needs verification** — A3 strict weak-evidence enhancement profile.
- **DONE** — A4 real RX 5700 XT execution-cost frontier.
- **DONE** — A5 draft-seeded private review tooling and 46-frame review pack preparation.
- **NOT DONE** — human-confirmed private CCTV annotations.
- **NOT DONE** — `cctv-golden-r1` freeze.
- **NOT DONE** — final cross-role candidate table on `cctv-golden-r1`.
- **LOCKED** — `agent/vnext-integrator`.
- **LOCKED** — production merge/release.

## A1 Detection / Fusion — issue #18

Current known Round-2 head:

`agent/vnext-detection @ b468836c9be1fd3669b46887a56ece4e9aafcd7f`

Done:

- evidence-aware weak-person fusion candidate exists;
- focused tests passed;
- default and threshold-grid MOT17-04 development evidence recorded;
- canonical evidence-aware replay artifacts emitted locally.

Not done:

- fixed multi-sequence MOT17 dev/validation split;
- held-out per-sequence/aggregate validation;
- CrowdHuman dense-safety validation for the new candidate;
- final Round-2 A1 accept/reject handoff.

## A2 Tracking — issue #19

Current known Round-2 head:

`agent/vnext-tracking @ d83017d52011b1c728b6bb86b2fd519041da2032`

Done:

- current tracker remains control;
- failure-mining tooling added;
- focused tests passed.

In progress / needs verification:

- current-tracker failure audits on hard-NMS, conservative-NMM, and weighted replay.

Not done:

- targeted ambiguity/hysteresis or occlusion candidate conclusion;
- validation against surviving A1 Round-2 replay;
- final retain-current vs targeted-fix handoff.

## A3 Enhancement — issue #20

Current known base/head before further Round-2 changes:

`agent/vnext-enhancement @ 86b1460b6f9879038de92eb1e1341a96221e6c03`

Done:

- strict weak-evidence ROI manifest prepared locally;
- one selected weak ROI maximum per source frame in the current experiment design.

In progress / needs verification:

- weak-person strict profile for bilateral / current_adaptive_cached / sharpen.

Not done:

- verified strict-gate metrics;
- post-fusion quality/cost decision;
- enhancement-off vs strict-gated final handoff.

## A4 Performance — issue #21

Current known Round-2 head:

`agent/vnext-performance @ b3afa9d3843faa2f74b101cf2e40beea27caf6a9`

Done:

- target-probe CLI blocker found and fixed;
- branch full tests: 164 passed;
- real RX 5700 XT / DirectML execution-cost frontier measured;
- source commit/model/provider/provenance recorded.

Not done:

- scheduler quality evaluation;
- new-person discovery/miss-latency evidence;
- combined A3 strict-enhancement + A4 scheduler budget conclusion.

## A5 QA — issue #22

Current known Round-2 head:

`agent/vnext-qa @ b62509b6bbcf90e138b71fe48de51eb3c43c0787`

Done:

- machine draft boxes can seed human review while human-saved rows take precedence;
- 46-frame private CCTV review pack extracted from the three local test videos;
- YOLO11x/DirectML draft pre-annotations and triage artifacts produced;
- private AI drafts remain explicitly non-GOLDEN.

Not done:

- human confirmation/correction;
- `cctv-golden-r1` freeze;
- final A1-A4 rerun/stamp table on private GOLDEN.

## Architect / Integrator — issue #23

State: **LOCKED**

Unlock only when:

1. A1 has held-out evidence for a surviving fusion policy.
2. A2 has a targeted improvement or explicit retain-current conclusion.
3. A3 has a verified strict-enhancement decision.
4. A4 has quality-validated target-PC scheduler evidence.
5. A5 has human-confirmed `cctv-golden-r1`.
6. Surviving candidates are compared on that same frozen private revision.


## Restart recovery note

- **DONE** — GitHub branch heads re-verified; none of A1-A5 advanced beyond the heads recorded above during the interruption.
- **DONE** — target PC answered a direct connectivity ping.
- **NEEDS VERIFICATION** — detailed target-PC process/GPU state; control queries timed out.
- **NEEDS VERIFICATION** — A2 audit output files after the interrupted run.
- **NEEDS VERIFICATION** — A3 `a3-round2-weak1.json` completion after the interrupted run.
- **RULE** — do not restart A2/A3 blindly until existing local process/artifact state is verified.
