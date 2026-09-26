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
- **DONE** — A1 resumable canonical public-corpus runner implemented and CI-validated.
- **NOT DONE** — A1 held-out multi-sequence Round-2 validation (target-PC run still required).
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

`agent/vnext-detection @ 002b698b06c439694906c9c5b3e765e543bd0029`

Done:

- evidence-aware weak-person fusion candidate exists;
- default and threshold-grid MOT17-04 development evidence recorded;
- canonical evidence-aware replay artifacts emitted locally;
- resumable public-corpus runner implemented and CI-validated;
- fixed Round-2 MOT17 split committed before held-out evaluation:
  - development: MOT17-04;
  - held out: MOT17-02/05/09/10/11/13;
- CrowdHuman kept separate as dense detection-safety validation;
- exact resumable held-out command documented.

Not done:

- target-PC held-out per-sequence/aggregate validation;
- CrowdHuman dense-safety validation for the new candidate;
- final Round-2 A1 accept/reject handoff.

## A2 Tracking — issue #19

Current known Round-2 head:

`agent/vnext-tracking @ 8c4c9b69f9d3f332ed2d655ff5f7bc9cb69cb6e4`

Done:

- current tracker remains control;
- failure-mining tooling added;
- ambiguity-scoped assignment candidate implemented;
- first too-narrow ambiguity implementation failed CI and is documented;
- fixed candidate passes the nearby-same-class mechanism probe with 0 ID switches;
- selected non-ambiguous synthetic probes remain metric-identical to current;
- GitHub CI run `36240708158`: success, 166 tests passed, build/package green.

In progress / needs verification:

- current-tracker failure-audit artifact state on the target PC after the interrupted run;
- real identical-replay evaluation of `current-ambiguity-guard`.

Not done:

- Round-2 gate decision on real replay (recall / IDSW / fragmentation / false tracks);
- validation against the surviving A1 Round-2 replay;
- final retain-current vs targeted-fix handoff.

## A3 Enhancement — issue #20

Current known Round-2 head:

`agent/vnext-enhancement @ a5481b1f4bbcc67e77b5eecefb3e8daea783ba5c`

Done:

- strict weak-evidence ROI manifest prepared locally;
- research profiler now has explicit `--max-enhanced-rois-per-frame` budget support;
- cap=1 semantics are covered by deterministic tests;
- raw probes remain separately counted and budget-skipped follow-ups are reported;
- GitHub CI run `36241047867`: success, 154 tests passed, build/package green.

In progress / needs verification:

- target-PC weak-person strict profile for bilateral / current_adaptive_cached / sharpen after the interrupted run.

Not done:

- verified strict-gate metrics on target evidence;
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


## Current execution blocker

- **BLOCKED (control channel)** — target PC Remote Desktop Commander substantive operations are timing out.
- This does **not** mean A2/A3 benchmarks failed.
- Do not restart A2/A3 until existing local process/artifact state is verified.
- GitHub coordination, branch verification, and documentation remain available.
