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

- **DONE** — A1 MOT17 held-out complete. Evidence-aware is the sole retained fusion challenger: F1 0.527449 vs hard-NMS 0.516517, precision 0.431776 vs 0.418094, recall 0.677588 vs 0.675549, FP 57,727 vs 60,868, fusion mistakes 540 vs 1016.
- **IN PROGRESS** — A1 CrowdHuman dense-safety full validation is actively running on target PC. Latest live check: 2,520 / 4,370 image-level prefusion artifacts persisted (~57.7%); worker CPU is increasing; final result/marker not yet present.
- **DONE** — A2 Round-2 decision: RETAIN CURRENT TRACKER. current-ambiguity-guard rejected by real identical-replay ID-switch gate.
- **WAITING** — A3 strict candidate set remains OFF / bilateral / current_adaptive_cached; NightOwls held-out waits for A5 frozen corpus/slice.
- **PARKED** — A4 scheduler quality search remains intentionally deferred until public quality finalists are ready.
- **DONE** — DanceTrack frozen as `dancetrack-public-r1`, corpus SHA `df240532ad3f2099947f318b682737ccdb3e6345dc9da1e380ff4cab8d6c6b5c`; 25 sequences / 25,508 frames / 225,148 people; validation clean.
- **IN PROGRESS** — A5 official NightOwls intake is active on isolated `E:\SpectraTrack-data\public\NightOwls`. Official JSON and SDK are already present; official validation ZIP is actively downloading. Latest live check: 5,866,102,784 / 57,481,286,834 bytes (~10.2%).
- **DONE** — NightOwls storage blocker cleared; E: had 210,024,607,744 bytes free before intake and still has >200 GB free during the current download.
- **NOT DONE** — NightOwls import/validation/freeze, A1/A3 NightOwls held-out, A4 final 1-3 config quality/cost check, reduced private CCTV human-confirmed sanity pack, final cross-role candidate table.
- **LOCKED** — `agent/vnext-integrator`, production merge/release.

## A1 Detection / Fusion — issue #18

Current known Round-2 head:

`agent/vnext-detection @ 043931fb0bda2a10716ca19d2381adbd56c8d59b`

Done:

- evidence-aware weak-person fusion candidate exists;
- default and threshold-grid MOT17-04 development evidence recorded;
- canonical evidence-aware replay artifacts emitted locally;
- resumable public-corpus runner implemented and CI-validated;
- canonical held-out replay-export tooling fix `704c9a6...` validated by CI run `36241884895` (success);
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

`agent/vnext-tracking @ d1470454129024d3767b6cb91a7d6ba66102d52d`

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

`agent/vnext-enhancement @ 21e7b06dc470a346dd01e86b45e3e581e4f0d59d`

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

`agent/vnext-performance @ 45bd0bc85f7e677cd602b435108b2095ab0ee78b`

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

`agent/vnext-qa @ 6760ac740b9875f2d0754e3cf5870a41b2f52a0e`

Done:

- machine draft boxes can seed human review while human-saved rows take precedence;
- 46-frame private CCTV review pack extracted from the three local test videos;
- YOLO11x/DirectML draft pre-annotations and triage artifacts produced;
- private AI drafts remain explicitly non-GOLDEN;
- review-progress tooling is CI-green (`36241763043`) and counts only human-saved rows as reviewed; draft rows remain pending.

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
- **NEEDS VERIFICATION** — detailed target-PC process/GPU state; repeated substantive control queries, including direct A3 artifact metadata, timed out.
- **DONE** — A2 old hard-NMS/NMM/weighted audit output files recovered and provenance verified; they do not include the ambiguity-guard comparison.
- **DONE** — A3 `a3-round2-weak1.json` recovered as a complete strict-profile artifact; final enhancement decision still requires post-fusion/NightOwls evidence.
- **RULE** — do not restart A2/A3 blindly until existing local process/artifact state is verified.


## Current execution blocker

- **DONE** — target-PC substantive Remote Desktop Commander control restored; process/artifact inspection succeeds. Blind restarts remain prohibited by policy, but the relay blocker is cleared.
- **DEFERRED (private domain gate)** — do not ask for 46-frame human review now. Public human-annotated benchmarks run first; later reduce to ~10-15 hardest frames + 3-5 temporal episodes before human confirmation.
- This does **not** mean A2/A3 benchmarks failed.
- Do not restart A2/A3 until existing local process/artifact state is verified.
- GitHub coordination, branch verification, and documentation remain available.
