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

- **DONE** — A1 MOT17 held-out complete. Evidence-aware passed MOT17 as the sole fusion challenger.
- **DONE / FAIL CHALLENGER** — A1 CrowdHuman full 4,370-image dense-safety gate complete. Evidence-aware improved precision/F1/FP/localization but recall fell 0.570240 -> 0.563173, violating the frozen no-recall-loss rule. Round-2 A1 production conclusion: **RETAIN HARD-NMS CONTROL**; no held-out retuning.
- **DONE** — A2 Round-2 decision: **RETAIN CURRENT TRACKER**. current-ambiguity-guard rejected by identical-replay ID-switch gate.
- **WAITING** — A3 strict candidate set remains OFF / bilateral / current_adaptive_cached; NightOwls held-out is still required for the enhancement decision.
- **PARKED** — A4 scheduler quality search remains deferred until A3/public quality finalists are resolved.
- **DONE** — DanceTrack frozen as `dancetrack-public-r1`, corpus SHA `df240532ad3f2099947f318b682737ccdb3e6345dc9da1e380ff4cab8d6c6b5c`; A2 rejected candidate does not advance to DanceTrack without explicit architect reopening.
- **IN PROGRESS** — A5 official NightOwls intake is active on isolated E:. Official JSON and SDK are present; validation ZIP is actively downloading. Latest observed size ~14.62 / 57.48 GB (~25.4%).
- **NOT DONE** — NightOwls import/validation/freeze, A3 NightOwls held-out/final ON-OFF decision, A4 final quality/cost check, reduced private CCTV human-confirmed sanity, final cross-role candidate table.
- **LOCKED** — `agent/vnext-integrator`, production merge/release.

## A1 Detection / Fusion — issue #18

Current known Round-2 head:

`agent/vnext-detection @ 3dc31c8fa8b323ce86afc14c3105a0c2247a2c48`

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


## Live blocker update — duplicate A1 writer

- **CLEARED / NEEDS RESULT VALIDATION** — the previously observed duplicate CrowdHuman PIDs `28260` and `19944` both exited on their own before the corrected probe. Exactly 4,370 prefusion files, the final result JSON, and completion marker now exist. No A1 process was killed or restarted; validate hashes/provenance/metrics next.
- **IN PROGRESS** — official NightOwls ZIP downloader PID `15200` remains active on the isolated E: root; observed ZIP size 13,480,124,416 bytes; no duplicate downloader was started.


## A1 CrowdHuman final gate

- **DONE** — result SHA `c09f91ec4c27f28057c8937c84390e10c80cbd16ae56fe3840cbf1363055690a`; marker SHA `23f3cc80df109fc1e0d93851b14f57a6f432e7e02039dd343b9b20e249de5957`; GT/model hashes match frozen provenance.
- **DONE** — 4,370/4,370 unique selected images and 4,370 prefusion artifacts.
- **FAIL / REJECT CHALLENGER** — evidence-aware vs hard-NMS: precision +4.447576 pp, recall -0.706668 pp, F1 +3.115044 pp, FP -20,677, bbox IoU +0.007883, fusion mistakes -502. Because the frozen rule required recall not lower, the challenger fails dense safety.
- **RETAIN CONTROL** — hard-NMS remains the A1 Round-2 fusion policy. No held-out retuning.
- **A1 NIGHTOWLS CHALLENGER GATE NO LONGER BLOCKING** — there is no surviving new fusion challenger to rescue in this cycle. NightOwls intake remains required for A3 and may be used for hard-NMS baseline characterization only.


## A1 specialist handoff synchronization

- **DONE** — A1 branch advanced handoff-only to `3dc31c8fa8b323ce86afc14c3105a0c2247a2c48`.
- **DONE** — issue #18 records the same CrowdHuman FAIL / hard-NMS retain decision.
- No production/runtime code was changed by this final handoff update.
