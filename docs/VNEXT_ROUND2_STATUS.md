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

- **DONE** — A1 full MOT17 held-out complete: evidence-aware F1 0.527449 vs hard-NMS 0.516517, FP -3141, TP +132, fusion mistakes 1016 -> 540; result/marker/replays verified.
- **DONE (download integrity)** — DanceTrack val.zip complete and valid with 25 GT files; A5 local import/freeze may proceed.
- **BLOCKED** — NightOwls official local bytes/freeze still absent.

- **IN PROGRESS** — A1 MOT17 held-out: 4/6 sequences complete (02/05/09/10); provisional evidence-aware aggregate F1 0.558231 vs hard-NMS 0.547869. Final decision waits for 11/13.

- **DONE** — A2 Round-2 decision: RETAIN CURRENT TRACKER; ambiguity-guard rejected by real identical-replay ID-switch gate.
- **IN PROGRESS** — A1 real MOT17 held-out retry is active on target PC; first MOT17-02 prefusion artifact is present.
- **IN PROGRESS** — official DanceTrack validation download is active on target PC; freeze waits for verified per-sequence GT.
- **BLOCKED** — NightOwls local official bytes/frozen revision are not yet available; A1/A3 night gates cannot start.

- **DONE** — hourly coordinator condition watch enabled for branch/CI/target-PC recovery and Git-persistent continuation.

- **DONE** — unified public evidence contract committed in `docs/VNEXT_ROUND2_PUBLIC_EVIDENCE.md`.

- **DONE** — all public-first role documentation heads are CI-green after strategy dispatch.

- **DONE** — public `mot17-public-r1` frozen.
- **DONE** — public `crowdhuman-val-fbox-r1` frozen.
- **DONE** — Round-1 detector/fusion/tracking/enhancement/scheduler evidence collected.
- **DONE** — Round-2 evidence review and assignments created.
- **DONE** — A1 resumable canonical public-corpus runner implemented and CI-validated.
- **DONE** — A1 canonical replay-export fix validated; branch docs head `32ae820...` CI-green (`36242653285`).
- **NOT DONE** — A1 held-out multi-sequence Round-2 validation (target-PC run still required).
- **DONE (old audit artifacts verified) / IN PROGRESS** — A2 hard-NMS/NMM/weighted current-tracker audit artifacts exist and are valid; required current-vs-ambiguity real-replay comparison remains NOT DONE.
- **DONE (MOT17 strict-profile artifact verified) / IN PROGRESS** — A3 weak-person strict profile exists with measured quality/cost; post-fusion and NightOwls decision remains NOT DONE.
- **DONE** — A4 real RX 5700 XT execution-cost frontier.
- **DONE** — A5 draft-seeded private review tooling and 46-frame review pack preparation.
- **DONE** — A5 review-progress accounting validated; branch docs head `8faf6c9...` CI-green (`36242654534`).
- **IN PROGRESS** — A5 DanceTrack official-terms/import/freeze work for `dancetrack-public-r1`.
- **IN PROGRESS** — A5 NightOwls official-terms/import/freeze work for `nightowls-public-r1`.
- **NOT DONE** — A2 DanceTrack association-only current vs ambiguity-guard benchmark.
- **NOT DONE** — A1/A3 NightOwls public validation.
- **NOT DONE / DEFERRED** — reduced private CCTV human-confirmed sanity pack.
- **NOT DONE / DEFERRED** — `cctv-golden-r1` freeze after public finalists.
- **NOT DONE** — final cross-role candidate table on `cctv-golden-r1`.
- **LOCKED** — `agent/vnext-integrator`.
- **LOCKED** — production merge/release.

## A1 Detection / Fusion — issue #18

Current known Round-2 head:

`agent/vnext-detection @ 89d58aad9890949ce410ed95e1112bcd3cd1cfd6`

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

`agent/vnext-qa @ bf63820f81cd13fdfae8680a200e25f030d314fc`

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
