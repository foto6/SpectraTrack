# SpectraTrack vNext Round 2 — Evidence Review

Status: coordination-only. No production behavior is changed by this branch.

Immutable product baseline:

- `integration @ 3ebc4d50213593cac62b97399447fccf6bbc1755`
- `main @ 2012eaae2f4ffe820a66d12e40346d911616cd03`
- research base: `vnext-base @ d03af3ae6425d3ea2e4d52e25389fecc09957394`

Target hardware used for local inference evidence: AMD Radeon RX 5700 XT through ONNX Runtime DirectML.

Current specialist heads used by this review:

- A1 detection: `943abee566c45116cee2b0e72b7d2c48753891ef`
- A2 tracking: `2a46322c9d4eb6b0bd86b3126670dc424f9abe11`
- A3 enhancement: `86b1460b6f9879038de92eb1e1341a96221e6c03`
- A4 performance: `1cd9f38a09d35f3b125c4a214d8514200ee0f76d`
- A5 QA: `e7251c964c63da496caac506d4d7da1876977230`

The detector model used in the local measurements was `yolo11x.onnx`, SHA-256
`e84cbad768b218d74ecc85e3e52d84631123719a6951b3ddf6eddc850d5b3f73`.

## Public frozen corpora used

- `mot17-public-r1`: MOT17 train ground truth, FRCNN variants, stable person IDs.
- `crowdhuman-val-fbox-r1`: CrowdHuman validation, full-body `fbox`, detection-only.

These public corpora are valid research evidence, but they do **not** replace the missing human-confirmed private CCTV GOLDEN corpus required before production integration.

## Detector threshold sweep

### MOT17

| threshold | precision | recall | F1 | bbox IoU |
| ---: | ---: | ---: | ---: | ---: |
| 0.35 | 0.8153 | 0.5222 | 0.6367 | 0.8236 |
| 0.20 | 0.8292 | 0.5876 | 0.6878 | 0.8154 |
| 0.12 | 0.7785 | 0.6289 | 0.6957 | 0.8086 |

### CrowdHuman validation

| threshold | precision | recall | F1 | bbox IoU |
| ---: | ---: | ---: | ---: | ---: |
| 0.35 | 0.7004 | 0.4210 | 0.5259 | 0.7722 |
| 0.20 | 0.6493 | 0.4763 | 0.5495 | 0.7642 |
| 0.12 | 0.6048 | 0.5129 | 0.5551 | 0.7590 |

Interpretation:

- `0.12` is the strongest current research threshold for recall/F1 on both public corpora.
- The precision and localization trade-off is material. It is not a production default.
- The next detector step should not be another blind threshold reduction. It should recover precision using cross-pass evidence while preserving weak-person recall.

## Full-frame + tile fusion evidence

MOT17-04, first 600 frames, person threshold 0.12:

| method | recall | precision | bbox IoU | center jitter px | fusion mistakes |
| --- | ---: | ---: | ---: | ---: | ---: |
| hard NMS | 0.7499 | 0.5462 | 0.8134 | 2.205 | 195 |
| conservative NMM | 0.7732 | 0.5160 | 0.8202 | 1.907 | 60 |
| weighted | 0.7696 | 0.5136 | 0.8198 | 1.814 | 60 |

Interpretation:

- hard NMS retains the best precision of these three but loses recall and has the worst jitter/fusion mistakes;
- conservative NMM has the highest recall;
- weighted fusion has the lowest measured center/size jitter and nearly the same recall as conservative NMM;
- both NMM/weighted pay a large FP/precision cost;
- therefore the next A1 problem is **evidence-aware acceptance/scoring**, not simply choosing one of the three existing methods.

## Tracking evidence

On the same 600-frame MOT17-04 replay set, the current tracker is substantially more stable than the research global-assignment/reference-style candidates.

Current tracker:

| fusion | tracking recall | ID switches | fragmentation | mean uninterrupted length |
| --- | ---: | ---: | ---: | ---: |
| hard NMS | 0.8585 | 335 | 281 | 41.12 |
| conservative NMM | 0.8671 | 590 | 296 | 28.98 |
| weighted | 0.8643 | 585 | 276 | 29.98 |

Global-assignment / Byte-style / BoT-SORT-style candidates produced roughly 1,100–1,300 ID switches on these replays. OC-SORT-style produced more than 4,000.

Interpretation:

- wholesale global assignment replacement is rejected for this cycle;
- current greedy two-stage semantics remain the control;
- the tracker problem should now be mined at the event level: crossings, ambiguous competing detections, occlusion recovery, and bbox instability;
- fusion choice materially changes tracker behavior. A1/A2 must compare against identical replay bytes.

## Enhancement evidence

MOT17-04 first 300 frames, 2,400 ROIs:

| operation | affected ROIs | extra inference calls | recovered GT persons | new FP observations pre-fusion |
| --- | ---: | ---: | ---: | ---: |
| current adaptive | 1762 | 1762 | 371 | 14886 |
| current adaptive cached | 1762 | 1762 | 371 | 14886 |
| bilateral | 283 | 283 | 67 | 2369 |
| sharpen | 1670 | 1670 | 313 | 13699 |
| gamma+CLAHE | 0 | 0 | 0 | 0 |

Important measured facts:

- current adaptive multi-candidate experiment took about 987 wall seconds for 10 source seconds;
- cached adaptive preserves quality result but only reduces preprocessing work; detector inference still dominates;
- broad enhancement activation creates too many extra calls and too much pre-fusion FP evidence;
- the current quality router marked blur on ~69.6% of ROIs, which is too broad to be a useful expensive-inference trigger on this corpus.

Interpretation:

- general adaptive enhancement is rejected as the default production path;
- enhancement remains a narrow recovery tool only;
- the next A3 candidate must be weak-evidence / track-context gated and hard-budgeted;
- `no enhancement` remains a first-class candidate.

## Scheduler evidence

1080p simulation at 30 FPS:

| policy | ONNX calls/frame | calls/source-second | periodic global discovery |
| --- | ---: | ---: | ---: |
| CURRENT | 17.0 | 510 | every detector frame |
| COARSE_TO_FINE | 6.0 | 180 | every detector frame |
| TRACK_GUIDED | 3.067 | 92 | 0.50 s |
| BUDGETED_ADAPTIVE | 4.0 | 120 | 0.33 s |

Interpretation:

- the field performance problem is structural and dominated by detector-call count;
- even the current research scheduler candidates remain expensive for YOLO11x/960 on RX 5700 XT;
- A4 must move from call-count simulation to target-PC wall-time measurement with detector cadence and hard time/call budgets.

## Round 2 architectural conclusions

1. Keep current tracker semantics as the control. Do not replace it with the current global/Byte/BoT/OC research candidates.
2. Continue A1 with evidence-aware fusion/filtering. Preserve weak-person recall but require stronger evidence for new low-confidence detections.
3. Treat weighted and conservative NMM as ingredients, not production winners.
4. Restrict enhancement to narrowly selected recovery ROIs; broad adaptive is not a viable production default.
5. Make scheduler budget explicit in both calls and wall time. Detector cadence is now part of the performance research problem.
6. Do not integrate into production until a private human-confirmed CCTV GOLDEN corpus exists and the surviving candidates pass it.
7. Keep all round-2 work isolated. No merge to `integration`, `main`, or RC.
