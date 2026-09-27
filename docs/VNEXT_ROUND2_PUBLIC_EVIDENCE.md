# SpectraTrack vNext Round 2 — Public Evidence Contract

Status: coordination-only research contract.

Immutable production baseline:

- `integration @ 3ebc4d50213593cac62b97399447fccf6bbc1755`
- `main @ 2012eaae2f4ffe820a66d12e40346d911616cd03`

No production merge/release is authorized by this document.

## 1. Purpose

Round 2 uses existing **human-annotated public pedestrian/tracking datasets** for the primary quantitative evidence.

Private user CCTV is deferred to the end as a small domain sanity check.

Public evidence must remain separated by dataset semantics. Do not collapse all corpora into one synthetic score.

## 2. Immutable existing public corpora

### MOT17

Revision:

`mot17-public-r1`

Primary roles:

- A1 detection/fusion;
- A2 tracking.

Frozen Round-2 split:

- development: MOT17-04;
- held-out validation: MOT17-02/05/09/10/11/13.

Rule:

- no threshold/gate retuning from held-out metrics.

### CrowdHuman

Revision:

`crowdhuman-val-fbox-r1`

Primary role:

- A1 dense detection / close-person / crowd / fusion safety.

BBox policy:

- full-body `fbox`.

Tracking metrics:

- disabled; validation images are independent samples without canonical stable IDs.

## 3. New public corpus: DanceTrack

Planned revision:

`dancetrack-public-r1`

Official source:

- https://github.com/DanceTrack/DanceTrack
- https://dancetrack.github.io/

Terms/provenance:

- annotations: CC BY 4.0;
- image/video dataset: non-commercial research only;
- code: MIT;
- no raw media committed to Git/product artifacts.

Official annotation structure:

- MOT-style sequence layout;
- `seqinfo.ini`;
- `gt/gt.txt`;
- rows contain frame, stable identity, xywh bbox and constant trailing fields.

Importer rules:

- reuse existing generic MOT sequence/geometry/provenance primitives;
- do not inherit MOT17-specific class IDs, distractor classes, visibility semantics or semantic tags;
- preserve source sequence;
- preserve original frame;
- namespace stable GT IDs by sequence;
- preserve exact source hashes/provenance;
- no test GT fabrication.

Primary Round-2 use:

- A2 association / crossing / ID-stability stress.

Held-out policy:

- public DanceTrack validation split is the primary Round-2 association held-out;
- no ambiguity-guard retuning from validation metrics in this cycle.

### Association-only phase

Before detector inference:

- GT bbox geometry -> deterministic `Detection`;
- score = 1.0;
- label/class = person;
- appearance = null;
- detector_ran = true;
- GT identity is evaluator-only and never enters tracker input.

This phase performs zero ONNX calls.

Only if `current-ambiguity-guard` survives association-only validation may A2 spend detector inference on a DanceTrack detector-replay phase.

## 4. New public corpus: NightOwls

Planned revision:

`nightowls-public-r1`

Official sources:

- https://www.nightowls-dataset.org/
- https://www.nightowls-dataset.org/download/
- official SDK: https://gitlab.com/vgg/nightowlsapi

Terms:

- non-commercial research/teaching/personal experimentation;
- citation required;
- redistribution of dataset or modified versions prohibited;
- no raw/modified NightOwls bytes in Git/product artifacts.

Dataset characteristics:

- night/dawn pedestrian benchmark;
- 279k frames / 40 sequences;
- 1024x640;
- official PNG/JSON and Caltech-compatible distributions;
- official annotations include pedestrian, bicycledriver, motorbikedriver and ignore;
- pedestrian metadata includes occlusion/difficulty/pose/truncation;
- official documentation states tracking information exists across frames;
- scenes include low illumination, low contrast, blur/noise/reflections and weather variation.

Primary Round-2 use:

- A1 night detection/fusion;
- A3 enhancement quality/cost;
- optional A2 temporal tracking only after A5 validates stable-ID semantics in official annotation bytes/SDK.

Importer rules:

- official distribution/SDK only; no third-party resized mirrors;
- score official pedestrian target class;
- preserve official ignore regions;
- do not silently map rider classes into pedestrian;
- decide rider/other target-like ignore behavior only from official evaluation semantics;
- preserve only source-backed attributes;
- `tracking_supported=true` only after real official identity field validation.

Held-out policy:

- NightOwls validation is held out for Round-2 A1/A3 evaluation;
- no threshold/gate retuning after validation candidate results are observed.

### Compute-controlled slice

A5 may freeze a deterministic stratified Round-2 validation slice before any candidate result if full validation is too expensive for YOLO11x/960 multi-pass evaluation.

Any slice must:

- be selected only from source sequence / official annotation metadata;
- never use candidate/model output for selection;
- be frozen and hashed before A1/A3 run;
- preserve both positive and background/negative frames;
- cover available occlusion/difficulty/pose/bbox-size strata;
- be byte-identical across candidates;
- be labeled as a slice, not as a full NightOwls validation result.

A larger/full confirmation may be reserved for surviving finalists.

## 5. Optional secondary corpus: LLVIP visible

Official source:

- https://github.com/bupt-ai-cz/LLVIP
- terms: https://github.com/bupt-ai-cz/LLVIP/blob/main/Term%20of%20Use%20and%20License.md

Terms:

- non-commercial research/teaching/personal experimentation;
- citation required;
- privacy/identification restrictions apply.

Official annotation semantics:

- rectangle detection annotations;
- `person` class;
- VOC/XML tooling provided by official repository;
- no stable trajectory identity is part of the documented detector annotation contract.

Round-2 use:

- optional secondary A1/A3 low-light evidence;
- **visible/RGB side only**;
- no A2 identity/tracking metrics unless a separate official identity contract is later validated;
- infrared input is not SpectraTrack production evidence.

## 6. Optional fallback: KAIST visible

Official source:

- https://github.com/SoonminHwang/rgbt-ped-detection

Official repository describes:

- 95k aligned visible/thermal pairs;
- dense pedestrian-related annotations;
- temporal correspondence;
- dataset licensing under CC BY-NC-SA 4.0 and code/tooling under BSD-2-Clause.

Round-2 use:

- fallback only if NightOwls/LLVIP do not provide enough night evidence;
- SpectraTrack inference uses visible/RGB side only;
- thermal/LWIR cannot substitute for RGB detector evidence;
- raw data stays outside Git/product artifacts.

## 7. Canonical import/freeze requirements

All public imports must use the existing canonical QA JSONL.

Do not create a second evaluator/GT format.

Every import/freeze must record:

- importer branch + exact commit SHA;
- dataset name/version/split;
- terms/license metadata;
- acknowledgement flag where required;
- source sequence/image identity;
- original frame number where temporal;
- stable identity where source supports it;
- bbox conversion policy;
- ignore semantics;
- source files and hashes;
- canonical GT SHA-256;
- deterministic import-manifest SHA-256;
- frozen corpus revision/hash.

Raw public dataset bytes are not committed.

## 8. Cross-role evidence mapping

| Corpus | A1 | A2 | A3 | A4 |
| --- | --- | --- | --- | --- |
| MOT17 | detection/fusion held-out | tracking held-out | strict enhancement support | finalists only |
| CrowdHuman | dense/crowd safety | no tracking | optional detector-side context only | no |
| DanceTrack | optional later detector replay | association/ID primary | no | no |
| NightOwls | night detection/fusion | optional if stable IDs validated | primary night enhancement | finalists only |
| LLVIP visible | optional low-light | no identity metrics | optional low-light | finalists only if used |
| KAIST visible | fallback | optional only if formally activated | fallback | finalists only if used |

## 9. Required metrics

A1:

- TP/FP/FN;
- precision/recall/F1;
- localization IoU;
- center/width/height/area jitter;
- temporal IoU where temporal GT is valid;
- duplicate count;
- fusion mistakes;
- actual ONNX calls;
- wall time.

A2:

- tracking recall;
- ID switches;
- fragmentation;
- false track creation where defined;
- recovery events/latency;
- same-ID vs wrong-ID recovery;
- mean uninterrupted track length.

A3:

- recovered GT;
- lost GT;
- post-fusion precision/recall;
- FP delta;
- bbox stability;
- extra inference calls;
- recovered GT per extra call;
- FP cost per recovered GT;
- wall time.

A4:

- only after quality finalists exist;
- quality impact + calls/source-second;
- processing seconds/source-second;
- median/p95 inference latency;
- new-person discovery latency;
- global discovery bound.

## 10. Private CCTV final gate

Do not request review of all 46 prepared frames now.

After public finalists:

- reduce to roughly 10-15 hardest standalone frames;
- add 3-5 short temporal episodes;
- cover crossing, occlusion, disappear/reappear, ID swap, bbox jitter, tiny/distant, night/compression and likely FP backgrounds;
- AI preannotation remains DRAFT;
- user review UI should reduce action to CONFIRM / FIX / REJECT where possible;
- temporal episodes require ID-continuity confirmation.

Only after human confirmation freeze:

`cctv-golden-r1`

This remains the final domain sanity gate before integrator review.
