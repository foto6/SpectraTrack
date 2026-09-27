# A1 backend and person-specialist research notes

Research snapshot for `agent/vnext-detection`, based on `vnext-base @ d03af3ae6425d3ea2e4d52e25389fecc09957394`.

This file records compatibility/feasibility gates. Published COCO metrics are not SpectraTrack results.

## Detector backend compatibility

| Candidate | Code / weights license gate | Export / ORT contract | Pre/postprocessing | DirectML status | A5 CCTV quality status |
| --- | --- | --- | --- | --- | --- |
| Current SpectraTrack YOLO-compatible ONNX | SpectraTrack adapter is MIT. Exact upstream model/weights license must come from model provenance; do not infer from filename. Ultralytics YOLO11 upstream is AGPL-3.0 or Enterprise. | Existing fixed-size YOLO ONNX decoder supports verified raw/end-to-end layouts. | Existing letterbox + RGB /255; decoder-local class-aware NMS. | Existing Windows runtime prefers DirectML with CPU fallback; exact frozen model still needs run provenance. | BLOCKED until A5 frozen corpus. |
| RF-DETR Nano/Small/Medium candidate | Upstream documents core Nano-Large training/inference code and models under Apache-2.0. Exact checkpoint hash/provenance still required. XLarge/2XLarge are outside this initial candidate set because upstream documents a different plus/PML license. | Upstream ONNX export and ONNX Runtime inference are documented. Export contract uses normalized cxcywh boxes + class logits. | RGB; half-pixel bilinear resize without antialias; [0,1]; ImageNet mean/std; NCHW. Decode uses per-class sigmoid and query/class top-k. | UNVERIFIED on SpectraTrack DirectML. Must probe the exact export; provider priority is not proof that every node executed on DML. | BLOCKED until A5 frozen corpus. |
| RT-DETRv2 R18/R34 candidate | Repository is Apache-2.0. Exact checkpoint/weights provenance must still be recorded because a repository license alone is not a substitute for checkpoint provenance. | Official exporter uses opset 16 with `images` + `orig_target_sizes` inputs and `labels` + `boxes` + `scores` outputs; ONNX Runtime deployment is documented. | Validation path resizes RGB to fixed input, scales to float [0,1]; exported postprocessor supplies labels/boxes/scores. | UNVERIFIED on SpectraTrack DirectML. Must probe exact exported model. | BLOCKED until A5 frozen corpus. |

Research adapter CLI: `python -m spectratrack.research.vnext_detector_backend_lab`.

The adapter lab records model SHA-256, input/output contract, provider order, thresholds, preprocessing/inference/postprocessing time, detector wall time, inference-call count, A5 ground-truth hash/revision, person metrics, apparent-height/tag breakdowns from the canonical QA evaluator, and matched bbox localization IoU.

VRAM remains null unless supplied from a real measurement source. Session provider priority is recorded, but the harness deliberately does not pretend it can infer per-node DirectML-to-CPU fallback from `session.get_providers()`.

## Required backend protocol once A5 freezes the corpus

For every backend:

1. freeze exact A5 corpus revision and ground-truth SHA-256;
2. freeze model file and SHA-256;
3. record code/weights license and export source;
4. run contract probe;
5. run one smoke inference and verify bbox coordinate/class mapping manually;
6. run the same annotated frames, same person evaluation IoU, same condition tags;
7. record person recall/precision/FP/FN, size buckets, night/blur/compression/occlusion/high-angle tags and bbox localization;
8. record preprocess/inference/postprocess/detector-wall/benchmark-wall times and inference calls;
9. record provider priority and real telemetry only when actually measured;
10. compare results only when corpus hash/revision is identical.

Do not use published COCO AP to fill any SpectraTrack result cell.

## Person-specialist feasibility

### Candidate A — current YOLO family fine-tuned for person

Feasibility: technically straightforward if an upstream checkpoint/export with acceptable license is chosen, but the legal gate is model-specific. Ultralytics YOLO11 documentation currently states AGPL-3.0 / Enterprise licensing. Do not train or redistribute a specialist checkpoint until the intended license path is explicit.

### Candidate B — RF-DETR Nano/Small fine-tuned for person

Feasibility: technically attractive for research because upstream documents Apache-2.0 for core Nano-Large code/models and direct fine-tuning from COCO/YOLO dataset formats. Start with Nano/Small for training-cost and export/DirectML probes; do not assume a larger model improves tiny-person CCTV recall.

### Dataset: CrowdHuman

Official terms permit only non-commercial research and educational use and forbid redistribution of the images. Therefore:

- allowed here only as a research-feasibility source under those terms;
- not approved as the basis for a redistributable/commercial specialist checkpoint without separate permission/legal review;
- images must not be committed to SpectraTrack;
- conversion should preserve provenance and source hashes.

Conversion proposal: parse CrowdHuman ODGT, map valid `person` entries to one person class, use `vbox` for a visible-body training variant aligned with the current CCTV evaluation convention, preserve `fbox` only as a separately named experiment, and map mask/ignored entries to ignore regions rather than positives.

### Dataset: WiderPerson

Feasibility: useful content-wise for dense/occluded pedestrians, but licensing is not cleared. Secondary indexes conflict between `non-commercial` and `unknown`, while the original project terms were not reliably retrievable during this research pass. Status: **BLOCKED pending official license confirmation**.

If cleared, preserve original classes during conversion. A proposed research mapping is pedestrian + partially-visible-person to person positives, crowd/ignore as ignore regions, and riders handled only by an explicit experiment rather than silently remapped.

### Future user-confirmed CCTV training split

May be used only after the owner explicitly defines a training split and provenance. The A5 golden validation corpus is forbidden for training.

Enforce leakage control by storing the source/video/frame hash list for both training and A5 validation and failing the preparation step on overlap.

## Training-cost gate

No full training run is justified yet.

Upstream RF-DETR training documentation gives illustrative memory starting points rather than guarantees: about batch 1 + gradient accumulation/checkpointing at 8 GB, batch 4 with accumulation at 16 GB, and batch 8 with accumulation at 24 GB for its documented workload. Those are upstream examples, not SpectraTrack measurements.

Required cost procedure after legal/data gates:

1. convert dataset and validate 100 random annotations visually;
2. run a tiny overfit/sanity subset to validate class/bbox semantics;
3. run a 3–5 epoch pilot on the intended model/resolution/hardware;
4. measure actual images/s, peak VRAM, wall time/epoch and checkpoint size;
5. extrapolate full-run time/cost from that pilot only;
6. proceed to a longer fine-tune only if the generic detector has a measured gap on the frozen A5 corpus that the specialist hypothesis could address.

Do not quote an estimated dollar/time budget before the pilot because model size, resolution, augmentation, physical batch size and hardware change it materially.

## Decision before A5

- Do not replace the production detector.
- Do not start full person-specialist training.
- Keep RF-DETR Nano/Small and RT-DETRv2 R18/R34 as first backend probes.
- Keep weighted cross-pass coordinate fusion as a candidate, not a production recommendation.
- Wait for A5 frozen validation revision/hash before any real quality ranking.
