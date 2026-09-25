# SpectraTrack PC v0.3+ engineering plan

This plan is finite. Work is closed against measurable validation criteria instead
of repeatedly lowering thresholds until the screen fills with false positives.

## Milestone A — v0.3.0 release closure
- [x] Cross-video batch graph
- [x] Best-tracklet previews + local HTML review
- [x] SAME / DIFFERENT / UNSURE review round-trip
- [x] Selected-target optical-flow refinement
- [x] One-click Windows video/folder launchers
- [x] Collision-safe recursive video IDs
- [x] End-to-end synthetic video regression
- [x] PC and Android version metadata aligned to 0.3.0
- [x] Publish v0.3.0 Windows ZIP + Android APK + SHA-256 checksums

## Milestone B — high-recall small-pedestrian detection

### Failure mode to solve

Representative footage is high-angle, night-time, compressed CCTV where a person
can occupy only a small number of pixels. The current generic detector can find
many large vehicles while missing visible pedestrians entirely.

The goal is **high pedestrian recall**, not a blanket confidence-threshold drop.

### B1 — benchmark before tuning
- [ ] Build a small validation set from representative overhead/night footage.
- [ ] Manually annotate every clearly visible person and a sample of confusing objects.
- [ ] Record current baseline person recall, precision and false positives/frame.
- [ ] Break metrics down by apparent person height in pixels and lighting/occlusion.
- [ ] Keep the validation clips out of tuning decisions after thresholds are frozen.

### B2 — quality-max person pipeline
- [x] Add a dedicated `people-recall` / `quality-max` mode.
- [x] Run the normal full-frame detector for cars/large objects.
- [x] Add a second **person-only high-resolution pass**.
- [x] Add tiled/sliced inference with overlap so tiny people are presented to the model at a useful scale.
- [x] Merge full-frame and tile detections with class-aware NMS/WBF without duplicate boxes.
- [x] Support class-specific thresholds: lower threshold for `person`, normal thresholds for other classes.
- [x] Keep a hard minimum quality gate so lowering person confidence does not flood the scene with single-frame junk.

### B3 — temporal recovery of weak people
- [ ] Keep weak person candidates for a short temporal window instead of discarding them immediately.
- [ ] Confirm a weak candidate when spatially consistent evidence appears across multiple frames.
- [ ] Allow confirmed person tracks to survive short detector dropouts.
- [ ] Permit temporally consistent low-confidence person detections to recover an existing track.
- [ ] Do not create a stable person track from one isolated low-confidence frame.

### B4 — image-analysis variants
- [ ] Benchmark original RGB vs non-generative low-light/contrast analysis for the person pass.
- [x] Add adaptive non-generative routing for blur/darkness/compression/resolution/noise; processed-only people require raw-frame corroboration.
- [ ] Benchmark multiple detector input sizes.
- [ ] Benchmark tile sizes/overlap and detector cadence.
- [ ] Do **not** use generative super-resolution as ground truth for detection; it can hallucinate detail.
- [ ] If neural restoration is tested, evaluate it only as an optional analysis branch and measure whether recall actually improves.

### B5 — model selection / training
- [ ] Compare the current generic YOLO model with at least one small-object/overhead-friendly detector configuration.
- [ ] Measure person recall before choosing a larger model solely from model size.
- [ ] If generic weights remain the bottleneck, fine-tune a person detector on legally usable overhead/night pedestrian data.
- [ ] Keep model provenance, license and SHA-256 in the existing model-manifest system.

### B6 — acceptance criteria

First target for **offline quality-max** on the representative validation set:
- [ ] >= 95% recall for clearly visible pedestrians at/above the validated minimum pixel size.
- [ ] >= 90% recall across all annotated visible pedestrians in the validation set.
- [ ] No more than 0.25 persistent false person tracks per frame on the validation clips.
- [ ] Every miss and false positive is exportable as a review sample for the next iteration.

These numbers are engineering targets, not a promise that every human in every
camera image can be detected. People smaller than the source image can actually
resolve, severe occlusion, glare, motion blur or compression can make detection
physically ambiguous.

### B7 — performance path
- [ ] First make quality-max correct even if processing is slower than real time.
- [ ] Profile full-frame pass, tile pass, merge and tracker separately.
- [ ] On Windows prefer DirectML where available and keep CPU fallback.
- [ ] Add adaptive tiling/cadence so live mode can trade recall for latency explicitly.
- [ ] Expose the active person-recall mode and measured detector latency in the HUD/log.

## Milestone C — validation UX
- [ ] Add a person-miss review export: frame, crop, timestamp, source and detector settings.
- [ ] Add a compact per-video summary: annotated/seen people, recovered tracks, missed intervals.
- [ ] Add side-by-side baseline vs people-recall benchmark command.
- [ ] Store the exact model hash, thresholds, tile layout and input size in every benchmark result.

## Data / safety semantics
- Person detection is object detection only.
- No face recognition, facial embedding, biometric identity or named-person identification.
- Cross-video person links remain `same_appearance_candidate`, never identity claims.
- Pseudo-thermal remains false-color RGB, not thermal sensing.
