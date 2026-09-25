# SpectraTrack QA benchmark data

This directory is for the local annotated regression set used by `spectratrack.qa_benchmark`.

Real benchmark video is intentionally **not committed by default**. Video licensing, privacy, and provenance must be reviewed before a clip is added or redistributed. Synthetic fixtures are useful for testing the evaluator, but they do not count as evidence that detection quality improved.

## Required coverage

The first representative set should contain short clips covering all of these categories:

- good / easy daylight video;
- generally poor video;
- darkness / low light;
- strong compression artifacts;
- small people;
- partially occluded people;
- crowds / close interactions;
- moving camera;
- fast target or camera motion;
- distant vehicles;
- small animals.

People are the primary acceptance target. A clip may have several tags.

Keep a held-out subset that is not used while tuning thresholds. Record provenance and license for every source clip.

## Layout

Recommended local layout:

```text
pc/benchmarks/
  ground_truth.jsonl
  videos/
    good_daylight.mp4
    night_small_people.mp4
    ...
  results/
    BASELINE.json
    AGENT_1.json
    AGENT_2.json
    AGENT_1_PLUS_3.json
```

`videos/` and generated `results/*.json` are ignored by Git. The annotation file may be committed only when doing so does not disclose sensitive/private footage metadata.

## Ground truth format

Ground truth is JSON Lines: one annotated video frame per line.

```json
{"video":"night_small_people.mp4","frame":120,"tags":["dark","compression","small_people"],"objects":[{"id":"p1","label":"person","bbox":[812,316,827,351],"attributes":["small"]}]}
```

Fields:

- `video`: path relative to `--video-root`;
- `frame`: zero-based decoded frame index;
- `tags`: frame/clip conditions used for metric breakdown;
- `objects[].id`: stable local object id across annotated frames; required for useful ID-switch/fragmentation metrics;
- `objects[].label`: detector label, normally `person`;
- `objects[].bbox`: `[x1,y1,x2,y2]` in source-frame pixels;
- `objects[].attributes`: optional object conditions such as `occluded`, `tiny`, or `blurred`;
- `objects[].ignore=true`: uncertain/unscorable object. A prediction overlapping an ignored object is not counted as a false positive.

Include annotated background/confuser frames with an empty `objects` list. Otherwise false-positive measurements are biased downward.

## Baseline run

From `pc/`:

```powershell
python -m spectratrack.qa_benchmark run `
  --ground-truth benchmarks/ground_truth.jsonl `
  --video-root benchmarks/videos `
  --model ..\models\yolo11n.onnx `
  --run-name BASELINE `
  --revision main `
  --output benchmarks/results/BASELINE.json
```

The runner uses the current `YoloOnnxDetector`, affine CMC, appearance cue, and `MultiObjectTracker` without changing production interfaces. It processes every frame through the detector up to the last annotated frame in each clip, while metrics are scored only on annotated frames.

Saved results include:

- person recall;
- person precision;
- false negatives;
- false positives;
- recall by apparent person height;
- metrics by frame/object tags;
- tracking recall on annotated frames;
- ID switches;
- track fragmentation;
- end-to-end FPS;
- detector-only FPS;
- exact model SHA-256 and providers;
- exact ground-truth SHA-256;
- detector settings.

Portable DirectML/AMD VRAM measurement is not available through ONNX Runtime. `peak_vram_mb` remains null unless a real externally measured value is supplied with both `--peak-vram-mb` and `--vram-source`. Never fill it with an estimate.

## Compare a branch against baseline

Run the same command on each branch and give each result a distinct `--run-name`, then:

```powershell
python -m spectratrack.qa_benchmark compare `
  --baseline benchmarks/results/BASELINE.json `
  --candidate benchmarks/results/AGENT_1.json `
  --output benchmarks/results/BASELINE_vs_AGENT_1.json
```

The comparison is strict by default. It fails with exit code 2 if the candidate introduces any of these regressions:

- a ground-truth object that baseline detected becomes a **NEW FALSE NEGATIVE**;
- recall decreases;
- precision decreases;
- ID switches increase;
- fragmentation increases.

Allowed regressions must be made explicit with the corresponding `--max-...` option. Results from different ground-truth files, labels, or matching IoU settings are rejected instead of silently compared.

Use the same procedure for `AGENT_2`, `AGENT_3`, `AGENT_4`, and integration combinations such as `AGENT_1_PLUS_3`.

## Annotation guidance

Use IoU 0.5 as the initial matching threshold unless the benchmark decision is deliberately changed for all compared runs.

For small-person work, annotate the visible body extent consistently. If a person is too ambiguous to score, mark it `ignore=true` rather than forcing a questionable box.

Do not reuse the final held-out clips to tune thresholds repeatedly. The point of the benchmark is to catch regressions, not to optimize against a tiny memorized set.
