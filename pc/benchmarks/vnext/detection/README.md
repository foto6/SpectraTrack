# A1 vNext detection / fusion research

Research-only artifacts for agent/vnext-detection.

Canonical start:

`vnext-base @ d03af3ae6425d3ea2e4d52e25389fecc09957394`

Nothing in this directory is production detector policy.

## Stage 1: cross-pass fusion

The research harness intentionally distinguishes:

1. **decoder-local NMS** inside each full-frame or tile inference — preserved from the current detector;
2. **final cross-pass fusion** across already-decoded full-frame/tile detections — the isolated variable under study.

Candidates:

- `hard-nms`: current class-aware winner-takes-box behavior;
- `conservative-nmm`: direct-to-seed GreedyNMM-style grouping with IoU, center-distance and size-ratio safety gates; geometry is the group envelope;
- `weighted`: the same conservative grouping, but score/evidence-weighted bbox coordinates.

The conservative candidates deliberately do not use transitive merge chains. A candidate must match the highest-score seed directly. This is intended to reduce accidental merges of nearby people in crowds.

Default synthetic config:

- final fusion IoU: `0.55`
- center-distance gate: `0.20 * min(box dimensions)`
- max width/height ratio: `1.80`
- score power: `1.0`
- full-frame evidence weight: `1.0`
- tile evidence weight: `1.0`
- evaluation IoU: `0.50`

Synthetic cases cover:

- alternating full-frame/tile score winner;
- tile-boundary duplicates;
- full-frame + tile duplicates;
- nearby people;
- partial-person vs full-body geometry;
- tiny person;
- frame-edge person;
- two distinct people with strongly overlapping boxes.

Regenerate the synthetic result from `pc/`:

```powershell
python -m spectratrack.research.vnext_detection_fusion synthetic `
  --source-commit d03af3ae6425d3ea2e4d52e25389fecc09957394 `
  --output benchmarks/vnext/detection/synthetic_fusion_report.json
```

The committed synthetic report is only a deterministic failure/safety fixture. It is **not** CCTV quality evidence and must not be used to claim recall or localization improvement on real footage.

## Pre-fusion dump

Capture the existing current-YOLO full-frame/tile outputs **after decoder-local NMS but before final cross-pass merge**:

```powershell
python -m spectratrack.research.vnext_detection_fusion collect `
  --model ..\models\yolo11n.onnx `
  --video D:\data\clip.mp4 `
  --video-id clip.mp4 `
  --source-commit <exact-branch-sha> `
  --output benchmarks\vnext\detection\clip.prefusion.jsonl `
  --input-size 640 `
  --conf 0.35 `
  --decoder-iou 0.45 `
  --person-conf 0.12 `
  --tile-size 640 `
  --tile-overlap 0.20
```

The dump records source commit, video/model hashes, provider, exact detector/tile config, dimensions/timestamps, candidate evidence source, policy-run count, actual inference-call count, detector stage timings, and wall time.

Enhancement is intentionally excluded from this A1 collector because A3 owns enhancement semantics.

## Canonical tracker replay

Convert one frozen pre-fusion dump into canonical `spectratrack-detection-replay-v1` without detector inference:

```powershell
python -m spectratrack.research.vnext_detection_fusion replay `
  --input benchmarks\vnext\detection\clip.prefusion.jsonl `
  --output benchmarks\vnext\detection\clip.weighted.replay.jsonl `
  --method weighted
```

Use `--method hard-nms`, `conservative-nmm`, or `weighted`.

The replay metadata records the selected cross-pass fusion config. Detection records contain only canonical bbox/score/class/label fields plus `appearance: null`, so A2 can consume the same frozen detector evidence without rerunning detector inference.

## Real-data gate

No fusion candidate is recommended for production integration until A5 publishes a frozen, human-confirmed CCTV corpus revision and all candidates are evaluated on the exact same detector outputs / ground truth.

The synthetic result currently supports only this research conclusion:

- hard NMS has a reproducible winner-flip failure mode by construction;
- conservative grouping can avoid the included high-overlap-distinct-person merge case;
- envelope NMM can worsen localization geometry;
- weighted coordinate fusion is worth continuing to real-corpus evaluation;
- none of these statements proves a real CCTV quality gain.
