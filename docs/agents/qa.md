# QA / regression agent state

## Ownership

Branch: `agent/qa-integration`

Primary scope:

- ground-truth format and validation;
- detector/tracker quality evaluation;
- TP/FP/FN, precision, recall;
- size/tag/attribute breakdowns;
- ID switches and fragmentation;
- baseline-vs-candidate comparison;
- new-false-negative regression gates;
- benchmark documentation and tests.

Normally outside scope:

- implementing detector/tracker/enhancement/performance features for other agents.

Historical note: the older `agent/qa` branch contains the pre-squash QA history. It was not rewritten or force-pushed. The current handoff branch was fast-forwarded to the canonical coordination base before this cleanup.

## Current objective

Provide an independent, reproducible gate for deciding whether agent branches actually improve or regress SpectraTrack, while keeping the evaluator input contract strict enough that malformed annotations cannot silently produce misleading metrics.

## Important invariants

- Baseline and candidate comparisons must use identical ground truth and compatible evaluation settings.
- A candidate must not silently introduce new false negatives.
- Synthetic evaluator fixtures validate the evaluator, not real detector quality.
- Real quality claims require a representative annotated real dataset.
- VRAM must not be fabricated or inferred without a trustworthy measurement source.
- Serialized benchmark-result schema changes require shared coordination; this handoff preserves result schema v1.

## Current branch state

Original shared base: `integration @ 573087cdf60c1a00bbc93b8764733b2ade3cbdd8`

Current coordination base read from `origin/integration`: `ee76f4e0c9728af9f94a363c3cee682130599521`

Working branch: `agent/qa-integration`

Validated implementation HEAD before this state-only handoff commit: `97bbf3259f1878f4ee97539a6d6857d01ec49149`

PR used only for CI/handoff evidence: `#9 QA: harden regression evaluator handoff`

The final branch HEAD is the commit containing this state update; its exact SHA is reported by the branch/PR after the commit because a Git commit cannot contain its own SHA.

## Implemented QA work

Already present in `integration` through squash commit `102a6426a0329ee128a424b4277cb31237acdcb4`:

- annotated JSONL ground-truth evaluator;
- TP/FP/FN + precision/recall;
- per-tag/per-size/per-attribute reporting;
- tracking recall, ID switches and fragmentation;
- saved baseline/candidate comparison;
- explicit NEW FALSE NEGATIVE gate;
- benchmark docs/tests and result/video gitignore handling.

Final cleanup on this branch:

- reject JSON boolean frame values such as `true`, which Python would otherwise accept as integers;
- require `objects[].ignore` to be a real JSON boolean instead of coercing strings/numbers with `bool(...)`;
- reject duplicate non-null object IDs within the same annotated frame;
- add regression tests for all three validation failures;
- preserve benchmark result schema v1 after self-review; an attempted size-breakdown/schema-v2 cleanup was intentionally reverted because shared serialized-schema changes require project-wide coordination.

## Final net files changed relative to current integration

- `pc/spectratrack/qa_benchmark.py`
- `pc/tests/test_qa_benchmark.py`
- `docs/agents/qa.md` (this handoff only)

No detector, tracker, enhancement, performance, runtime-config, app, batch, or shared root documentation file is changed by the final net diff.

## Commits owned

Already integrated QA implementation:

- `102a6426a0329ee128a424b4277cb31237acdcb4` — squash merge: annotated QA benchmark and new-FN gate.

Current handoff branch commits after `ee76f4e0...`:

- `f5198bdf5ebad70ddd536eb9b336d69bf3426a94` — initial self-review hardening; included a schema change later reverted.
- `f8711dd4a9155a317f9202b6d9f0a2c6af34c13e` — added validation/schema tests; schema-related part later reverted.
- `049efcd8646717ff2b40bc5db62f18438c5661a7` — test formatting cleanup.
- `cfa25ead891f4c3857d9a2a7f03ca0b5ce89f29c` — restored benchmark result schema v1.
- `97bbf3259f1878f4ee97539a6d6857d01ec49149` — removed schema-only tests and left contract-safe validation fixes.

Do not cherry-pick the early intermediate commits individually; review/integrate the final branch diff as a unit.

## Tests actually run

GitHub Actions run `36144474240` on code HEAD `97bbf3259f1878f4ee97539a6d6857d01ec49149`: **PASS**.

Observed results from the Windows CI job log:

- `ruff check spectratrack tests`: **PASS** — `All checks passed!`
- `python -m compileall -q spectratrack tests`: **PASS**
- `pytest -q`: **PASS — 91 passed in 1.19s**
- `python -m spectratrack.benchmark --frames 500 --targets 24`: **PASS**
  - frames: 500
  - targets: 24
  - observations: 11970
  - elapsed: 0.363 s
  - tracker_fps: 1377.7
  - active_tracks: 24
- `python -m spectratrack.diagnostics`: **PASS**
- `python -m spectratrack.selfcheck`: **PASS — SELF_CHECK=PASS**
- PyInstaller standalone Windows build: **PASS**
- `SpectraTrack-PC.exe --help`: **PASS**
- `SpectraTrack-PC.exe batch --help`: **PASS**
- source and Windows artifacts: **uploaded successfully**

The synthetic tracker FPS above is only a CI smoke/performance number for the current run. It is not a before/after detector-quality measurement.

## Real before/after measurements

No representative real CCTV before/after benchmark was run because the repository still does not contain a validated representative annotated corpus and detector-weight bundle for that measurement.

Therefore the following remain **not measured** on real data:

- person recall / precision;
- false-negative / false-positive deltas;
- ID-switch / fragmentation deltas;
- end-to-end FPS delta on the target CCTV workload;
- DirectML/AMD VRAM delta.

Functional validation change, not a quality benchmark:

- before: JSON `"frame": true` was accepted because `bool` is an `int` subclass in Python; after: rejected with `ValueError`;
- before: `"ignore": "false"` was coerced to `True`; after: rejected with `ValueError`;
- before: duplicate object IDs in one frame were accepted; after: rejected with `ValueError`.

## Known metric limitations / unverified claims

- No representative validated real CCTV dataset has yet been established. The harness is infrastructure, not proof that one detector branch is objectively better.
- `by_size` currently reuses the generic rate shape even though false positives cannot be meaningfully assigned to a ground-truth size bucket. Treat size buckets as recall-oriented; changing/removing that serialized precision field should be a coordinated schema change, not a hidden QA-only edit.
- Object matching is greedy by IoU, not a global assignment solver. Dense/ambiguous overlaps can therefore affect which individual GT object receives a prediction.
- ID-switch and fragmentation metrics are sampled only on annotated frames; unannotated intervals are not scored.
- The QA runner exercises the standard `YoloOnnxDetector.detect()` path every frame. Branch-specific people-recall/tiled/enhanced modes are not automatically benchmarked until integration exposes/configures them through a compatible runner path.
- VRAM stays null unless supplied from a real external measurement source.
- No Android validation was needed for this PC-only QA cleanup.

## Cross-agent overlaps / conflicts

Snapshot checked against the current feature-branch diffs:

- `agent/detection @ 4b59cb68b581df6ee798491a816c91f41cb651fd`: no direct file overlap with this final QA cleanup.
- `agent/tracking @ ccf2b14e1606049a313d3471fc9e762723b1e095`: no direct file overlap with `qa_benchmark.py` or its tests; it does modify the separate synthetic `benchmark.py`.
- `agent/enhancement @ 9a9d6a65fc0487753b23c5d3d35f661f60c0d51b`: no direct file overlap with this final QA cleanup.
- `agent/performance @ 113c1d8d91922da429cd0034304ca2549380b456`: no direct file overlap with this final QA cleanup.

Semantic handoff issues outside QA scope:

- detection and enhancement both implement/affect high-recall person paths; QA cannot select between them without the same real annotated corpus and comparable runner configuration;
- tracking identity improvements still need real ID-switch/fragmentation evidence on annotated video;
- performance claims must be checked on the same workload without hiding recall regressions.

No other agent branch was merged into this branch.

## Exact integration comparison procedure

After a representative corpus is available, run the same ground truth/model/workload for baseline and candidate, then compare saved results:

```powershell
python -m spectratrack.qa_benchmark run `
  --ground-truth benchmarks/ground_truth.jsonl `
  --video-root benchmarks/videos `
  --model ..\models\yolo11n.onnx `
  --run-name BASELINE `
  --revision <baseline-sha> `
  --output benchmarks/results/BASELINE.json

python -m spectratrack.qa_benchmark run `
  --ground-truth benchmarks/ground_truth.jsonl `
  --video-root benchmarks/videos `
  --model ..\models\yolo11n.onnx `
  --run-name CANDIDATE `
  --revision <candidate-sha> `
  --output benchmarks/results/CANDIDATE.json

python -m spectratrack.qa_benchmark compare `
  --baseline benchmarks/results/BASELINE.json `
  --candidate benchmarks/results/CANDIDATE.json `
  --output benchmarks/results/BASELINE_vs_CANDIDATE.json
```

Default compare gates remain strict, including explicit NEW FALSE NEGATIVE reporting.

## Readiness

**Ready for integration: yes, for this QA validation-hardening diff.**

Reason:

- final net diff is limited to QA evaluator input validation, its regression tests, and this role state file;
- current integration base is unchanged at `ee76f4e0c9728af9f94a363c3cee682130599521`;
- branch is ahead of integration and not behind it at handoff preparation time;
- full Windows PC CI passed on the validated code HEAD;
- no production detector/tracker behavior is changed;
- no shared result schema change remains in the final diff.

This readiness does **not** mean any detection/tracking/enhancement feature branch has proven a real quality improvement. That decision remains blocked on representative annotated real data.
