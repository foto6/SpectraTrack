# Offline benchmark failure analysis

spectratrack.error_analysis is a deterministic, offline post-processing tool for the annotated QA benchmark. It does not change detector thresholds, models, enhancement, fusion, or tracking.

The intended first use is the externally frozen smoke selection. A full 5000-frame NightOwls run is not required to generate this report, and the tool refuses to pretend that an arbitrary result was scored on the supplied frame manifest.

## Inputs

Required inputs:

- canonical qa_benchmark ground-truth JSONL;
- an externally frozen frame manifest;
- a canonical qa_benchmark result JSON for the control;
- optionally, the candidate result JSON for the same selection.

The frame manifest may be JSON with a frames or selected_frames array, a bare JSON array, or JSONL. Frame rows use {"video":"clip.mp4","frame":123}. A VIDEO#FRAME string is also accepted.

For geometry-dependent attribution, pass an optional observation JSONL sidecar for each run. Each row is:

~~~json
{"video":"clip.mp4","frame":123,"width":1920,"height":1080,"predictions":[{"label":"person","score":0.31,"bbox":[812,316,827,351]}]}
~~~

The observation sidecar is analysis evidence only. It does not authorize any detector change and is not required for summary-only reports.

## Exact-selection proof

Candidate/control comparison is only accepted when the tool can prove each result refers to exactly the manifest frames. Proof is one of:

1. the manifest contains every frame in the supplied ground-truth JSONL; or
2. the result carries a selection hash equal to the raw manifest SHA-256 or normalized selected-frame-ID SHA-256.

When observation sidecars are supplied, their frame IDs must also match the manifest exactly, but a sidecar alone is not treated as proof that the result excluded other frames. A subset manifest with an old full-corpus result and no result-side selection evidence is rejected.

## Taxonomy

The report is intentionally multi-label. A single missed person may be both small and dark; category rates therefore do not need to sum to 100%.

- missed_small_person: person box height below 48 px, or explicit small/tiny/distant metadata.
- low_contrast_night_darkness: explicit dark/night/low-light/low-contrast frame or object metadata.
- occlusion_crowd: explicit occlusion/crowd metadata.
- localization_iou_miss: a same-label prediction overlaps the missed GT at IoU >= 0.10 but below the benchmark match IoU.
- duplicate_fragmented_detection: an unmatched prediction overlaps GT already matched by another prediction.
- background_false_positive: an unmatched prediction has same-label GT IoU below 0.05.
- edge_of_frame: explicit edge/truncation metadata, or a box within 2% / 4 px of an observed image edge.
- scale_extreme_aspect: aspect ratio below 0.12 or above 1.25, or observed area fraction below 0.00015 / above 0.45, or explicit metadata.
- unknown: evidence is insufficient for any explicit heuristic.

When only the aggregate QA result is available, false-positive counts remain unknown; the tool does not relabel them as background failures without geometry.

## Determinism and bounded output

The report contains SHA-256 hashes for the manifest, normalized frame IDs, ground truth, results, and optional observation sidecars. report_sha256 is computed from canonical JSON without timestamps, so identical inputs produce an identical report.

Representative IDs are deterministically ranked by a simple severity key and capped by --top-k. The hard maximum is 25 per category. The tool does not render images or contact sheets.

## Usage

From pc/:

~~~text
python -m spectratrack.error_analysis --ground-truth benchmarks/ground_truth.jsonl --manifest benchmarks/nightowls-smoke400.json --control-result benchmarks/results/CONTROL_SMOKE400.json --candidate-result benchmarks/results/CANDIDATE_SMOKE400.json --control-observations benchmarks/results/CONTROL_SMOKE400.observations.jsonl --candidate-observations benchmarks/results/CANDIDATE_SMOKE400.observations.jsonl --top-k 5 --output benchmarks/results/CONTROL_vs_CANDIDATE.error-analysis.json
~~~

The comparison emits category-count deltas plus bounded factual guidance. For example, it can say that a candidate reduced geometry-proven background false positives while recovering zero low-light GT. It does not recommend threshold retuning from the smoke set.

## Current NightOwls status

No exact A5 smoke400 manifest was available to A7 during implementation, so no real NightOwls smoke scoring is claimed here. Tests and the committed example report use only the existing tiny QA fixture. The reproducible example files are `pc/benchmarks/error_analysis.manifest.example.json`, `pc/benchmarks/error_analysis.control.example.json`, and `pc/benchmarks/error_analysis.report.example.json`. Run the real report only after the exact A5 manifest/hash is published.
