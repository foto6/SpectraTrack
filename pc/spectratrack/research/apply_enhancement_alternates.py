from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from spectratrack.integrity import sha256_file

PREFUSION_SCHEMA = "spectratrack-detection-prefusion-v1"
EVIDENCE_SCHEMA = "spectratrack-vnext-enhancement-alternates-v1"


def _load_evidence(path: str | Path) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    metadata: dict[str, Any] | None = None
    alternates: list[dict[str, Any]] = []
    summary: dict[str, Any] = {}
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            if not raw.strip():
                continue
            data = json.loads(raw)
            kind = data.get("type")
            if kind == "metadata":
                if metadata is not None:
                    raise ValueError("enhancement evidence has multiple metadata records")
                if data.get("schema") != EVIDENCE_SCHEMA:
                    raise ValueError(f"unsupported enhancement evidence schema: {data.get('schema')!r}")
                metadata = dict(data)
            elif kind == "alternate":
                alternates.append(dict(data))
            elif kind == "summary":
                summary = dict(data)
            else:
                raise ValueError(f"{path}:{line_number}: unknown evidence record type {kind!r}")
    if metadata is None:
        raise ValueError("enhancement evidence is missing metadata")
    return metadata, alternates, summary


def _candidate_exact_match(candidate: dict[str, Any], support: dict[str, Any]) -> bool:
    if int(candidate.get("class_id", -1)) != int(support.get("class_id", -2)):
        return False
    if str(candidate.get("label", "")) != str(support.get("label", "")):
        return False
    if not math.isclose(
        float(candidate.get("score", -1.0)),
        float(support.get("score", -2.0)),
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        return False
    candidate_bbox = candidate.get("bbox")
    support_bbox = support.get("bbox")
    if not isinstance(candidate_bbox, list) or not isinstance(support_bbox, list):
        return False
    if len(candidate_bbox) != 4 or len(support_bbox) != 4:
        return False
    return all(
        math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=1e-9)
        for left, right in zip(candidate_bbox, support_bbox)
    )


def _bbox_iou(left: list[float], right: list[float]) -> float:
    xx1 = max(float(left[0]), float(right[0]))
    yy1 = max(float(left[1]), float(right[1]))
    xx2 = min(float(left[2]), float(right[2]))
    yy2 = min(float(left[3]), float(right[3]))
    inter = max(0.0, xx2 - xx1) * max(0.0, yy2 - yy1)
    area_left = max(0.0, float(left[2]) - float(left[0])) * max(
        0.0, float(left[3]) - float(left[1])
    )
    area_right = max(0.0, float(right[2]) - float(right[0])) * max(
        0.0, float(right[3]) - float(right[1])
    )
    return inter / max(area_left + area_right - inter, 1e-9)


def apply_alternates(
    prefusion_path: str | Path,
    evidence_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    evidence_meta, alternates, evidence_summary = _load_evidence(evidence_path)
    evidence_manifest = evidence_meta.get("roi_manifest", {})
    manifest_metadata = evidence_manifest.get("metadata", {}) if isinstance(evidence_manifest, dict) else {}
    prefusion_provenance = (
        manifest_metadata.get("prefusion", {}) if isinstance(manifest_metadata, dict) else {}
    )
    expected_prefusion_sha = prefusion_provenance.get("sha256")
    actual_prefusion_sha = sha256_file(prefusion_path)
    if expected_prefusion_sha and expected_prefusion_sha != actual_prefusion_sha:
        raise ValueError("enhancement evidence was derived from a different prefusion artifact")

    alternates_by_frame: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for alternate in alternates:
        if alternate.get("semantics", {}).get("independent_evidence_increment") != 0:
            raise ValueError("enhancement alternate must not add an independent evidence source")
        if alternate.get("semantics", {}).get("raw_corroborated") is not True:
            raise ValueError("enhancement alternate must be raw corroborated")
        key = (str(alternate["video"]), int(alternate["frame"]))
        alternates_by_frame.setdefault(key, []).append(alternate)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    seen_metadata = False
    seen_evidence_frames: set[tuple[str, int]] = set()
    replaced = 0
    appended = 0

    with Path(prefusion_path).open("r", encoding="utf-8") as source, output.open(
        "w", encoding="utf-8"
    ) as target:
        for line_number, raw in enumerate(source, start=1):
            if not raw.strip():
                continue
            data = json.loads(raw)
            kind = data.get("type")
            if kind == "metadata":
                if seen_metadata:
                    raise ValueError("prefusion dump has multiple metadata records")
                seen_metadata = True
                if data.get("schema") != PREFUSION_SCHEMA:
                    raise ValueError(f"unsupported prefusion schema: {data.get('schema')!r}")
                model_sha = data.get("model_sha256")
                evidence_model_sha = evidence_meta.get("model", {}).get("sha256")
                if model_sha and evidence_model_sha and model_sha != evidence_model_sha:
                    raise ValueError("enhancement evidence model SHA-256 differs from prefusion")
                augmented = dict(data)
                augmented["a3_enhancement"] = {
                    "operation": evidence_meta.get("operation"),
                    "evidence_sha256": sha256_file(evidence_path),
                    "roi_manifest_sha256": evidence_meta.get("roi_manifest", {}).get("sha256"),
                    "semantics": "same-source alternate measurement; no independent evidence increment",
                }
                target.write(json.dumps(augmented, sort_keys=True) + "\n")
                continue

            if kind == "frame":
                key = (str(data["video"]), int(data["frame"]))
                frame_alternates = alternates_by_frame.get(key, [])
                if frame_alternates:
                    seen_evidence_frames.add(key)
                candidates = [dict(item) for item in data.get("candidates", [])]
                used_candidate_indexes: set[int] = set()

                for alternate in frame_alternates:
                    source_kind = str(alternate["source_kind"])
                    source_id = str(alternate["source_id"])
                    raw_support = alternate.get("raw_support", [])
                    enhanced = alternate.get("candidate")
                    if not isinstance(enhanced, dict) or not isinstance(raw_support, list) or not raw_support:
                        raise ValueError(f"invalid alternate payload for frame {key}")

                    support_ranked = sorted(
                        raw_support,
                        key=lambda support: (
                            -_bbox_iou(enhanced["bbox"], support["bbox"]),
                            -float(support["score"]),
                            tuple(float(value) for value in support["bbox"]),
                        ),
                    )
                    chosen_index: int | None = None
                    chosen_support: dict[str, Any] | None = None
                    for support in support_ranked:
                        matches = [
                            index
                            for index, candidate in enumerate(candidates)
                            if index not in used_candidate_indexes
                            and candidate.get("source_kind") == source_kind
                            and candidate.get("source_id") == source_id
                            and _candidate_exact_match(candidate, support)
                        ]
                        if matches:
                            chosen_index = matches[0]
                            chosen_support = support
                            break
                    if chosen_index is None or chosen_support is None:
                        raise ValueError(
                            f"cannot find exact frozen raw support in prefusion for {key} {source_id}"
                        )

                    used_candidate_indexes.add(chosen_index)
                    candidates[chosen_index] = {
                        "bbox": [float(value) for value in enhanced["bbox"]],
                        "score": float(enhanced["score"]),
                        "class_id": int(enhanced["class_id"]),
                        "label": str(enhanced["label"]),
                        "source_kind": source_kind,
                        "source_id": source_id,
                        "source_region": [int(value) for value in alternate["source_region"]],
                    }
                    replaced += 1

                updated = dict(data)
                updated["candidates"] = candidates
                target.write(json.dumps(updated, sort_keys=True) + "\n")
                continue

            if kind == "summary":
                updated = dict(data)
                updated["a3_enhancement"] = {
                    "operation": evidence_meta.get("operation"),
                    "alternate_measurements_replaced": replaced,
                    "independent_sources_added": 0,
                    "evidence_summary": evidence_summary,
                }
                target.write(json.dumps(updated, sort_keys=True) + "\n")
                continue

            raise ValueError(f"{prefusion_path}:{line_number}: unknown record type {kind!r}")

    if not seen_metadata:
        raise ValueError("prefusion dump is missing metadata")
    missing_frames = sorted(set(alternates_by_frame) - seen_evidence_frames)
    if missing_frames:
        raise ValueError(f"prefusion dump is missing enhancement evidence frames: {missing_frames[:10]}")
    if appended != 0:
        raise AssertionError("strict alternate application must never append a new source candidate")

    return {
        "prefusion_sha256": actual_prefusion_sha,
        "evidence_sha256": sha256_file(evidence_path),
        "output_sha256": sha256_file(output),
        "alternate_measurements_replaced": replaced,
        "independent_sources_added": 0,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Apply A3 same-source alternate measurements to an A1 prefusion artifact"
    )
    parser.add_argument("--prefusion", required=True)
    parser.add_argument("--enhancement-evidence", required=True)
    parser.add_argument("--output", required=True)
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    result = apply_alternates(args.prefusion, args.enhancement_evidence, args.output)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
